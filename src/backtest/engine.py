"""Portfolio backtest engine with score/probability/EV based entry filters.

The engine re-scores historical windows, maps predictions to execution dates,
simulates position lifecycle with TP/SL/time exits, and reports strategy stats.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

from src.core.config import AppConfig, pct_to_fraction
from src.core.device import (
    inference_context,
    log_device_info,
    make_amp_context,
    optimal_batch_size,
    optimal_num_workers,
    pin_memory_for,
    resolve_device,
    try_compile,
    tune_for_device,
)
from src.core.paths import RunPaths
from src.backtest.costs import CostParams, total_cost_one_side
from src.models.calibration import apply_calibration
from src.models.windowing import build_inference_windows


def _estimated_round_trip_cost_fraction(cfg: AppConfig, slot_size: float) -> float:
    """Estimate round-trip cost fraction used in EV ranking filter."""
    bps = 2.0 * (float(cfg.costs.slippage_bps_per_side) + float(cfg.costs.regulatory_bps_per_side)) / 10000.0
    brokerage_frac = 0.0
    if cfg.costs.groww_brokerage.enabled and slot_size > 0:
        brokerage_frac = (2.0 * 20.0) / float(slot_size)
    return float(bps + brokerage_frac)


def _expected_value_pct(prob: np.ndarray, tp_pct: float, sl_pct: float, cost_frac: float) -> np.ndarray:
    """Expected return proxy per trade candidate after approximate costs."""
    p = np.asarray(prob, dtype=float)
    ev = p * tp_pct - (1.0 - p) * sl_pct - cost_frac
    return ev.astype(float)


def run_backtest(cfg: AppConfig, paths: RunPaths, feats: pd.DataFrame, ohlcv: pd.DataFrame, alpha_lib, model_bundle) -> str:
    """Run full historical simulation and write backtest reports.

    Returns:
        str: CSV path of summary stats.
    """
    from src.core.logging import get_logger
    log = get_logger(__name__)
    log.info("Starting V2 Portfolio Backtest with Optimizations...")

    model = model_bundle.model
    scaler = model_bundle.scaler
    feature_cols = model_bundle.feature_cols
    lookback = int(model_bundle.lookback)
    signal_timing = getattr(model_bundle, "signal_timing", cfg.execution.signal_timing)
    if signal_timing != cfg.execution.signal_timing:
        log.warning(
            "Config signal_timing=%s differs from model timing=%s. Using model timing.",
            cfg.execution.signal_timing,
            signal_timing,
        )

    df = feats.copy()

    # 1. Normalize dataframe to have ticker and date columns
    if isinstance(df.index, pd.MultiIndex):
        idx_names = list(df.index.names)
        if "ticker" in idx_names or "date" in idx_names:
            df = df.reset_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "date"})

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    tickers = paths.load_tickers(cfg.universe.tickers_file)
    if tickers:
        df = df[df["ticker"].isin(set(tickers))].copy()
    if cfg.universe.min_price > 0 and "Close" in df.columns:
        df = df[df["Close"] >= float(cfg.universe.min_price)].copy()
    if cfg.universe.min_median_turnover_20d > 0 and "turnover_med_20d" in df.columns:
        df = df[df["turnover_med_20d"] >= float(cfg.universe.min_median_turnover_20d)].copy()

    # 2. Compute Alphas dynamically using canonical safe executor
    if alpha_lib and hasattr(alpha_lib, "formulas") and alpha_lib.formulas:
        log.info("Computing alphas on historical data...")
        from src.llm_alpha.alpha_executor import compute_alphas_on_df
        df = compute_alphas_on_df(
            df,
            formulas=alpha_lib.formulas,
            feature_names=alpha_lib.feature_names,
            max_chars=200,
            max_window=60,
            logger=log,
        )

    # 3. Build Historical Windows for Inference
    log.info("Scaling and building sliding windows...")
    df = df.dropna(subset=feature_cols).copy()
    raw_price_df = df[["ticker", "date", "Open", "High", "Low", "Close"]].copy()
    df[feature_cols] = scaler.transform(df[feature_cols])

    windows, tickers, pred_dates, exec_dates = build_inference_windows(
        df=df,
        feature_cols=feature_cols,
        lookback=lookback,
        signal_timing=signal_timing,
        require_execution_day=True,
    )

    if not windows:
        raise ValueError("No historical windows could be built.")

    X = torch.tensor(np.stack(windows), dtype=torch.float32)
    ds = TensorDataset(X)

    # 4. Generate Predictions
    device = resolve_device(
        prefer_cuda=getattr(cfg.model, "use_cuda_if_available", True),
        prefer_mps=getattr(cfg.model, "use_mps_if_available", True),
    )
    tune_for_device(device)
    log_device_info(device, log)

    bs = optimal_batch_size(device)["inference"]
    pm = pin_memory_for(device)
    nw = optimal_num_workers(device)
    autocast_fn, _ = make_amp_context(device)

    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pm,
        persistent_workers=(nw > 0),
    )

    if getattr(cfg.model, "torch_compile", True):
        model = try_compile(model.to(device), device)
    else:
        model = model.to(device)
    model.eval()

    preds = []
    log.info("Running AI inference on history...")
    with inference_context():
        for (bx,) in loader:
            bx = bx.to(device, non_blocking=pm)
            with autocast_fn():
                out = model(bx)
            probs = torch.sigmoid(out).float().cpu().numpy().reshape(-1)
            preds.extend(probs.tolist())

    calibration = paths.load_model_calibration()
    calibrated = apply_calibration(np.asarray(preds, dtype=float), calibration)

    pred_df = pd.DataFrame(
        {
            "ticker": tickers,
            "prediction_date": pd.to_datetime(pred_dates),
            "execution_date": pd.to_datetime(exec_dates),
            "score": preds,
            "calibrated_prob": calibrated,
        }
    )
    pred_df = pred_df.dropna(subset=["execution_date"]).copy()

    # Merge predictions with actual price data for simulation
    price_df = raw_price_df.rename(columns={"date": "execution_date"})
    sim_df = pd.merge(pred_df, price_df, on=["ticker", "execution_date"], how="left")
    sim_df = sim_df.sort_values("execution_date")

    # 5. Portfolio Simulation Engine
    log.info("Simulating Portfolio execution (TP/SL) with strict AI thresholds...")
    capital = cfg.portfolio.capital_inr
    max_slots = cfg.portfolio.max_open_positions
    slot_size = capital * cfg.portfolio.slot_capital_fraction

    tp_pct = pct_to_fraction(cfg.label.profit_take_pct)
    sl_pct = pct_to_fraction(cfg.label.stop_loss_pct)
    max_hold_days = max(1, int(cfg.label.horizon_days))
    stop_first = cfg.label.same_day_both_hit_rule == "STOP_FIRST"
    min_score = float(cfg.decision.min_score_threshold)
    min_prob = float(cfg.decision.min_calibrated_prob)
    if calibration and cfg.decision.no_trade_if_below_threshold:
        min_prob = max(min_prob, float(calibration.get("top1_threshold", 0.0)))
    est_cost_frac = _estimated_round_trip_cost_fraction(cfg, slot_size=max(slot_size, 1.0))
    sim_df["expected_value_pct"] = _expected_value_pct(
        prob=sim_df["calibrated_prob"].to_numpy(dtype=float),
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        cost_frac=est_cost_frac,
    )
    rank_col = "expected_value_pct" if cfg.decision.rank_by == "expected_value" else "calibrated_prob"
    cost_params = CostParams(
        slippage_bps_per_side=cfg.costs.slippage_bps_per_side,
        regulatory_bps_per_side=cfg.costs.regulatory_bps_per_side,
        groww_brokerage_enabled=cfg.costs.groww_brokerage.enabled,
    )

    open_positions = []  # dicts: ticker, entry_price, shares, date_entered
    equity_curve = []    # tuples: date, total_value
    trade_log = []       # Record of closed trades
    last_exit_day_idx: dict[str, int] = {}
    entry_day_indices: dict[str, list[int]] = defaultdict(list)

    unique_dates = sim_df["execution_date"].unique()

    for day_idx, current_date in enumerate(unique_dates):
        day_data = sim_df[sim_df["execution_date"] == current_date]

        # A. Check Exits for Open Positions
        surviving_positions = []
        for pos in open_positions:
            ticker = pos["ticker"]
            tkr_data = day_data[day_data["ticker"] == ticker]
            if tkr_data.empty:
                surviving_positions.append(pos)
                continue

            high = tkr_data["High"].values[0]
            low = tkr_data["Low"].values[0]
            close = tkr_data["Close"].values[0]

            entry = pos["entry_price"]
            tp_price = entry * (1 + tp_pct)
            sl_price = entry * (1 - sl_pct)
            pos["days_held"] = int(pos.get("days_held", 0)) + 1

            exited = False
            exit_price = 0

            if low <= sl_price and high >= tp_price:
                if stop_first:
                    exited = True
                    exit_price = sl_price
                else:
                    exited = True
                    exit_price = tp_price
            elif low <= sl_price:
                exited = True
                exit_price = sl_price
            elif high >= tp_price:
                exited = True
                exit_price = tp_price
            elif pos["days_held"] >= max_hold_days:
                exited = True
                exit_price = close

            if exited:
                gross_exit = float(pos["shares"] * exit_price)
                exit_cost = total_cost_one_side(gross_exit, cost_params)
                net_exit = gross_exit - exit_cost
                capital += net_exit
                pnl = net_exit - pos["cost_basis"]
                trade_log.append({"ticker": ticker, "entry": entry, "exit": exit_price, "pnl": pnl, "win": pnl > 0})
                last_exit_day_idx[ticker] = day_idx
            else:
                pos["current_value"] = pos["shares"] * close
                surviving_positions.append(pos)

        open_positions = surviving_positions

        # B. Enter New Positions
        open_slots = max_slots - len(open_positions)
        if open_slots > 0:
            high_conviction_data = day_data[
                (day_data["score"] >= min_score)
                & (day_data["calibrated_prob"] >= min_prob)
                & (day_data["expected_value_pct"] >= float(cfg.decision.min_expected_value_pct))
            ]
            top_candidates = high_conviction_data.sort_values(rank_col, ascending=False).head(cfg.output.signals_top_n)
            day_entries = 0
            max_new = min(int(cfg.portfolio.max_new_positions_per_day), int(cfg.decision.max_picks_per_day))

            for _, row in top_candidates.iterrows():
                if open_slots <= 0:
                    break
                if day_entries >= max_new:
                    break

                tkr = str(row["ticker"])
                if any(p["ticker"] == tkr for p in open_positions):
                    continue
                cooldown = int(cfg.decision.reentry_cooldown_days)
                last_exit_idx = last_exit_day_idx.get(tkr)
                if last_exit_idx is not None and (day_idx - last_exit_idx) <= cooldown:
                    continue
                hist = [x for x in entry_day_indices.get(tkr, []) if (day_idx - x) < 20]
                if len(hist) >= int(cfg.decision.max_reentries_per_ticker_20d):
                    continue
                if capital < slot_size:
                    break

                entry_price = float(row["Open"])
                if entry_price <= 0:
                    continue

                risk_capital = capital * float(cfg.portfolio.risk_per_trade_pct)
                risk_notional = slot_size if sl_pct <= 0 else (risk_capital / sl_pct)
                target_notional = min(slot_size, risk_notional, capital)

                shares = int(target_notional // entry_price)
                if shares > 0:
                    gross_entry = float(shares * entry_price)
                    entry_cost = total_cost_one_side(gross_entry, cost_params)
                    cost_basis = gross_entry + entry_cost
                    if cost_basis > capital:
                        continue
                    capital -= cost_basis
                    open_positions.append({
                        "ticker": tkr, "entry_price": entry_price,
                        "shares": shares, "cost_basis": cost_basis, "current_value": gross_entry, "days_held": 0
                    })
                    entry_day_indices[tkr].append(day_idx)
                    open_slots -= 1
                    day_entries += 1

        # C. Record Daily Equity
        portfolio_val = capital + sum(p.get("current_value", p["cost_basis"]) for p in open_positions)
        equity_curve.append({"date": current_date, "equity": portfolio_val})

    # 6. Compute Metrics
    eq_df = pd.DataFrame(equity_curve).set_index("date")
    eq_df["returns"] = eq_df["equity"].pct_change()
    eq_df["drawdown"] = (eq_df["equity"] / eq_df["equity"].cummax()) - 1

    total_return = (eq_df["equity"].iloc[-1] / cfg.portfolio.capital_inr) - 1
    max_dd = eq_df["drawdown"].min()
    win_rate = sum(1 for t in trade_log if t["win"]) / max(1, len(trade_log))
    ret_std = eq_df["returns"].std()
    sharpe = (eq_df["returns"].mean() / ret_std) * np.sqrt(252) if len(eq_df) > 1 and ret_std and not np.isnan(ret_std) else 0.0

    log.info(f"Backtest Complete. Total Return: {total_return:.2%}, Win Rate: {win_rate:.2%}, Max DD: {max_dd:.2%}, Sharpe: {sharpe:.2f}")

    # 7. Save Artifacts
    report_dir = paths.runs / "reports"
    report_dir.mkdir(exist_ok=True)

    stats_df = pd.DataFrame([{
        "Total_Return": f"{total_return:.2%}",
        "Win_Rate": f"{win_rate:.2%}",
        "Total_Trades": len(trade_log),
        "Max_Drawdown": f"{max_dd:.2%}",
        "Sharpe_Ratio": round(sharpe, 2),
        "Signal_Timing": signal_timing,
        "Min_Raw_Score": round(min_score, 6),
        "Min_Calibrated_Prob": round(min_prob, 6),
        "Rank_By": rank_col,
    }])
    csv_path = report_dir / "backtest_stats.csv"
    stats_df.to_csv(csv_path, index=False)

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(eq_df.index, eq_df["equity"], label="Portfolio Strategy")
        plt.title(f"Swing Trading Equity Curve (Win Rate: {win_rate:.1%})")
        plt.ylabel("Capital (INR)")
        plt.grid(True)
        plt_path = report_dir / "equity_curve.png"
        plt.savefig(plt_path)
    except Exception as e:
        log.warning("Could not save equity curve plot: %s", e)

    return str(csv_path)
