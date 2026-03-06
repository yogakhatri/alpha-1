# NSE Swing Research (Signals Only)

This project creates **NSE swing-trading candidate lists** (not auto-execution).
It is designed to help pick **1-2 high-conviction stocks** for a short holding window (1-5 trading days).

## What It Does
1. Downloads and merges EOD OHLCV data (NSE bhavcopy primary, yfinance fallback).
2. Builds technical + alpha features.
3. Trains a transformer model with walk-forward validation.
4. Calibrates model probabilities and derives confidence thresholds from out-of-sample data.
5. Generates daily signals with EV/risk filters.
6. Runs backtests with costs and position constraints.

## Project Structure
- `config/config.yaml`: main configuration.
- `src/cli.py`: entrypoint for all commands.
- `data/processed/ohlcv.parquet`: merged market data.
- `data/features/features.parquet`: model feature table.
- `data/runs/models/model_bundle.joblib`: trained model bundle.
- `data/runs/models/model_calibration.json`: calibration + selected threshold.
- `data/runs/signals/signals_YYYYMMDD.csv`: date-stamped final signals (historical snapshots).
- `data/runs/signals/signals_top_10.csv`: latest signal output (overwritten every run).
- `data/runs/reports/backtest_stats.csv`: backtest summary.
- `data/runs/reports/oos_predictions.csv`: walk-forward out-of-sample predictions.
- `data/runs/reports/monitoring_daily.csv`: daily quality metrics and drift flags.
- `data/runs/reports/monitoring_summary.json`: monitoring summary snapshot.

## Setup
```bash
cd /Users/yk/work/alpha-1
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Full Run (First Time / Rebuild)
```bash
cd /Users/yk/work/alpha-1
source .venv/bin/activate

PYTHONPATH=. python -m src.cli --config config/config.yaml download-data
PYTHONPATH=. python -m src.cli --config config/config.yaml build-features

# Alpha generation in manual mode:
# 1) run once to write prompt
# 2) paste alpha JSON into data/runs/alpha_cache/alpha_response.json
# 3) run again to save alpha library
PYTHONPATH=. python -m src.cli --config config/config.yaml generate-alphas

PYTHONPATH=. python -m src.cli --config config/config.yaml train
PYTHONPATH=. python -m src.cli --config config/config.yaml backtest
PYTHONPATH=. python -m src.cli --config config/config.yaml daily-signals
```

## Daily Routine (No Retraining)
```bash
cd /Users/yk/work/alpha-1
source .venv/bin/activate

PYTHONPATH=. python -m src.cli --config config/config.yaml download-data
PYTHONPATH=. python -m src.cli --config config/config.yaml build-features
PYTHONPATH=. python -m src.cli --config config/config.yaml daily-signals
```

## Important Runtime Rule
Run only one training process at a time. If you change config for a new training run, stop any old training process first.

## Main Factors That Affect Output
1. **Data freshness and completeness**
   - `data.end_date`, `data.max_stale_trading_days`.
   - If data is stale, pipeline now fails fast.
2. **Universe and liquidity filters**
   - `universe.tickers_file`, `min_price`, `min_median_turnover_20d`.
   - Wider universe gives more ideas but may reduce precision.
3. **Label design (target definition)**
   - `label.horizon_days`, `profit_take_pct`, `stop_loss_pct`, `same_day_both_hit_rule`.
   - This directly defines what the model learns as a "successful" trade.
4. **Signal timing**
   - `execution.signal_timing`.
   - Must match your operational behavior (EOD decision vs pre-open decision).
5. **Walk-forward setup**
   - `backtest.walk_forward.train_days/val_days/test_days/step_days/max_folds`.
   - Controls robustness and regime adaptation.
6. **Model complexity and training behavior**
   - `model.d_model`, `n_layers`, `dropout`, `lr`, `batch_size`, `max_epochs`, `early_stop_patience`.
7. **Alpha quality**
   - `llm_alpha` formulas and safety constraints.
   - Better alpha candidates improve ranking quality.
8. **Decision and ranking filters**
   - `decision.min_score_threshold`, `min_calibrated_prob`, `min_expected_value_pct`.
   - `decision.rank_by` (`expected_value` vs raw probability).
   - `decision.max_picks_per_day`, cooldown and re-entry limits.
9. **Transaction costs**
   - `costs.slippage_bps_per_side`, `regulatory_bps_per_side`, brokerage flag.
   - Higher costs reduce true edge and should be modeled conservatively.
10. **Drift and monitoring**
   - `monitoring.rolling_short_days`, `rolling_long_days`, `drift_alert_drop`.
   - Use these files to pause/review strategy when hit-rate degrades.

## Interpreting Output
1. Check `signals_YYYYMMDD.csv` and `signals_top_10.csv`:
   - If empty, model found no high-confidence setup for the day (valid no-trade day).
2. Check `backtest_stats.csv` after retraining:
   - Review trade count, win rate, drawdown, Sharpe with costs.
3. Check `monitoring_summary.json`:
   - If drift alerts rise, retrain/revalidate before relying on signals.

## Notes
- This is a research decision-support tool, not guaranteed profit.
- Real-money usage requires strict position sizing and risk controls.
