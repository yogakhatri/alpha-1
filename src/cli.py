"""Command-line entrypoint for the full research pipeline.

Each subcommand maps to one pipeline stage so users can run stages independently
or end-to-end with clear reproducible steps.
"""

import argparse

from src.core.config import load_config
from src.core.logging import get_logger
from src.core.paths import RunPaths

from src.data_layer.bhavcopy_loader import download_bhavcopy_range, load_bhavcopy_ohlcv
from src.data_layer.yfinance_loader import download_yfinance_range, load_yfinance_ohlcv
from src.data_layer.ohlcv_normalize import merge_primary_with_fallback
from src.core.data_quality import assert_fresh_enough
from src.features.feature_store import build_feature_table
from src.llm_alpha.alpha_selection import build_or_load_alpha_library
from src.models.train import train_walk_forward
from src.backtest.engine import run_backtest
from src.signals.generate_signals import generate_daily_signals


log = get_logger(__name__)


def cmd_download_data(cfg_path: str) -> None:
    """Download raw data and build the normalized OHLCV parquet."""
    cfg = load_config(cfg_path)
    paths = RunPaths.from_config(cfg)

    tickers = paths.load_tickers(cfg.universe.tickers_file)
    paths.ensure_dirs()

    if cfg.data.use_bhavcopy:
        download_bhavcopy_range(cfg, paths)

    if cfg.data.use_yfinance_fallback:
        download_yfinance_range(cfg, paths, tickers)

    bhav = load_bhavcopy_ohlcv(cfg, paths)
    yfin = load_yfinance_ohlcv(cfg, paths)

    merged = merge_primary_with_fallback(primary=bhav, fallback=yfin)
    if tickers:
        merged = merged[merged["ticker"].isin(set(tickers))].copy()
    merged = merged.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)
    latest = assert_fresh_enough(
        merged,
        cfg_end_date=cfg.data.end_date,
        max_stale_trading_days=cfg.data.max_stale_trading_days,
        dataset_name="processed OHLCV",
        date_col="date",
    )

    out_path = paths.processed_ohlcv_path()
    merged.to_parquet(out_path, index=False)
    log.info("Saved processed OHLCV: %s (latest=%s)", out_path, latest.date())


def cmd_build_features(cfg_path: str) -> None:
    """Build the model feature table from processed OHLCV."""
    cfg = load_config(cfg_path)
    paths = RunPaths.from_config(cfg)
    paths.ensure_dirs()

    ohlcv = paths.read_processed_ohlcv()
    assert_fresh_enough(
        ohlcv,
        cfg_end_date=cfg.data.end_date,
        max_stale_trading_days=cfg.data.max_stale_trading_days,
        dataset_name="input OHLCV",
        date_col="date",
    )
    feats = build_feature_table(cfg, ohlcv)
    latest = assert_fresh_enough(
        feats,
        cfg_end_date=cfg.data.end_date,
        max_stale_trading_days=cfg.data.max_stale_trading_days,
        dataset_name="feature table",
        date_col="date",
    )
    out_path = paths.features_path()
    feats.to_parquet(out_path, index=False)
    log.info("Saved features: %s (latest=%s)", out_path, latest.date())


def cmd_generate_alphas(cfg_path: str) -> None:
    """Generate (or load) validated alpha expressions and save library JSON."""
    cfg = load_config(cfg_path)
    paths = RunPaths.from_config(cfg)
    paths.ensure_dirs()

    feats = paths.read_features()
    alpha_lib = build_or_load_alpha_library(cfg, paths, feats)
    out_path = paths.alpha_library_path()
    out_path.write_text(alpha_lib.model_dump_json(indent=2))
    log.info("Saved alpha library: %s", out_path)


def cmd_train(cfg_path: str) -> None:
    """Train model using configured walk-forward procedure."""
    cfg = load_config(cfg_path)
    paths = RunPaths.from_config(cfg)
    paths.ensure_dirs()

    feats = paths.read_features()
    assert_fresh_enough(
        feats,
        cfg_end_date=cfg.data.end_date,
        max_stale_trading_days=cfg.data.max_stale_trading_days,
        dataset_name="training features",
        date_col="date",
    )
    alpha_lib = paths.read_alpha_library()
    model_artifact = train_walk_forward(cfg, paths, feats, alpha_lib)
    log.info("Saved model artifact: %s", model_artifact)


def cmd_backtest(cfg_path: str) -> None:
    """Run historical backtest simulation and save performance reports."""
    cfg = load_config(cfg_path)
    paths = RunPaths.from_config(cfg)
    paths.ensure_dirs()

    feats = paths.read_features()
    assert_fresh_enough(
        feats,
        cfg_end_date=cfg.data.end_date,
        max_stale_trading_days=cfg.data.max_stale_trading_days,
        dataset_name="backtest features",
        date_col="date",
    )
    ohlcv = paths.read_processed_ohlcv()
    alpha_lib = paths.read_alpha_library()
    model_bundle = paths.load_model_bundle()

    report_paths = run_backtest(cfg, paths, feats, ohlcv, alpha_lib, model_bundle)
    log.info("Backtest done: %s", report_paths)


def cmd_daily_signals(cfg_path: str) -> None:
    """Generate current-day signal file(s) from the latest trained model."""
    cfg = load_config(cfg_path)
    paths = RunPaths.from_config(cfg)
    paths.ensure_dirs()

    feats = paths.read_features()
    assert_fresh_enough(
        feats,
        cfg_end_date=cfg.data.end_date,
        max_stale_trading_days=cfg.data.max_stale_trading_days,
        dataset_name="signal features",
        date_col="date",
    )
    ohlcv = paths.read_processed_ohlcv()
    alpha_lib = paths.read_alpha_library()
    model_bundle = paths.load_model_bundle()

    out_csv = generate_daily_signals(cfg, paths, feats, ohlcv, alpha_lib, model_bundle)
    log.info("Signals written: %s", out_csv)


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI parser with all supported pipeline commands."""
    p = argparse.ArgumentParser(prog="nse-swing")
    p.add_argument("--config", required=True)
    sp = p.add_subparsers(dest="cmd", required=True)

    sp.add_parser("download-data")
    sp.add_parser("build-features")
    sp.add_parser("generate-alphas")
    sp.add_parser("train")
    sp.add_parser("backtest")
    sp.add_parser("daily-signals")
    return p


def main() -> None:
    """Parse arguments and dispatch to the selected command handler."""
    p = build_parser()
    args = p.parse_args()

    if args.cmd == "download-data":
        cmd_download_data(args.config)
    elif args.cmd == "build-features":
        cmd_build_features(args.config)
    elif args.cmd == "generate-alphas":
        cmd_generate_alphas(args.config)
    elif args.cmd == "train":
        cmd_train(args.config)
    elif args.cmd == "backtest":
        cmd_backtest(args.config)
    elif args.cmd == "daily-signals":
        cmd_daily_signals(args.config)
    else:
        raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
