from __future__ import annotations

"""Centralized filesystem paths used by each pipeline stage.

Keeping path construction in one place avoids scattered string literals and
helps keep artifact naming conventions consistent across the project.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from src.core.config import AppConfig
from src.data_layer.symbol_master import load_tickers as load_ticker_file


@dataclass(frozen=True)
class RunPaths:
    root: Path
    raw: Path
    processed: Path
    features: Path
    runs: Path

    @staticmethod
    def from_config(cfg: AppConfig) -> "RunPaths":
        """Build all root-level paths from the loaded config."""
        root = Path(".").resolve()
        raw = root / "data" / "raw"
        processed = root / "data" / "processed"
        features = root / "data" / "features"
        runs = root / cfg.output.runs_dir
        return RunPaths(root=root, raw=raw, processed=processed, features=features, runs=runs)

    def ensure_dirs(self) -> None:
        """Create all required directories for pipeline artifacts."""
        self.raw.mkdir(parents=True, exist_ok=True)
        self.processed.mkdir(parents=True, exist_ok=True)
        self.features.mkdir(parents=True, exist_ok=True)
        self.runs.mkdir(parents=True, exist_ok=True)
        (self.runs / "models").mkdir(parents=True, exist_ok=True)
        (self.runs / "reports").mkdir(parents=True, exist_ok=True)
        (self.runs / "signals").mkdir(parents=True, exist_ok=True)
        (self.runs / "alpha_cache").mkdir(parents=True, exist_ok=True)

    def load_tickers(self, tickers_file: str | None = None) -> list[str]:
        """Load the universe list from a plain-text ticker file."""
        # user provides file (one ticker per line) to avoid brittle web scraping
        # example tickers are Yahoo-style: RELIANCE.NS
        if tickers_file is None:
            cfg_path = self.root / "config" / "config.yaml"
            if not cfg_path.exists():
                return []
            cfg_obj = yaml.safe_load(cfg_path.read_text()) or {}
            tickers_file = (cfg_obj.get("universe") or {}).get("tickers_file")

        if not tickers_file:
            return []

        p = Path(tickers_file)
        if not p.is_absolute():
            p = self.root / p
        if not p.exists():
            return []
        return load_ticker_file(str(p))

    def processed_ohlcv_path(self) -> Path:
        """Path for normalized merged OHLCV parquet."""
        return self.processed / "ohlcv.parquet"

    def features_path(self) -> Path:
        """Path for engineered feature parquet."""
        return self.features / "features.parquet"

    def alpha_library_path(self) -> Path:
        """Path for validated alpha formula library JSON."""
        return self.runs / "alpha_cache" / "alpha_library.json"

    def model_bundle_path(self) -> Path:
        """Path for serialized model + scaler bundle."""
        return self.runs / "models" / "model_bundle.joblib"

    def model_calibration_path(self) -> Path:
        """Path for probability calibration metadata."""
        return self.runs / "models" / "model_calibration.json"

    def oos_predictions_path(self) -> Path:
        """Path for walk-forward out-of-sample prediction table."""
        return self.runs / "reports" / "oos_predictions.csv"

    def monitoring_daily_path(self) -> Path:
        """Path for daily monitoring metrics CSV."""
        return self.runs / "reports" / "monitoring_daily.csv"

    def monitoring_summary_path(self) -> Path:
        """Path for monitoring summary JSON."""
        return self.runs / "reports" / "monitoring_summary.json"

    def read_processed_ohlcv(self) -> pd.DataFrame:
        """Read merged OHLCV dataset."""
        return pd.read_parquet(self.processed_ohlcv_path())

    def read_features(self) -> pd.DataFrame:
        """Read feature dataset."""
        return pd.read_parquet(self.features_path())

    def read_alpha_library(self):
        """Read alpha library JSON into typed model."""
        import json
        from src.llm_alpha.alpha_whitelist import AlphaLibrary

        return AlphaLibrary.model_validate(json.loads(self.alpha_library_path().read_text()))

    def load_model_bundle(self):
        """Load trained model bundle using torch serialization."""
        import torch
        # Load using torch instead of joblib
        return torch.load(self.model_bundle_path(), map_location="cpu", weights_only=False)

    def load_model_calibration(self):
        """Load calibration JSON if present; otherwise return None."""
        import json
        p = self.model_calibration_path()
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def signals_csv_path(self) -> Path:
        """Stable 'latest' signals path.

        This file is overwritten each run and is useful for dashboards that
        always read a fixed filename.
        """
        return self.runs / "signals" / "signals_top_10.csv"

    def signals_csv_path_for_date(self, signal_date: pd.Timestamp | str) -> Path:
        """Date-specific signals filename.

        Example output: data/runs/signals/signals_20260227.csv
        """
        dt = pd.Timestamp(signal_date)
        return self.runs / "signals" / f"signals_{dt.strftime('%Y%m%d')}.csv"
