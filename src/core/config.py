from __future__ import annotations

"""Typed configuration schema for the full pipeline.

The project reads YAML config files and validates them against these models so
invalid or missing settings fail early with explicit errors.
"""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class ProjectCfg(BaseModel):
    """Project-level metadata and reproducibility settings."""
    run_name: str
    timezone: str = "Asia/Kolkata"
    seed: int = 42


class UniverseCfg(BaseModel):
    """Stock-universe controls and liquidity gates."""
    tickers_file: str
    min_price: float = 0.0
    min_median_turnover_20d: float = 0.0


class DataCfg(BaseModel):
    """Raw data download/date range settings."""
    start_date: str
    end_date: str = ""  # Empty string means "use today's date"
    use_bhavcopy: bool = True
    use_yfinance_fallback: bool = True
    max_stale_trading_days: int = 5


class LabelCfg(BaseModel):
    """Barrier-label definition used as supervised learning target."""
    horizon_days: int = 3
    profit_take_pct: float = 0.03
    stop_loss_pct: float = 0.015
    same_day_both_hit_rule: Literal["STOP_FIRST", "TARGET_FIRST"] = "STOP_FIRST"


class FeaturesCfg(BaseModel):
    """Feature engineering settings."""
    lookback_days_for_model: int = 60
    compute_indicators: bool = True


class OpenAICompatCfg(BaseModel):
    """Connection settings for OpenAI-compatible LLM providers."""
    base_url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"


class LLMAlphaCfg(BaseModel):
    """Controls for alpha-expression generation and validation."""
    enabled: bool = True
    k_alphas: int = 20
    max_expr_chars: int = 200
    max_rolling_window: int = 60
    provider: Literal["manual", "openai_compatible"] = "manual"
    cache_dir: str = "data/runs/alpha_cache"
    openai_compatible: Optional[OpenAICompatCfg] = None


class ModelCfg(BaseModel):
    """Transformer architecture and optimizer hyperparameters."""
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    lr: float = 5e-4
    weight_decay: float = 1e-2
    batch_size: int = 256
    max_epochs: int = 30
    early_stop_patience: int = 5
    use_mps_if_available: bool = True
    use_cuda_if_available: bool = True
    torch_compile: bool = True          # torch.compile for PyTorch 2+ speed-up


class WalkForwardCfg(BaseModel):
    """Temporal split sizes for walk-forward training/evaluation."""
    train_days: int = 756
    val_days: int = 63
    test_days: int = 63
    step_days: int = 21
    max_folds: Optional[int] = None


class BacktestCfg(BaseModel):
    """Backtest settings container."""
    walk_forward: WalkForwardCfg


class ExecutionCfg(BaseModel):
    """Signal timing convention used by training/inference/backtest."""
    signal_timing: Literal["PREOPEN_SAME_DAY", "EOD_NEXT_OPEN"] = "EOD_NEXT_OPEN"


class DecisionCfg(BaseModel):
    """Decision-layer filters and ranking constraints for final picks."""
    max_picks_per_day: int = 2
    min_score_threshold: float = 0.0
    min_calibrated_prob: float = 0.0
    min_expected_value_pct: float = 0.0
    rank_by: Literal["score", "expected_value"] = "expected_value"
    target_top1_precision: float = 0.30
    min_days_for_threshold: int = 30
    max_reentries_per_ticker_20d: int = 2
    reentry_cooldown_days: int = 3
    no_trade_if_below_threshold: bool = True


class PortfolioCfg(BaseModel):
    """Portfolio-level risk and position sizing controls."""
    capital_inr: float = 100000.0
    max_open_positions: int = 5
    max_new_positions_per_day: int = 2
    risk_per_trade_pct: float = 0.005
    slot_capital_fraction: float = 0.20


class GrowwBrokerageCfg(BaseModel):
    """Brokerage toggle for cost simulation."""
    enabled: bool = True


class CostsCfg(BaseModel):
    """Execution cost assumptions used in backtest and EV filters."""
    slippage_bps_per_side: float = 15.0
    regulatory_bps_per_side: float = 5.0
    groww_brokerage: GrowwBrokerageCfg = Field(default_factory=GrowwBrokerageCfg)


class OutputCfg(BaseModel):
    """Artifact output directory and signal-output size settings."""
    runs_dir: str = "data/runs"
    signals_top_n: int = 10


class MonitoringCfg(BaseModel):
    """Settings for rolling quality metrics and drift alerts."""
    rolling_short_days: int = 20
    rolling_long_days: int = 60
    drift_alert_drop: float = 0.08


class AppConfig(BaseModel):
    """Root typed config object assembled from all nested sections."""
    project: ProjectCfg
    universe: UniverseCfg
    data: DataCfg
    label: LabelCfg
    features: FeaturesCfg
    llm_alpha: LLMAlphaCfg
    model: ModelCfg
    backtest: BacktestCfg
    execution: ExecutionCfg = Field(default_factory=ExecutionCfg)
    decision: DecisionCfg = Field(default_factory=DecisionCfg)
    portfolio: PortfolioCfg
    costs: CostsCfg
    monitoring: MonitoringCfg = Field(default_factory=MonitoringCfg)
    output: OutputCfg


def load_config(path: str) -> AppConfig:
    """Load YAML config and validate it against AppConfig schema.

    If data.end_date is empty or 'today', it is replaced with today's date.
    """
    import datetime
    p = Path(path)
    obj = yaml.safe_load(p.read_text())
    # Dynamic end_date: use today if not explicitly set
    data_cfg = obj.get("data", {})
    end = data_cfg.get("end_date", "")
    if not end or str(end).strip().lower() == "today":
        data_cfg["end_date"] = datetime.date.today().isoformat()
        obj["data"] = data_cfg
    return AppConfig.model_validate(obj)


def pct_to_fraction(value: float) -> float:
    """Accept 0.06 or 6.0 and return a decimal fraction."""
    v = float(value)
    if v < 0:
        raise ValueError("Percentage value cannot be negative")
    return v / 100.0 if v > 1.0 else v
