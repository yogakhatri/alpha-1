from __future__ import annotations

"""Safe alpha expression executor on pandas dataframes."""

import ast
from typing import Callable

import numpy as np
import pandas as pd

from src.llm_alpha.alpha_parser import validate_expression


def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-12) -> pd.Series:
    """Division helper that avoids exploding near-zero denominators."""
    denom = b.abs().clip(lower=eps)
    return a / denom


def _zscore(x: pd.Series, w: int) -> pd.Series:
    """Rolling z-score helper used by expression runtime."""
    m = x.rolling(w).mean()
    s = x.rolling(w).std().replace(0.0, np.nan)
    return (x - m) / s


SAFE_FUNCS: dict[str, Callable] = {
    "shift": lambda x, n: x.shift(int(n)),
    "rolling_mean": lambda x, w: x.rolling(int(w)).mean(),
    "rolling_std": lambda x, w: x.rolling(int(w)).std(),
    "rolling_min": lambda x, w: x.rolling(int(w)).min(),
    "rolling_max": lambda x, w: x.rolling(int(w)).max(),
    "ewm_mean": lambda x, span: x.ewm(span=int(span), adjust=False).mean(),
    "zscore": _zscore,
    "delta": lambda x, n: x - x.shift(int(n)),
    "safe_div": _safe_div,
    "clip": lambda x, lo, hi: x.clip(lower=float(lo), upper=float(hi)),
    "abs": lambda x: x.abs(),
    "sign": lambda x: np.sign(x),
    "log1p": lambda x: np.log1p(x.clip(lower=-0.99)),
}


def compute_alpha(
    df: pd.DataFrame,
    expr: str,
    feature_names: list[str],
    max_chars: int,
    max_window: int | None = None,
) -> pd.Series:
    """Validate and execute one alpha expression on dataframe columns."""
    vr = validate_expression(expr, feature_names=feature_names, max_chars=max_chars, max_window=max_window)
    if not vr.ok:
        raise ValueError(f"Alpha rejected: {vr.reason}")

    tree = ast.parse(expr, mode="eval")
    code = compile(tree, "<alpha>", "eval")

    local_env = {name: df[name] for name in feature_names if name in df.columns}
    local_env.update(SAFE_FUNCS)
    local_env["eps"] = 1e-12

    out = eval(code, {"__builtins__": {}}, local_env)  # noqa: S307 (AST-validated)
    if isinstance(out, (pd.Series, np.ndarray)):
        s = pd.Series(out, index=df.index)
    else:
        s = pd.Series([out] * len(df), index=df.index)

    s = s.replace([np.inf, -np.inf], np.nan)
    return s.astype(float)


def compute_alphas_on_df(
    df: pd.DataFrame,
    formulas: dict[str, str],
    feature_names: list[str],
    max_chars: int = 200,
    max_window: int | None = 60,
    logger=None,
) -> pd.DataFrame:
    """Compute all alpha expressions on a DataFrame, grouped by ticker.

    This is the single canonical entry-point used by training, backtest,
    and signal generation. Each alpha is evaluated per-ticker so that
    rolling windows never cross ticker boundaries, using the same safe
    executor (builtins suppressed, AST-validated) as alpha selection.

    Args:
        df: Must have columns 'ticker', 'date', and all feature_names.
            Should be sorted by ['ticker', 'date'].
        formulas: {alpha_name: expression_string} from alpha library.
        feature_names: Base feature column names available in df.
        max_chars: Max expression length for validation.
        max_window: Max rolling window size for validation.
        logger: Optional logger for warnings.

    Returns:
        df with new alpha columns added in-place.
    """
    for alpha_name, formula in formulas.items():
        if alpha_name in df.columns:
            continue
        try:
            s = df.groupby("ticker", group_keys=False).apply(
                lambda g: compute_alpha(g, formula, feature_names, max_chars, max_window)
            )
            # Realign index after groupby apply
            if isinstance(s.index, pd.MultiIndex):
                s = s.reset_index(level=0, drop=True)
            df[alpha_name] = s.reindex(df.index)
        except Exception as e:
            if logger:
                logger.warning("Alpha %s failed (%s); filling with 0.0", alpha_name, e)
            df[alpha_name] = 0.0
    return df
