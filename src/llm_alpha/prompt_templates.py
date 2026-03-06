from __future__ import annotations

"""Prompt construction for LLM-generated alpha formula proposals."""

import json
from typing import Sequence


def build_alpha_prompt(
    feature_names: Sequence[str],
    k: int,
    candidate_pool_size: int,
    max_window: int,
    max_chars: int,
    horizon_days: int,
    profit_take_pct: float,
    stop_loss_pct: float,
    min_price: float,
    min_turnover: float,
    data_context: dict[str, object],
) -> str:
    """Build a robust alpha-generation prompt tailored to current dataset/config.

    The prompt intentionally encodes:
    - real execution assumptions (EOD decision, next-open entry),
    - objective alignment (barrier-hit success for 1-5 day swings),
    - strict syntax/safety limits,
    - cross-sectional diversity requirements to reduce redundant formulas.
    """
    cols = ", ".join(feature_names)
    context_json = json.dumps(data_context, indent=2, default=str)
    return f"""
You are Alpha Researcher for an NSE cash-equities swing signal system (long-only).

Primary objective:
- Propose realistic formulas that help rank stocks likely to hit profit target
  before stop-loss within a short holding horizon.
- Trading setup for labeling/backtest:
  - decision timing: end of day
  - entry: next-day open
  - horizon_days: {horizon_days}
  - profit_take_pct: {profit_take_pct}
  - stop_loss_pct: {stop_loss_pct}

Universe / data profile:
- min_price filter: {min_price}
- min_median_turnover_20d filter: {min_turnover}
- current dataset summary:
{context_json}

Feature space available (ONLY these columns may be used):
{cols}

Allowed functions (ONLY these):
shift, rolling_mean, rolling_std, rolling_min, rolling_max, ewm_mean, zscore, delta, safe_div, clip, abs, sign, log1p

Hard constraints:
1) No lookahead leakage:
   - shift(x,n) must use n>=0 only.
   - never reference future data implicitly.
2) Window/span limits:
   - rolling/ewm/zscore/delta/shift argument windows <= {max_window}
   - prefer practical windows for short swing trading (roughly 3 to 40)
3) Expression size:
   - each expression length <= {max_chars} chars
4) Numerical robustness:
   - use safe_div for ratios where denominator can be near zero
   - avoid unstable constructs that create mostly NaN/inf/sparse triggers
5) Economic plausibility:
   - prefer interpretable signals combining trend, momentum, volatility, and liquidity
   - avoid overfit constants and extremely brittle threshold logic

Diversity requirements for the candidate pool:
- Return {candidate_pool_size} candidates (NOT only {k})
- Include multiple archetypes:
  - momentum continuation
  - mean reversion after extension
  - breakout/volatility expansion
  - trend + participation (volume/turnover confirmation)
  - volatility-adjusted strength/weakness
- Avoid near-duplicate formulas (small variants of same structure).

Output format requirements:
- Output ONLY valid JSON object, no markdown, no comments, no prose.
- JSON must be exactly key/value pairs like:
  {{
    "Alpha_1": "<expression>",
    "Alpha_2": "<expression>"
  }}
- Values must be expression strings only.

Now return the JSON candidate pool.
""".strip()
