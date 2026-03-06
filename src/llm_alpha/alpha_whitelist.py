from __future__ import annotations

"""Whitelisted alpha-function vocabulary and alpha-library schema."""

from typing import Literal
from pydantic import BaseModel, Field

ALLOWED_FUNCS = {
    "shift",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "ewm_mean",
    "zscore",
    "delta",
    "safe_div",
    "clip",
    "abs",
    "sign",
    "log1p",
}

ALLOWED_BINOPS = {"+", "-", "*", "/"}
ALLOWED_UNARYOPS = {"+", "-"}

class AlphaLibrary(BaseModel):
    """Validated set of selected alpha expressions used by train/inference."""
    k: int
    feature_names: list[str]
    formulas: dict[str, str] = Field(default_factory=dict)
    provider: Literal["manual", "openai_compatible"] = "manual"
