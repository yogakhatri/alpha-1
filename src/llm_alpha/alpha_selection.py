from __future__ import annotations

"""LLM alpha generation, validation, and objective-aligned selection."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.core.config import AppConfig
from src.core.logging import get_logger
from src.core.paths import RunPaths
from src.backtest.simulator import add_label_column_barrier
from src.llm_alpha.alpha_executor import compute_alpha
from src.llm_alpha.alpha_whitelist import AlphaLibrary
from src.llm_alpha.prompt_templates import build_alpha_prompt
from src.models.explain import feature_schema_from_df

log = get_logger(__name__)


def _summarize_data_context(feats: pd.DataFrame) -> dict[str, object]:
    """Build compact dataset context to make LLM prompt data-aware."""
    ctx: dict[str, object] = {
        "rows": int(len(feats)),
        "n_tickers": int(feats["ticker"].nunique()) if "ticker" in feats.columns else None,
    }
    if "date" in feats.columns:
        d = pd.to_datetime(feats["date"], errors="coerce")
        if d.notna().any():
            ctx["date_start"] = str(d.min().date())
            ctx["date_end"] = str(d.max().date())

    for c in ("returns_1d", "returns_5d", "ATR_14", "RSI_14", "vol_z", "turnover_med_20d"):
        if c not in feats.columns:
            continue
        s = pd.to_numeric(feats[c], errors="coerce").dropna()
        if s.empty:
            continue
        ctx[f"{c}_mean"] = float(s.mean())
        ctx[f"{c}_std"] = float(s.std())
        ctx[f"{c}_p01"] = float(s.quantile(0.01))
        ctx[f"{c}_p99"] = float(s.quantile(0.99))

    return ctx


def _extract_first_json_object(text: str) -> str | None:
    """Extract first balanced JSON object from free-form text."""
    # Fast path: whole string is JSON.
    try:
        obj = json.loads(text)
        return json.dumps(obj)
    except Exception:
        pass

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _coerce_alpha_map(obj: object) -> dict[str, str]:
    """Normalize provider output into {name: expression} dictionary."""
    candidates: list[str] = []

    def add_expr(x: object) -> None:
        if isinstance(x, str):
            e = x.strip()
            if e:
                candidates.append(e)
            return
        if isinstance(x, dict):
            for key in ("expr", "expression", "formula", "alpha", "value"):
                v = x.get(key)
                if isinstance(v, str) and v.strip():
                    candidates.append(v.strip())
                    return

    if isinstance(obj, dict):
        # Most common expected shape.
        if obj and all(isinstance(v, str) for v in obj.values()):
            out: dict[str, str] = {}
            for i, (_, expr) in enumerate(obj.items(), start=1):
                e = expr.strip()
                if e:
                    out[f"AlphaCandidate_{i}"] = e
            if out:
                return out

        # Alternate shape: {"alphas": [...]}
        if isinstance(obj.get("alphas"), list):
            for item in obj["alphas"]:
                add_expr(item)
        else:
            # Any dict values that hold alpha objects.
            for v in obj.values():
                add_expr(v)
    elif isinstance(obj, list):
        for item in obj:
            add_expr(item)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for e in candidates:
        if e in seen:
            continue
        seen.add(e)
        uniq.append(e)

    if not uniq:
        raise RuntimeError("Could not extract alpha expressions from provider response.")

    return {f"AlphaCandidate_{i+1}": expr for i, expr in enumerate(uniq)}


def _parse_response_text_to_alpha_map(text: str) -> dict[str, str]:
    """Parse free-form LLM text response into normalized alpha map."""
    try:
        obj = json.loads(text)
    except Exception:
        obj_text = _extract_first_json_object(text)
        if not obj_text:
            raise RuntimeError("LLM response does not contain a valid JSON object.")
        obj = json.loads(obj_text)
    return _coerce_alpha_map(obj)


def build_or_load_alpha_library(cfg: AppConfig, paths: RunPaths, feats: pd.DataFrame) -> AlphaLibrary:
    """Load cached alpha library or build a new one from LLM output."""
    cache_dir = Path(cfg.llm_alpha.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    alpha_path = paths.alpha_library_path()
    if alpha_path.exists():
        return AlphaLibrary.model_validate(json.loads(alpha_path.read_text()))

    feature_names = feature_schema_from_df(feats)
    candidate_pool_size = max(int(cfg.llm_alpha.k_alphas) * 3, 40)
    prompt = build_alpha_prompt(
        feature_names=feature_names,
        k=cfg.llm_alpha.k_alphas,
        candidate_pool_size=candidate_pool_size,
        max_window=cfg.llm_alpha.max_rolling_window,
        max_chars=cfg.llm_alpha.max_expr_chars,
        horizon_days=cfg.label.horizon_days,
        profit_take_pct=cfg.label.profit_take_pct,
        stop_loss_pct=cfg.label.stop_loss_pct,
        min_price=cfg.universe.min_price,
        min_turnover=cfg.universe.min_median_turnover_20d,
        data_context=_summarize_data_context(feats),
    )

    resp_json = _get_llm_alphas(cfg, cache_dir, prompt)
    formulas = _validate_and_select(cfg, feats, feature_names, resp_json)

    return AlphaLibrary(
        k=cfg.llm_alpha.k_alphas,
        feature_names=feature_names,
        formulas=formulas,
        provider=cfg.llm_alpha.provider,
    )


def _get_llm_alphas(cfg: AppConfig, cache_dir: Path, prompt: str) -> dict[str, str]:
    """Request alpha expressions from configured provider or manual cache."""
    prompt_path = cache_dir / "alpha_prompt.txt"
    prompt_path.write_text(prompt)

    out_path = cache_dir / "alpha_response.json"
    if out_path.exists():
        return _parse_response_text_to_alpha_map(out_path.read_text())

    if cfg.llm_alpha.provider == "manual":
        raise RuntimeError(
            f"Manual alpha mode: prompt written to {prompt_path}. "
            f"Paste LLM JSON response into {out_path} then rerun."
        )

    if cfg.llm_alpha.provider == "openai_compatible":
        oc = cfg.llm_alpha.openai_compatible
        if oc is None:
            raise ValueError("openai_compatible config missing")
        key = os.environ.get(oc.api_key_env)
        if not key:
            raise RuntimeError(f"Missing API key env var: {oc.api_key_env}")

        url = oc.base_url.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": oc.model,
            "messages": [
                {"role": "system", "content": "You output only JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=60)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        alpha_map = _parse_response_text_to_alpha_map(content)
        out_path.write_text(json.dumps(alpha_map, indent=2))
        return alpha_map

    raise ValueError(f"Unknown provider: {cfg.llm_alpha.provider}")


def _validate_and_select(
    cfg: AppConfig,
    feats: pd.DataFrame,
    feature_names: list[str],
    llm_obj: dict[str, str],
) -> dict[str, str]:
    """Score candidate alphas and keep best `k` formulas.

    Selection objective:
    weighted daily top-1 / top-2 precision on the barrier label.
    """
    # Select alphas by barrier-label ranking quality (top-1/top-2 precision by day),
    # aligned to the actual swing objective.
    df = feats.sort_values(["ticker", "date"]).copy()
    if "label" not in df.columns:
        df = add_label_column_barrier(cfg, df)

    def daily_topk_precision(tmp: pd.DataFrame, score_col: str, k: int, ascending: bool = False) -> float:
        x = tmp.sort_values(["date", score_col], ascending=[True, ascending]).groupby("date", as_index=False).head(k)
        if x.empty:
            return 0.0
        return float(x["y"].mean())

    candidates = []
    for name, expr in llm_obj.items():
        if not isinstance(expr, str):
            continue
        try:
            s = df.groupby("ticker", group_keys=False).apply(
                lambda g: compute_alpha(
                    g,
                    expr,
                    feature_names,
                    cfg.llm_alpha.max_expr_chars,
                    cfg.llm_alpha.max_rolling_window,
                )
            )
            s = s.reset_index(level=0, drop=True)
            tmp = pd.DataFrame({"date": pd.to_datetime(df["date"]), "a": s, "y": df["label"]}).dropna()
            if len(tmp) < 3000 or tmp["date"].nunique() < 120:
                continue

            top1_long = daily_topk_precision(tmp, "a", k=1, ascending=False)
            top2_long = daily_topk_precision(tmp, "a", k=2, ascending=False)
            metric_long = 0.7 * top1_long + 0.3 * top2_long

            top1_short = daily_topk_precision(tmp, "a", k=1, ascending=True)
            top2_short = daily_topk_precision(tmp, "a", k=2, ascending=True)
            metric_short = 0.7 * top1_short + 0.3 * top2_short

            if metric_short > metric_long:
                oriented_expr = f"(-1.0*({expr}))"
                metric = metric_short
            else:
                oriented_expr = expr
                metric = metric_long

            if np.isnan(metric):
                continue
            candidates.append((float(metric), name, oriented_expr))
        except Exception:
            continue

    log.info("Alpha candidates accepted after validation/scoring: %s", len(candidates))
    candidates.sort(reverse=True, key=lambda x: x[0])
    chosen = candidates[: cfg.llm_alpha.k_alphas]
    out = {}
    for i, (_, _, expr) in enumerate(chosen, start=1):
        out[f"Alpha_{i}"] = expr
    if len(out) < cfg.llm_alpha.k_alphas:
        raise RuntimeError("Not enough safe alphas produced; rerun with a better LLM response.")
    return out
