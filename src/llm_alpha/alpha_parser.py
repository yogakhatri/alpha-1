from __future__ import annotations

"""AST-based safety validator for LLM alpha expressions."""

import ast
from dataclasses import dataclass
from typing import Iterable

from src.llm_alpha.alpha_whitelist import ALLOWED_FUNCS


@dataclass(frozen=True)
class ParseResult:
    """Validation result for one expression parse attempt."""
    ok: bool
    reason: str = ""


ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
)


WINDOW_FUNCS = {"rolling_mean", "rolling_std", "rolling_min", "rolling_max", "ewm_mean", "zscore", "delta", "shift"}


def _const_number(node: ast.AST) -> float | None:
    """Best-effort constant folding for numeric AST nodes."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _const_number(node.operand)
        if val is None:
            return None
        return val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        lv = _const_number(node.left)
        rv = _const_number(node.right)
        if lv is None or rv is None:
            return None
        if isinstance(node.op, ast.Add):
            return lv + rv
        if isinstance(node.op, ast.Sub):
            return lv - rv
        if isinstance(node.op, ast.Mult):
            return lv * rv
        if rv == 0:
            return None
        return lv / rv
    return None


def validate_expression(expr: str, feature_names: Iterable[str], max_chars: int, max_window: int | None = None) -> ParseResult:
    """Validate expression against whitelist, node safety, and window rules."""
    if len(expr) > max_chars:
        return ParseResult(False, "too_long")
    if "__" in expr:
        return ParseResult(False, "dunder_forbidden")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return ParseResult(False, "syntax_error")

    allowed_names = set(feature_names) | set(ALLOWED_FUNCS) | {"eps"}

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            return ParseResult(False, f"node_forbidden:{type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            return ParseResult(False, f"name_forbidden:{node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return ParseResult(False, "call_non_name")
            if node.func.id not in ALLOWED_FUNCS:
                return ParseResult(False, f"func_forbidden:{node.func.id}")

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        fn = node.func.id

        if fn == "shift" and len(node.args) >= 2:
            n = _const_number(node.args[1])
            if n is not None and n < 0:
                return ParseResult(False, "negative_shift")

        if max_window is not None and fn in WINDOW_FUNCS and len(node.args) >= 2:
            w = _const_number(node.args[1])
            if w is not None:
                if fn in {"rolling_mean", "rolling_std", "rolling_min", "rolling_max", "ewm_mean", "zscore", "delta"} and w < 1:
                    return ParseResult(False, "window_non_positive")
                if w > max_window:
                    return ParseResult(False, "window_too_large")

    return ParseResult(True, "")
