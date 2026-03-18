"""Centralized device resolution and performance tuning.

Automatically selects the best available accelerator (CUDA > MPS > CPU)
and configures PyTorch for maximum throughput on each platform:

  - **CUDA (Kaggle / cloud)**: cuDNN benchmark, TF32, AMP float16, GradScaler,
    pin_memory, multi-worker DataLoaders, torch.compile (PyTorch ≥ 2).
  - **MPS (Apple Silicon M-series)**: TF32 matmul, float32 (MPS float16 is
    still buggy), moderate batch sizes, torch.compile where supported.
  - **CPU**: OpenMP / MKL thread tuning, no AMP.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import platform
from functools import lru_cache

import torch

# ---------------------------------------------------------------------------
# Thread / backend tuning (run once at import time)
# ---------------------------------------------------------------------------

_NUM_PHYSICAL_CORES = max(1, multiprocessing.cpu_count())

# Set threading env-vars before any BLAS / OpenMP library reads them.
for _env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env, str(_NUM_PHYSICAL_CORES))

torch.set_num_threads(_NUM_PHYSICAL_CORES)
try:
    torch.set_num_interop_threads(max(1, _NUM_PHYSICAL_CORES // 2))
except RuntimeError:
    pass  # already set or unsupported


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def resolve_device(*, prefer_cuda: bool = True, prefer_mps: bool = True) -> torch.device:
    """Pick the best accelerator: CUDA > MPS > CPU.

    Returns the same cached device instance across the whole process.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Per-device tuning (called once after device is resolved)
# ---------------------------------------------------------------------------

_TUNED = False


def tune_for_device(device: torch.device) -> None:
    """Apply one-time global PyTorch settings for *device*."""
    global _TUNED
    if _TUNED:
        return
    _TUNED = True

    if device.type == "cuda":
        # cuDNN auto-tuner: finds fastest convolution algorithm for input shapes
        torch.backends.cudnn.benchmark = True
        # TF32 on Ampere+ gives ~2× matmul throughput at negligible precision loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Tell PyTorch's internal GEMM dispatcher to prefer TF32
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    elif device.type == "mps":
        # MPS benefits from "high" precision hint on Apple Silicon
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # CPU: threading was already configured at module import


# ---------------------------------------------------------------------------
# AMP (Automatic Mixed Precision)
# ---------------------------------------------------------------------------

def make_amp_context(device: torch.device):
    """Return ``(autocast_ctx_factory, grad_scaler_or_None)`` for *device*.

    * CUDA  → float16 autocast + GradScaler
    * MPS   → float32 (MPS float16 still experimental/buggy)
    * CPU   → no-op
    """
    if device.type == "cuda":
        autocast_fn = lambda: torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        scaler = torch.amp.GradScaler("cuda")
        return autocast_fn, scaler
    # MPS and CPU: no mixed precision
    return contextlib.nullcontext, None


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def optimal_num_workers(device: torch.device) -> int:
    """Heuristic for DataLoader ``num_workers``.

    * CUDA  → min(cores, 4)   faster than 0 thanks to pinned-memory pre-fetch.
    * MPS   → 0               multi-process data loading on macOS is unreliable
              with fork-safety issues. Keep main-process loading.
    * CPU   → 0               avoids IPC overhead for small-ish datasets.
    """
    if device.type == "cuda":
        return min(_NUM_PHYSICAL_CORES, 4)
    # macOS (MPS) + CPU: num_workers > 0 often slower due to fork/spawn overhead
    return 0


def pin_memory_for(device: torch.device) -> bool:
    """``pin_memory`` is only beneficial with CUDA page-locked transfers."""
    return device.type == "cuda"


def optimal_batch_size(device: torch.device, base: int = 256) -> dict[str, int]:
    """Return ``{"train": …, "inference": …}`` batch sizes tuned per device.

    * CUDA  → 2× base for training, 4× for inference (GPU memory is plentiful).
    * MPS   → base (unified memory, so moderate).
    * CPU   → base // 2.
    """
    if device.type == "cuda":
        # Scale up on GPU; caller should halve if OOM
        return {"train": base * 2, "inference": base * 4}
    if device.type == "mps":
        return {"train": base, "inference": base * 2}
    return {"train": max(64, base // 2), "inference": base}


# ---------------------------------------------------------------------------
# torch.compile wrapper
# ---------------------------------------------------------------------------

def try_compile(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Attempt ``torch.compile`` for a speed-up on PyTorch ≥ 2.

    Falls back to the original model silently if compilation is unavailable
    or fails (e.g. unsupported ops, older PyTorch, or Apple MPS).
    """
    if not hasattr(torch, "compile"):
        return model

    # torch.compile with "reduce-overhead" mode using CUDA graphs
    if device.type == "cuda":
        try:
            return torch.compile(model, mode="reduce-overhead")
        except Exception:
            return model

    # MPS: inductor backend has partial support starting PyTorch 2.4+
    if device.type == "mps":
        try:
            return torch.compile(model, backend="aot_eager")
        except Exception:
            return model

    # CPU: inductor can help on x86; aot_eager is safe fallback
    try:
        return torch.compile(model, backend="inductor")
    except Exception:
        return model


# ---------------------------------------------------------------------------
# Inference context
# ---------------------------------------------------------------------------

def inference_context():
    """Prefer ``torch.inference_mode`` over ``torch.no_grad`` for speed."""
    return torch.inference_mode()


# ---------------------------------------------------------------------------
# Summary logger
# ---------------------------------------------------------------------------

def log_device_info(device: torch.device, logger=None) -> None:
    """Emit a one-line summary of the active compute device."""
    _log = logger or __import__("logging").getLogger(__name__)

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        _log.info(
            "Device: %s | %s | VRAM %.1f GB | CUDA %s",
            device,
            props.name,
            props.total_mem / 1e9,
            torch.version.cuda,
        )
    elif device.type == "mps":
        _log.info(
            "Device: MPS (Apple Silicon %s) | PyTorch %s",
            platform.processor() or "arm",
            torch.__version__,
        )
    else:
        _log.info("Device: CPU | %s cores | PyTorch %s", _NUM_PHYSICAL_CORES, torch.__version__)
