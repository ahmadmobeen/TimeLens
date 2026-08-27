"""Paths and frozen hyperparameters for the C boundary-head build (NS-P1).

Single source of truth for the model id, data splits, the eval/sampling recipe (kept
identical to the 085540 baseline reproduction so the cached hidden states are drawn
under the same frame sampling), and the cache / checkpoint output roots. Imported by
``cache_hidden.py``, ``train_head.py``, and the repo-root ``score_head.py``.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- repo layout ---------------------------------------------------------------
TIMELENS_ROOT = Path(__file__).resolve().parents[1]   # .../research/TimeLens
REPO_ROOT = TIMELENS_ROOT.parents[1]                  # .../NeedleScan

# --- model: FROZEN for all of C (no LoRA, no backbone update) -------------------
MODEL_PATH = os.environ.get("C_MODEL_PATH", "TencentARC/TimeLens-7B")
HIDDEN_SIZE = 3584                                    # Qwen2.5-VL-7B LLM width

# --- sampling recipe: MUST match the baseline repro (FPS-2 TOTAL-14336 MIN-64) --
FPS = 2
MIN_TOKENS = 64
TOTAL_TOKENS = 14336

# --- data splits ----------------------------------------------------------------
DATA_DIR = TIMELENS_ROOT / "data"
TRAIN_NATURAL_JSON = DATA_DIR / "charades_full" / "charades_train_natural.json"
TRAIN_SPEEDAUG_JSON = DATA_DIR / "charades_full" / "charades_train_speedaug.json"
EVAL_JSON = DATA_DIR / "TimeLens-Bench" / "charades-timelens.json"

# text-decoding baseline (and B-arm logs) the head is compared against
BASELINE_EVAL_JSONL = (
    TIMELENS_ROOT / "logs" / "TimeLens-7B_20260615_085540" / "charades-timelens.jsonl"
)

# --- outputs --------------------------------------------------------------------
OUT_ROOT = TIMELENS_ROOT / "output" / "C-boundary-head"
CACHE_DIR = OUT_ROOT / "cache"
CKPT_DIR = OUT_ROOT / "ckpt"
