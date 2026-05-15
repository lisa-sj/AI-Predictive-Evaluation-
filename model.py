"""
CS321M Predictive Evaluation Challenge - model.py (required)
=============================================================

NCF baseline with condition embedding + online Platt scaling.

Architecture:
  - Sentence Transformer (all-mpnet-base-v2) encodes subject_content & item_content
  - Condition embedding table (32-dim, with fallback for unseen conditions)
  - MLP head: [768*2 + 32] -> 256 -> 256 -> 1
  - Online Platt scaling using labeled data (temperature calibration)
  - Probability clipping to [0.01, 0.99]

Fixes applied:
  #1:  Embedding cache — avoid re-encoding same text (prevents Codabench timeout)
  #2:  Platt logit cache — _fit_temperature benefits from embedding cache
  #3:  Auto-reset calibrator when labeled data changes between rounds
  #5:  Rasch θ blending — subject ability prior from IRT model
  #10: Widened temperature search range [0.2, 5.0] with 200 grid points
  #12: Error handling in predict() — fallback to base rate on failure
"""

import json
import math
import os
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# ══════════════════════════════════════════════════════════
#  Module-level loading (runs once when container starts)
# ══════════════════════════════════════════════════════════

ENCODER = SentenceTransformer("all-mpnet-base-v2")
ENCODER.eval()

# Load condition map (includes "unknown" as fallback)
with open("condition_map.json") as f:
    CONDITION_MAP = json.load(f)

# Fallback index for unseen conditions
_UNKNOWN_IDX = CONDITION_MAP.get("unknown", CONDITION_MAP.get("none", 0))

N_CONDITIONS = len(CONDITION_MAP)
EMB_DIM = 768
COND_DIM = 32
HIDDEN = 256

# NCF model definition (must match training)
class NCFWithCondition(nn.Module):
    def __init__(self, emb_dim, n_conditions, cond_dim=32, hidden=256):
        super().__init__()
        self.condition_emb = nn.Embedding(n_conditions, cond_dim)
        input_dim = 2 * emb_dim + cond_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, u, v, c):
        c_emb = self.condition_emb(c)
        x = torch.cat([u, v, c_emb], dim=-1)
        return self.mlp(x).squeeze(-1)

NCF = NCFWithCondition(EMB_DIM, N_CONDITIONS, COND_DIM, HIDDEN)
NCF.load_state_dict(torch.load("ncf_head.pt", map_location="cpu", weights_only=True))
NCF.eval()

# ── Rasch θ subject priors (disabled) ────────────────────
# rasch_subject_map.json maps training-time fallback text → θ, but at
# inference the platform sends real model descriptions as subject_content.
# The lookup always misses, falling back to mean θ — adding a constant
# to every logit with no subject-specific information.
# Disabled until embedding-based nearest-neighbor matching is implemented.
_RASCH_THETA = {}  # intentionally empty — feature disabled

# ── Fix #1: Embedding cache ─────────────────────────────
_embedding_cache = {}    # text → tensor (cpu, L2-normalized)
_MAX_CACHE_SIZE = 2000   # prevent unbounded memory growth

# ── Fix #3: Platt scaling state with change detection ────
_calibrator = None           # fitted temperature T, None = not yet fitted
_last_labeled_hash = None    # fingerprint of labeled list for auto-reset


def _get_condition_idx(condition_str):
    """Safely get condition index, fallback to 'unknown' for unseen conditions."""
    if condition_str is None or condition_str == "":
        condition_str = "none"
    return CONDITION_MAP.get(condition_str, _UNKNOWN_IDX)


def _encode_cached(text):
    """
    Fix #1: Encode text with LRU-style cache.
    Subjects repeat across all items in a round, so caching gives ~N× speedup.
    """
    if text not in _embedding_cache:
        # Evict half the cache if full (simple FIFO approximation)
        if len(_embedding_cache) >= _MAX_CACHE_SIZE:
            keys_to_evict = list(_embedding_cache.keys())[:_MAX_CACHE_SIZE // 2]
            for k in keys_to_evict:
                del _embedding_cache[k]
        emb = ENCODER.encode(text, convert_to_tensor=True).cpu()
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        _embedding_cache[text] = emb
    return _embedding_cache[text]


def _raw_logit(input_dict):
    """Compute raw NCF logit for a single input dict, with embedding caching."""
    with torch.no_grad():
        # Fix #1: Use cached encoding
        u = _encode_cached(input_dict["subject_content"])
        v = _encode_cached(input_dict["item_content"])
        c_idx = _get_condition_idx(input_dict.get("condition"))
        c = torch.tensor([c_idx], dtype=torch.long)
        logit = NCF(u.unsqueeze(0), v.unsqueeze(0), c).item()

    return logit


def _compute_labeled_hash(labeled):
    """Fix #3: Compute a fingerprint of the labeled list for change detection."""
    if not labeled:
        return ("empty",)
    # Use length + hash of first/last item content as fast fingerprint
    parts = [len(labeled)]
    for ex in labeled[:2]:
        parts.append(hash(ex.get("item_content", "")[:50]))
    if len(labeled) > 2:
        parts.append(hash(labeled[-1].get("item_content", "")[:50]))
    return tuple(parts)


def _fit_temperature(labeled):
    """
    Fit temperature T on labeled data: p_calibrated = sigmoid(logit / T).
    Uses grid search to minimize negative log-loss.

    Fix #2:  Benefits from _encode_cached — labeled logits computed once.
    Fix #10: Widened search range [0.2, 5.0] with 200 grid points.

    Safety: if calibrated NLL >= uncalibrated NLL (T=1), fall back to T=1.0
    to avoid hurting performance when labeled data is too small/unrepresentative.
    """
    if not labeled or len(labeled) < 5:
        # Too few labeled examples -> don't risk overfitting calibration
        return 1.0

    logits = []
    labels = []
    for ex in labeled:
        logit = _raw_logit(ex)  # Fix #2: benefits from embedding cache
        logits.append(logit)
        labels.append(ex["label"])

    logits = np.array(logits, dtype=np.float64)
    labels = np.array(labels, dtype=np.float64)

    # Compute uncalibrated NLL (T=1) as baseline
    p_uncal = 1.0 / (1.0 + np.exp(-logits))
    p_uncal = np.clip(p_uncal, 0.01, 0.99)
    nll_uncal = -np.mean(labels * np.log(p_uncal) + (1 - labels) * np.log(1 - p_uncal))

    # Fix #10: Widened grid search range [0.2, 5.0] with 200 points
    best_T = 1.0
    best_loss = nll_uncal  # Start with uncalibrated as baseline
    for T in np.linspace(0.2, 5.0, 200):
        p = 1.0 / (1.0 + np.exp(-logits / T))
        p = np.clip(p, 0.01, 0.99)
        nll = -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
        if nll < best_loss:
            best_loss = nll
            best_T = T

    # Safety: only use calibrated T if it actually improves over T=1
    if best_loss >= nll_uncal:
        return 1.0

    return best_T


# ══════════════════════════════════════════════════════════
#  predict() — called once per test input
# ══════════════════════════════════════════════════════════

def predict(input: dict,
            labeled: list = None) -> float:
    """
    Args:
        input: dict with keys "benchmark", "condition",
               "subject_content", "item_content" (all Python str)
        labeled: list of dicts with same 4 keys + "label" (int 0/1).
                 Same list on every call within a round. May be empty or None.
    Returns:
        native Python float in [0, 1]
    """
    global _calibrator, _last_labeled_hash

    # Fix #3: Auto-reset calibrator when labeled data changes between rounds
    current_hash = _compute_labeled_hash(labeled)
    if current_hash != _last_labeled_hash:
        _calibrator = None
        _last_labeled_hash = current_hash

    # Fit calibrator once per round (labeled is identical across all calls)
    if _calibrator is None:
        if labeled:
            _calibrator = _fit_temperature(labeled)
        else:
            _calibrator = 1.0

    T = _calibrator

    # Fix #12: Error handling — fallback to base rate on failure
    try:
        logit = _raw_logit(input)
        p = 1.0 / (1.0 + math.exp(-logit / T))
        p = max(0.01, min(0.99, p))
        return float(p)
    except Exception:
        # Fallback: use labeled base rate if available, else 0.5
        if labeled:
            base_rate = np.mean([ex["label"] for ex in labeled])
            return float(np.clip(base_rate, 0.01, 0.99))
        return 0.5
