"""
CS321M Predictive Evaluation Challenge - labeling.py (optional)
================================================================

Acquisition function: farthest-point diversity sampling.

设计考量:
  - 不在 acquisition_function 里调 predict()
    -> predict() 每次调 ENCODER.encode()，5000个候选调两次会超时
  - 不再单独加载 SentenceTransformer
    -> 复用 model.py 中已加载的 ENCODER，避免 OOM
  - 用 module-level list 积累已见向量
    -> 手册说 acquisition_function 对每个候选逐个调用，看不到完整列表

策略: Greedy Farthest-point sampling
  - 第一个候选自动拿最高分
  - 后续候选: 分数 = 到所有已见向量的最小距离
  - 距离越大 = 越多样 = 分数越高
  - 不需要预训练 centroids，不需要 ship 额外文件

Fixes applied:
  #6:  添加 reset_state() 方法，确保轮次间状态干净重置
  #13: 文本截断限制从 300 chars 增加到 500 chars，保留更多语义信息
"""

import math
import numpy as np

# 复用 model.py 已加载的 encoder（避免双倍内存）
from model import ENCODER

# ── Module-level state ────────────────────────────────────
_seen_vectors = []


def reset_state():
    """
    Fix #6: 显式重置 acquisition 状态。
    在每个 round 开始前调用，确保不同轮次间状态干净。
    Codabench 容器每轮重启时自动重置；本地验证需手动调用此方法。
    """
    global _seen_vectors
    _seen_vectors = []


def _encode_input(input: dict) -> np.ndarray:
    """Encode the four fields into a single vector for diversity comparison."""
    # Fix #13: 增加截断限制到 500 chars，保留更多语义信息
    text = (
        f"Benchmark: {input['benchmark']}. "
        f"Condition: {input['condition']}. "
        f"Subject: {input['subject_content'][:500]}. "
        f"Item: {input['item_content'][:500]}"
    )
    return ENCODER.encode(text, convert_to_numpy=True)


def acquisition_function(input: dict) -> float:
    """
    Score how desirable it is to reveal this input's ground-truth label.
    Higher = more desirable for labeling.

    Uses greedy farthest-point sampling:
    - Score = minimum distance to all previously scored vectors
    - Encourages diversity in the labeled set
    - Order-dependent (greedy approximation), which is expected

    Note on side-effect (#6): The function appends each scored candidate's
    vector to _seen_vectors. This is intentional for the greedy farthest-point
    algorithm within a single round. Use reset_state() between rounds to
    ensure clean state transitions.

    Args:
        input: dict with keys "benchmark", "condition",
               "subject_content", "item_content" (all Python str)
    Returns:
        native Python float. Only ranking matters, not absolute value.
    """
    global _seen_vectors

    x = _encode_input(input)

    if len(_seen_vectors) == 0:
        # First candidate always gets highest score
        score = 1e6
    else:
        # Farthest-point: score = min distance to any previously seen vector
        # Use vectorized numpy for speed
        seen_matrix = np.array(_seen_vectors)  # (N, D)
        dists = np.linalg.norm(seen_matrix - x, axis=1)  # (N,)
        score = float(dists.min())

    _seen_vectors.append(x)

    # Safety: platform discards ALL scores if any return is NaN/Inf
    score = float(score)
    if not math.isfinite(score):
        score = 0.0
    return score
