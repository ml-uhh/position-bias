"""Positional encoding utilities for rollout analysis."""

import einops
from jaxtyping import Bool, Float
import numpy as np
from numpy.typing import NDArray


def compute_alibi_scores(
    alpha: float,
    mask: Bool[NDArray, "sequence_length sequence_length"],
) -> Float[NDArray, "sequence_length sequence_length"]:
    """
    Compute the positional-only ALiBi scores.

    Positional-only ALiBi scores for a single head:
      score(i,j) = 0            if j==i
                = -alpha*(i-j)  if j<i
                = -inf          if j>i (handled by mask; we fill with large negative)

    Args:
        alpha : ALiBi slope parameter (>0)
        mask  : boolean mask

    Returns:
        scores: ALiBi scores

    """
    sequence_length, _ = mask.shape

    i = einops.rearrange(np.arange(sequence_length), "n -> n 1")
    j = einops.rearrange(np.arange(sequence_length), "n -> 1 n")
    distances = (i - j).astype(float)
    scores = -alpha * distances
    np.fill_diagonal(scores, 0.0)
    scores = np.where(mask, scores, -1e30)
    return scores
