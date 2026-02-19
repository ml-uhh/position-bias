"""Attention utilities for rollout analysis."""

from jaxtyping import Bool, Float
import numpy as np
from numpy.typing import NDArray
import pydantic

from src.rollout.positional_encoding import compute_alibi_scores


class AttentionConfig(pydantic.BaseModel):
    """Configuration for attention computation with ALiBi."""

    num_heads: pydantic.PositiveInt = pydantic.Field(
        description="Number of attention heads",
    )
    alibi_slopes: list[list[float]] = pydantic.Field(
        description="ALiBi slopes (one per head and and per layer). Outer list must have length 1 or match number of layers",
    )
    head_weights: list[list[pydantic.NonNegativeFloat]] | None = pydantic.Field(
        default=None,
        description="Head weights (must match alibi_slopes length) or None for uniform",
    )
    softmax_temperature: pydantic.PositiveFloat = 1.0

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.model_validator(mode="after")
    def validate_heads(self) -> "AttentionConfig":
        """Validates alibi_slopes and head_weights consistency."""
        if len(self.alibi_slopes) == 0:
            raise ValueError("alibi_slopes must be non-empty")

        for layer_slopes in self.alibi_slopes:
            if len(layer_slopes) not in (1, self.num_heads):
                raise ValueError(
                    f"Each alibi_slopes list must have length 1 or match num_heads but got {len(layer_slopes)}",
                )

            if len(layer_slopes) != len(self.alibi_slopes[0]):
                raise ValueError(
                    "All alibi_slopes lists must have the same length",
                )

        if self.head_weights is not None:
            if len(self.head_weights) != len(self.alibi_slopes):
                raise ValueError(
                    "head_weights must have same length as alibi_slopes",
                )
            for hw, ls in zip(self.head_weights, self.alibi_slopes, strict=True):
                if len(hw) != len(ls):
                    raise ValueError(
                        "Each head_weights list must match corresponding alibi_slopes length",
                    )

                if not any(w > 0 for w in hw):
                    raise ValueError("At least one head weight must be positive")

        return self


def compute_row_softmax(
    scores: Float[NDArray, "sequence_length sequence_length"],
    mask: Bool[NDArray, "sequence_length sequence_length"],
) -> Float[NDArray, "sequence_length sequence_length"]:
    """
    Compute a row-wise softmax with a boolean mask.

    Rows where the mask is all-False are returned as zero rows to avoid NaNs.

    Args:
        scores: 2D array of scores to softmax across rows.
        mask: Boolean mask indicating valid entries for softmax.

    Returns:
        The row-wise softmaxed matrix.

    """
    masked_scores = np.where(mask, scores, -np.inf)

    # stable softmax per row
    row_max = np.max(masked_scores, axis=1, keepdims=True)  # may be -inf
    shifted = masked_scores - row_max
    exps = np.zeros_like(shifted, dtype=float)
    finite = np.isfinite(shifted)
    exps[finite] = np.exp(shifted[finite])

    denom = np.sum(exps, axis=1, keepdims=True)
    softmax_scores = np.zeros_like(exps, dtype=float)
    good = denom[:, 0] > 0
    softmax_scores[good] = exps[good] / denom[good]
    return softmax_scores


def make_head_attention(
    config: AttentionConfig,
    *,
    mask: Bool[NDArray, "sequence_length sequence_length"],
    content_scores: Float[NDArray, "sequence_length sequence_length"],
    alibi_slope: float,
) -> Float[NDArray, "sequence_length sequence_length"]:
    """
    Compute single-head attention from content and ALiBi scores.

    Combines content scores with ALiBi positional biases, applies temperature scaling and a masked row-wise softmax.

    Args:
        config: Attention configuration containing softmax temperature.
        mask: Boolean mask for valid positions.
        content_scores: Per-position content scores.
        alibi_slope: ALiBi slope for this head.

    Returns:
        The attention matrix for the head.

    """
    scores = content_scores + compute_alibi_scores(alibi_slope, mask)

    scores = scores / config.softmax_temperature
    return compute_row_softmax(scores, mask)


def make_multihead_attention(
    config: AttentionConfig,
    *,
    mask: Bool[NDArray, "sequence_length sequence_length"],
    content_scores: Float[NDArray, "num_heads sequence_length sequence_length"],
    layer_index: int,
) -> Float[NDArray, "sequence_length sequence_length"]:
    """
    Compute weighted multi-head attention from per-head content scores and ALiBi.

    Computes each head's attention and returns their weighted average according to the configuration (uniform if head_weights is None).

    Args:
        config: Attention configuration including head weights and slopes.
        mask: Boolean mask for valid positions.
        content_scores: Per-head content scores tensor.
        layer_index: Layer index selecting ALiBi slopes and head weights.

    Returns:
        The aggregated attention matrix.

    """
    layer_index = min(layer_index, len(config.alibi_slopes) - 1)
    num_heads = config.num_heads

    if config.head_weights is None:
        head_weights = np.ones(num_heads, dtype=float) / num_heads
    else:
        head_weights = np.asarray(config.head_weights[layer_index], dtype=float)
        s = np.sum(head_weights)
        assert s > 0.0
        head_weights = head_weights / s

    head_attention_weighted_average = np.zeros_like(mask, dtype=float)
    for h in range(num_heads):
        head_attention = make_head_attention(
            config,
            mask=mask,
            alibi_slope=config.alibi_slopes[layer_index][
                min(h, len(config.alibi_slopes[layer_index]) - 1)
            ],
            content_scores=content_scores[h],
        )
        head_attention_weighted_average += head_weights[h] * head_attention

    return head_attention_weighted_average
