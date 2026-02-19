"""Content prior utilities for rollout analysis."""

from typing import Literal

from jaxtyping import Float
import numpy as np
from numpy.typing import NDArray
import pydantic

ContentPriorType = Literal["uniform_past", "exp_recency"]


class ContentPriorConfig(pydantic.BaseModel):
    """Configuration for content prior matrix C."""

    diagonal_scores: list[list[float]] = pydantic.Field(
        description="Additional content scores of the diagonal entries (j=i)",
    )
    base_scores: list[list[float]] = pydantic.Field(
        description="Base scores for the score matrix",
    )

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.model_validator(mode="after")
    def check_dimensions(self) -> "ContentPriorConfig":
        """Validate the dimensions of the content prior configuration."""
        if len(self.diagonal_scores) == 0:
            raise ValueError("diagonal_scores must not be empty")
        if len(self.base_scores) == 0:
            raise ValueError("base_scores must not be empty")

        if len(self.diagonal_scores) != len(self.base_scores):
            raise ValueError(
                "Length of diagonal_scores must match length of base_scores",
            )

        for diag_row, base_row in zip(
            self.diagonal_scores,
            self.base_scores,
            strict=True,
        ):
            if len(diag_row) == 0:
                raise ValueError("Rows in diagonal_scores must not be empty")
            if len(base_row) == 0:
                raise ValueError("Rows in base_scores must not be empty")

            if len(diag_row) != len(base_row):
                raise ValueError(
                    "Each row in diagonal_scores must match length of corresponding row in base_scores",
                )

        return self


def make_content_prior(
    config: ContentPriorConfig,
    *,
    sequence_length: int,
    layer_idx: int,
    num_heads: int,
) -> Float[NDArray, "num_heads sequence_length sequence_length"]:
    """
    Build the content prior matrix C.

    Args:
        config: ContentPriorConfig with diagonal and base scores.
        sequence_length: Sequence length to build matrices for.
        layer_idx: Layer index to select scores from.
        num_heads: Number of heads for the output tensor.

    Returns:
        The content prior tensor.

    """
    provided_layers = len(config.diagonal_scores)
    provided_heads = len(config.diagonal_scores[0])

    layer_idx = min(layer_idx, provided_layers - 1)
    layer_diag_scores = config.diagonal_scores[layer_idx]
    layer_base_scores = config.base_scores[layer_idx]

    content_priors = np.zeros(
        (num_heads, sequence_length, sequence_length),
        dtype=float,
    )

    for i in range(num_heads):
        head_idx = min(i, provided_heads - 1)
        diag_score = layer_diag_scores[head_idx]
        base_score = layer_base_scores[head_idx]

        content_priors[head_idx] = np.full(
            (sequence_length, sequence_length),
            base_score,
            dtype=float,
        ) + diag_score * np.eye(sequence_length, dtype=float)

    return content_priors
