"""Attention mask utilities for rollout analysis."""

from typing import Literal

import einops
from jaxtyping import Bool
import numpy as np
from numpy.typing import NDArray
import pydantic

MaskType = Literal["causal", "full", "sliding_window"]


class MaskConfig(pydantic.BaseModel):
    """Configuration for the attention mask."""

    mask_type: MaskType = "causal"
    window_size: pydantic.PositiveInt | None = pydantic.Field(
        default=None,
        description="Window size for sliding_window mask_type",
    )

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.model_validator(mode="after")
    def validate_window_present(self) -> "MaskConfig":
        """Validate that window_size is provided for sliding_window mask_type."""
        if self.mask_type == "sliding_window" and self.window_size is None:
            raise ValueError(
                "window_size must be provided for sliding_window mask_type",
            )
        return self


def make_mask(
    config: MaskConfig,
    sequence_length: int,
) -> Bool[NDArray, "sequence_length sequence_length"]:
    """
    Build the attention mask according to the specified type.

    Args:
        config: MaskConfig describing the mask type and parameters.
        sequence_length: Length of the sequence.

    Returns:
        The resulting attention mask.

    """
    match config.mask_type:
        case "full":
            return np.ones((sequence_length, sequence_length), dtype=bool)

    i = einops.rearrange(np.arange(sequence_length), "n -> n 1")
    j = einops.rearrange(np.arange(sequence_length), "n -> 1 n")
    match config.mask_type:
        case "causal":
            return j <= i
        case "sliding_window":
            assert config.window_size is not None
            return (j <= i) & (j >= i - (config.window_size - 1))
        case _:
            raise ValueError(f"Unknown mask_type: {config.mask_type}")
