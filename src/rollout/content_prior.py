"""Content prior utilities for rollout analysis."""

from functools import lru_cache
from pathlib import Path

from jaxtyping import Float
import numpy as np
from numpy.typing import NDArray
import pydantic
import torch


class ContentPriorConfig(pydantic.BaseModel):
    """Configuration for content prior matrix C."""

    diagonal_scores: list[list[float]] | None = pydantic.Field(
        default=None,
        description="Additional content scores of the diagonal entries (j=i)",
    )
    base_scores: list[list[float]] | None = pydantic.Field(
        default=None,
        description="Base scores for the score matrix",
    )
    attention_heatmap_path: pydantic.DirectoryPath | None = pydantic.Field(
        default=None,
        description="Optional path to the attention heatmaps directory.",
    )
    attention_probs_path: pydantic.DirectoryPath | None = pydantic.Field(
        default=None,
        description="Optional path to measured expected attention probabilities.",
    )

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.model_validator(mode="after")
    def check_dimensions(self) -> "ContentPriorConfig":  # noqa: C901
        """Validate the dimensions of the content prior configuration."""
        has_heatmap = self.attention_heatmap_path is not None
        has_probs = self.attention_probs_path is not None
        has_scalar_config = (
            self.diagonal_scores is not None or self.base_scores is not None
        )

        if sum((has_heatmap, has_probs, has_scalar_config)) > 1:
            raise ValueError(
                "Provide exactly one of attention_heatmap_path, "
                "attention_probs_path, or diagonal/base scores",
            )

        if not has_heatmap and not has_probs:
            if self.diagonal_scores is None:
                raise ValueError(
                    "diagonal_scores is required when attention_heatmap_path is not set",
                )
            if self.base_scores is None:
                raise ValueError(
                    "base_scores is required when attention_heatmap_path is not set",
                )

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


@lru_cache(maxsize=1)
def _load_attention_heatmap(path: Path, sequence_length: int) -> NDArray[np.float64]:
    heatmap: Float[torch.Tensor, "n_layers n_heads seq_len seq_len"] = torch.load(
        path / str(sequence_length) / "qk_mean.pt",
        map_location="cpu",
    )

    return np.asarray(heatmap.detach().cpu().numpy(), dtype=float)


@lru_cache(maxsize=1)
def _load_attention_probabilities(
    path: Path,
    sequence_length: int,
) -> NDArray[np.float64]:
    probabilities: Float[
        torch.Tensor,
        "n_layers n_heads seq_len seq_len",
    ] = torch.load(
        path / str(sequence_length) / "attention_probs_mean.pt",
        map_location="cpu",
    )
    return np.asarray(probabilities.detach().cpu().numpy(), dtype=float)


def make_measured_attention(
    config: ContentPriorConfig,
    *,
    layer_idx: int,
    sequence_length: int,
    num_heads: int,
) -> Float[NDArray, "num_heads sequence_length sequence_length"] | None:
    """Load E[softmax(L)] for one layer when a measured kernel is configured."""
    if config.attention_probs_path is None:
        return None

    probabilities = _load_attention_probabilities(
        config.attention_probs_path,
        sequence_length,
    )[layer_idx]
    expected_shape = (num_heads, sequence_length, sequence_length)
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"Measured attention shape {probabilities.shape} does not match "
            f"expected shape {expected_shape}",
        )
    return probabilities


def _make_content_prior_from_heatmap(
    config: ContentPriorConfig,
    *,
    layer_idx: int,
    sequence_length: int,
) -> Float[NDArray, "num_heads sequence_length sequence_length"]:
    assert config.attention_heatmap_path is not None
    heatmap = _load_attention_heatmap(config.attention_heatmap_path, sequence_length)

    return heatmap[layer_idx, :, :, :]


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
    if config.attention_probs_path is not None:
        raise ValueError(
            "Measured attention probabilities are already normalized kernels "
            "and cannot be used as pre-softmax content scores.",
        )
    if config.attention_heatmap_path is not None:
        content_priors = _make_content_prior_from_heatmap(
            config,
            layer_idx=layer_idx,
            sequence_length=sequence_length,
        )

        if content_priors.shape != (num_heads, sequence_length, sequence_length):
            raise ValueError(
                f"Heatmap shape {content_priors.shape} does not match expected shape {(num_heads, sequence_length, sequence_length)}",
            )

        return content_priors

    if config.diagonal_scores is None or config.base_scores is None:
        raise ValueError(
            "diagonal_scores and base_scores are required when attention_heatmap_path is not set",
        )

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

        content_priors[i] = np.full(
            (sequence_length, sequence_length),
            base_score,
            dtype=float,
        ) + diag_score * np.eye(sequence_length, dtype=float)

    return content_priors
