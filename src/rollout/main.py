"""
Model for residual-aware attention rollout.

Residual-aware (lazy) attention rollout with:
  - positional-only ALiBi (single- or multi-head),
  - optional "rough average content attention" prior C,
  - lambda schedules: constant / power-law decay / custom,
  - rollout P^(T) = R^(T) ... R^(0),
  - plotting: last-row distribution of P (where last token draws from after rollout).

Conventions:
  - tokens indexed i,j in {0,...,n-1}
  - A[i,j] is weight used to update token i from token j (edge j -> i).
  - causal masking (default): j <= i.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from jaxtyping import Float
import numpy as np
from numpy.typing import NDArray
import pydantic
import structlog
import typer

from src.rollout.attention import AttentionConfig, make_multihead_attention
from src.rollout.content_prior import ContentPriorConfig, make_content_prior
from src.rollout.lambda_schedule import (
    LambdaScheduleConfig,
    make_lambda_schedule,
)
from src.rollout.mask import MaskConfig, make_mask
from src.rollout.util import (
    max_offdiag_mass,
    plot_lambda_schedule,
    plot_last_row_distribution,
    row_entropy,
)
from src.util.pydantic import load_config

logger = structlog.get_logger()


class ModelConfig(pydantic.BaseModel):
    """Configuration for the residual-aware attention rollout model."""

    sequence_length: pydantic.PositiveInt = pydantic.Field(
        description="Sequence length (number of tokens)",
    )
    num_layers: pydantic.PositiveInt = pydantic.Field(
        description="Number of layers",
    )

    mask_config: MaskConfig = pydantic.Field(
        default_factory=MaskConfig,
        description="Configuration for the attention mask",
    )
    attention_config: AttentionConfig = pydantic.Field(
        description="Configuration for the attention mechanism",
    )
    content_config: ContentPriorConfig = pydantic.Field(
        description="Configuration for the content prior",
    )
    lambda_schedule_config: LambdaScheduleConfig = pydantic.Field(
        description="Configuration for the lambda schedule",
    )

    save_dir: Path | None = pydantic.Field(
        default=None,
        description="Directory to save plots and outputs",
    )

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.model_validator(mode="after")
    def validate_attention_layers(self) -> "ModelConfig":
        """Validates that attention config matches number of layers."""
        if len(self.attention_config.alibi_slopes) not in (1, self.num_layers):
            raise ValueError(
                "alibi_slopes outer list length must be 1 or match num_layers",
            )

        if len(self.content_config.diagonal_scores) not in (1, self.num_layers):
            raise ValueError(
                "content prior diagonal_scores outer list length must be 1 or match num_layers",
            )

        if len(self.content_config.base_scores) not in (1, self.num_layers):
            raise ValueError(
                "content prior base_scores outer list length must be 1 or match num_layers",
            )

        return self


def add_residual(
    attention_scores: Float[NDArray, "sequence_length sequence_length"],
    attention_weight: float,
) -> Float[NDArray, "sequence_length sequence_length"]:
    """
    Residual operator: R = (1-λ)I + λA.

    Args:
        attention_scores: Attention scores matrix A.
        attention_weight: Residual weight lambda in [0, 1].

    Returns:
        Residual operator R.

    """
    assert 0.0 <= attention_weight <= 1.0

    sequence_length, _ = attention_scores.shape

    return (1.0 - attention_weight) * np.eye(
        sequence_length,
    ) + attention_weight * attention_scores


def rollout(
    layer_scores_list: Sequence[Float[NDArray, "sequence_length sequence_length"]],
) -> Float[NDArray, "sequence_length sequence_length"]:
    """
    Compute P = R_T @ ... @ R_0 (assuming Rs is ordered [R_0, R_1, ..., R_T]).

    Args:
        layer_scores_list: Sequence of layer score matrices R^(t).

    Returns:
        The rollout matrix P^(T).

    """
    sequence_length, _ = layer_scores_list[0].shape
    scores_product = np.eye(sequence_length, dtype=float)

    for layer_scores in layer_scores_list:
        scores_product = layer_scores @ scores_product

    return scores_product


@dataclass
class ModelResults:
    """Results of the model computation."""

    attention_scores_list: list[
        Float[NDArray, "sequence_length sequence_length"]
    ]  # List of effective attention matrices A^(t)
    layer_scores_list: list[
        Float[NDArray, "sequence_length sequence_length"]
    ]  # List of residual operators R^(t)
    scores_product: Float[
        NDArray,
        "sequence_length sequence_length",
    ]  # Rollout matrix P^(T)
    lambdas: Float[NDArray, "num_layers"]  # Array lambda_t


def compute_model_results(
    config: ModelConfig,
) -> ModelResults:
    """
    Computes the model results including attention scores, layer scores, and rollout matrix.

    Args:
        config: ModelConfig containing all necessary configurations.

    Returns:
        ModelResults containing attention scores, layer scores, rollout matrix, and lambda schedule.

    """
    sequence_length, num_layers = config.sequence_length, config.num_layers
    mask = make_mask(config.mask_config, sequence_length)

    lambdas = make_lambda_schedule(config.lambda_schedule_config, num_layers)

    attention_scores_list: list[Float[NDArray, "sequence_length sequence_length"]] = []
    layer_scores_list: list[Float[NDArray, "sequence_length sequence_length"]] = []

    for t in range(num_layers):
        content_scores = make_content_prior(
            config.content_config,
            sequence_length=sequence_length,
            layer_idx=t,
            num_heads=config.attention_config.num_heads,
        )

        attention_scores = make_multihead_attention(
            config.attention_config,
            mask=mask,
            content_scores=content_scores,
            layer_index=t,
        )

        layer_scores = add_residual(attention_scores, attention_weight=lambdas[t])

        attention_scores_list.append(attention_scores)
        layer_scores_list.append(layer_scores)

    scores_product = rollout(layer_scores_list)

    return ModelResults(
        attention_scores_list,
        layer_scores_list,
        scores_product,
        lambdas,
    )


def main(config_file: Path) -> None:
    """
    Main function to compute and log rollout results.

    Args:
        config_file: Path to the model configuration file.

    """
    config = load_config(config_file, ModelConfig)

    model_results = compute_model_results(config)

    logger.info("=== Summary ===")
    logger.info(config)
    logger.info(
        f"row sums max |sum-1| in P: {float(np.max(np.abs(model_results.scores_product.sum(axis=1) - 1.0))):.3e}",
    )
    logger.info(
        f"min_i P[i,i]: {float(np.min(np.diag(model_results.scores_product))):.6f}",
    )
    logger.info(
        f"max_i P[i,0]: {float(np.max(model_results.scores_product[:, 0])):.6f}",
    )
    logger.info(
        f"mean row entropy: {float(np.mean(row_entropy(model_results.scores_product))):.6f} nats",
    )

    # Approximate theoretical control quantity sum_t lambda_t * delta_t
    deltas = np.array(
        [max_offdiag_mass(A) for A in model_results.attention_scores_list],
        dtype=float,
    )
    logger.info(
        f"sum_t lambda_t * delta_t: {float(np.sum(model_results.lambdas * deltas)):.6f}",
    )
    # Plots
    plot_lambda_schedule(
        model_results.lambdas,
        save_dir=config.save_dir,
    )
    plot_last_row_distribution(
        model_results.scores_product,
        save_dir=config.save_dir,
    )


if __name__ == "__main__":
    typer.run(main)
