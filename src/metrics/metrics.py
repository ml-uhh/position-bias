"""Metrics of a transformer model."""

from typing import Literal

from jaxtyping import Float
import numpy as np
from scipy.stats import wasserstein_distance
import structlog
import torch

logger = structlog.get_logger()

NormType = Literal["frobenius", "spectral"]


def normalized_wasserstein_1(left: np.ndarray, right: np.ndarray) -> float:
    """Return discrete one-dimensional Wasserstein distance divided by ``n-1``."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError(f"profiles must be same-length vectors, got {left.shape} and {right.shape}")
    if left.size < 2 or left.sum() <= 0 or right.sum() <= 0:
        raise ValueError("profiles must contain at least two positions and positive total mass")
    positions = np.arange(left.size, dtype=float)
    return float(
        wasserstein_distance(
            positions,
            positions,
            u_weights=left / left.sum(),
            v_weights=right / right.sum(),
        )
        / (left.size - 1),
    )


def _compute_norm(
    tensor: Float[torch.Tensor, "batch_size sequence_length features"],
    norm_type: NormType,
) -> Float[torch.Tensor, "batch_size sequence_length"]:
    """Compute per-position norm: L2 (frobenius) or operator 2-norm (spectral) broadcast to all positions."""
    if norm_type == "frobenius":
        return torch.norm(tensor, dim=2)
    _, spectral, _ = torch.svd_lowrank(tensor, q=1, niter=6)
    return spectral.expand(-1, tensor.shape[1])


def effective_attention_strength(
    residual: Float[torch.Tensor, "batch_size sequence_length features"],
    attention_output: Float[torch.Tensor, "batch_size sequence_length features"],
    norm_type: NormType,
) -> Float[torch.Tensor, "batch_size"]:
    """
    Compute the effective attention strength of a transformer model.

    If the output of a transformer block is given by

    x_t + Attn(Norm(x_t)) + MLP(.)

    and we denote the attention part as

    a_t = Attn(Norm(x_t)),

    the effective attention strength is defined as

    lambda_t_eff = E[||a_t|| / (||x_t|| + ||a_t||)]

    Args:
        residual: The residual tensor.
        attention_output: The attention output tensor.
        norm_type: Norm to use: "frobenius" for per-position L2 norms, "spectral" for operator 2-norm (largest singular value).

    Returns:
        The mean effective attention strength of the layer.

    """
    residual_norm = _compute_norm(residual, norm_type)
    attention_norm = _compute_norm(attention_output, norm_type)

    if torch.isnan(residual_norm).any():
        logger.warning("NaN values found in residual_norm")
    if torch.isnan(attention_norm).any():
        logger.warning("NaN values found in attention_norm")

    zero_residual_mask = residual_norm == 0
    zero_attention_mask = attention_norm == 0
    both_zero_mask = zero_residual_mask & zero_attention_mask

    if both_zero_mask.any():
        num_degenerate = both_zero_mask.sum().item()
        total_positions = both_zero_mask.numel()
        logger.warning(
            f"Degenerate case: both residual and attention norms are zero "
            f"at {num_degenerate}/{total_positions} positions. "
            f"This may indicate bfloat16 underflow or all-zero activations.",
            function_name="effective_attention_strength",
        )

    denominator = residual_norm + attention_norm
    denominator = denominator.clamp_min(1e-10)  # Prevent division by zero

    effective_strength = attention_norm / denominator

    return torch.mean(effective_strength, dim=1)


def effective_residual_strength(
    residual: Float[torch.Tensor, "batch_size sequence_length features"],
    attention_output: Float[torch.Tensor, "batch_size sequence_length features"],
    norm_type: NormType = "frobenius",
) -> Float[torch.Tensor, "batch_size"]:
    """
    Compute the effective residual strength of a transformer model.

    If the output of a transformer block is given by
    x_t + Attn(Norm(x_t)) + MLP(.)

    and we denote the attention part as
    a_t = Attn(Norm(x_t)),
    the effective residual strength is defined as

    lambda_t_eff = E[||x_t|| / (||x_t|| + ||a_t||)]

    Args:
        residual: The residual tensor.
        attention_output: The attention output tensor.
        norm_type: Norm to use — "frobenius" for per-position L2 norms,
            "spectral" for operator 2-norm (largest singular value).

    Returns:
        The mean effective residual strength of the layer.

    """
    residual_norm = _compute_norm(residual, norm_type)
    attention_norm = _compute_norm(attention_output, norm_type)

    denominator = residual_norm + attention_norm
    denominator = denominator.clamp_min(1e-10)  # Prevent division by zero

    effective_strength = residual_norm / denominator

    return torch.mean(effective_strength, dim=1)
