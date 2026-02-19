"""Metrics of a transformer model."""

from jaxtyping import Float
import structlog
import torch

logger = structlog.get_logger()


def effective_attention_strength(
    residual: Float[torch.Tensor, "batch_size sequence_length features"],
    attention_output: Float[torch.Tensor, "batch_size sequence_length features"],
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

    Returns:
        The mean effective attention strength of the layer.

    """
    residual_norm: Float[torch.Tensor, "batch_size sequence_length"] = torch.norm(
        residual,
        dim=2,
    )
    attention_norm: Float[torch.Tensor, "batch_size sequence_length"] = torch.norm(
        attention_output,
        dim=2,
    )

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

    Returns:
        The mean effective residual strength of the layer.

    """
    residual_norm: Float[torch.Tensor, "batch_size sequence_length"] = torch.norm(
        residual,
        dim=2,
    )
    attention_norm: Float[torch.Tensor, "batch_size sequence_length"] = torch.norm(
        attention_output,
        dim=2,
    )

    denominator = residual_norm + attention_norm
    denominator = denominator.clamp_min(1e-10)  # Prevent division by zero

    effective_strength = residual_norm / denominator

    return torch.mean(effective_strength, dim=1)
