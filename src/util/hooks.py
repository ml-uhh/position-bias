"""Hooks to capture intermediate computations in Falcon modules."""

from collections.abc import Callable
from typing import Any

from jaxtyping import Float
import structlog
import torch
from torch import nn
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomForCausalLM
from transformers.models.falcon.modeling_falcon import (
    FalconDecoderLayer,
    FalconForCausalLM,
)
from transformers.models.mpt.modeling_mpt import MptBlock, MptForCausalLM
import wandb

from src.metrics.metrics import (
    effective_attention_strength,
    effective_residual_strength,
)

logger = structlog.get_logger()


def create_attention_hook(
    batch_idx: int,
    layer_idx: int,
    *,
    get_attention: Callable[
        [Any],
        Float[torch.Tensor, "batch_size sequence_length features"],
    ],
    storage: dict[str, Float[torch.Tensor, "batch_size sequence_length features"]],
) -> Callable[
    [
        nn.Module,
        Any,
        tuple[Float[torch.Tensor, "batch_size sequence_length features"], Any],
    ],
    None,
]:
    """Create a hook to capture attention output of an attention module."""

    def hook(
        _module: nn.Module,
        _input_data: Any,  # noqa: ANN401
        output: Any,  # noqa: ANN401
    ) -> None:
        """Capture the attention output."""
        storage[f"attention_output_{batch_idx}_{layer_idx}"] = (
            get_attention(output).detach().cpu()
        )

    return hook


def create_residual_hook(
    batch_idx: int,
    layer_idx: int,
    *,
    get_residual: Callable[
        [Any],
        Float[torch.Tensor, "batch_size sequence_length features"],
    ],
    storage: dict[str, Float[torch.Tensor, "batch_size sequence_length features"]],
    metrics: dict[int, dict[str, float | int]],
) -> Callable[
    [
        nn.Module,
        tuple[Float[torch.Tensor, "batch_size sequence_length features"]],
        Any,
    ],
    None,
]:
    """Create a hook to capture layer input and compute metrics."""

    def hook(
        _module: nn.Module,
        input_data: Any,  # noqa: ANN401
        _output: Any,  # noqa: ANN401
    ) -> None:
        """Capture the layer input (residual) and compute metrics."""
        residual = get_residual(input_data).detach().cpu().float()

        attention_key = f"attention_output_{batch_idx}_{layer_idx}"
        assert attention_key in storage, (
            f"Attention output for batch {batch_idx} layer {layer_idx} not found in storage."
        )
        attention_output = storage[attention_key].to(residual.device).float()

        if torch.isnan(residual).any():
            logger.warning(
                f"NaN detected in residual at batch {batch_idx}, layer {layer_idx}",
            )
        if torch.isnan(attention_output).any():
            logger.warning(
                f"NaN detected in attention_output at batch {batch_idx}, layer {layer_idx}",
            )

        residual_all_zero = (residual == 0).all().item()
        attention_all_zero = (attention_output == 0).all().item()
        if residual_all_zero:
            logger.warning(
                f"All-zero residual tensor at batch {batch_idx}, layer {layer_idx}. "
                f"Possible bfloat16 underflow - consider using torch_dtype=torch.float16 or float32.",
            )
        if attention_all_zero:
            logger.warning(
                f"All-zero attention_output tensor at batch {batch_idx}, layer {layer_idx}. "
                f"Possible bfloat16 underflow - consider using torch_dtype=torch.float16 or float32.",
            )

        attn_strength = effective_attention_strength(
            residual=residual,
            attention_output=attention_output,
        )
        res_strength = effective_residual_strength(
            residual=residual,
            attention_output=attention_output,
        )

        attn_mean = attn_strength.mean().item()
        res_mean = res_strength.mean().item()

        # logger.info(
        #     f"Batch {batch_idx} layer {layer_idx} metrics",
        #     effective_attention_strength=attn_mean,
        #     effective_residual_strength=res_mean,
        # )

        wandb.log(
            {
                f"hook/layer_{layer_idx}/attn_strength_mean": attn_mean,
                f"hook/layer_{layer_idx}/res_strength_mean": res_mean,
            },
            step=batch_idx,
        )

        layer_metrics = metrics.setdefault(
            layer_idx,
            {
                "attn_sum": 0.0,
                "attn_sum_sq": 0.0,
                "attn_count": 0,
                "res_sum": 0.0,
                "res_sum_sq": 0.0,
                "res_count": 0,
            },
        )

        attn_flat = attn_strength.flatten().to(dtype=torch.float64)
        layer_metrics["attn_sum"] += attn_flat.sum().item()
        layer_metrics["attn_sum_sq"] += (attn_flat**2).sum().item()
        layer_metrics["attn_count"] += attn_flat.numel()

        res_flat = res_strength.flatten().to(dtype=torch.float64)
        layer_metrics["res_sum"] += res_flat.sum().item()
        layer_metrics["res_sum_sq"] += (res_flat**2).sum().item()
        layer_metrics["res_count"] += res_flat.numel()

        del storage[attention_key]

    return hook


def register_hooks(
    layer: nn.Module,
    attention: nn.Module,
    *,
    batch_idx: int,
    layer_idx: int,
    get_attention: Callable[
        [Any],
        Float[torch.Tensor, "batch_size sequence_length features"],
    ],
    get_residual: Callable[
        [Any],
        Float[torch.Tensor, "batch_size sequence_length features"],
    ],
    hook_storage: dict[
        str,
        Float[torch.Tensor, "batch_size sequence_length features"],
    ],
    metrics: dict[int, dict[str, float | int]],
) -> tuple[torch.utils.hooks.RemovableHandle, torch.utils.hooks.RemovableHandle]:
    """Register hooks for attention and residual on a given layer."""
    handle_attention = attention.register_forward_hook(
        create_attention_hook(
            batch_idx,
            layer_idx,
            get_attention=get_attention,
            storage=hook_storage,
        ),
    )
    handle_residual = layer.register_forward_hook(
        create_residual_hook(
            batch_idx,
            layer_idx,
            get_residual=get_residual,
            storage=hook_storage,
            metrics=metrics,
        ),
    )
    return handle_attention, handle_residual


def register_metric_hooks(
    model: torch.nn.Module,
    batch_idx: int,
    metrics: dict[int, dict[str, float | int]],
) -> list[torch.utils.hooks.RemovableHandle]:
    """
    Register hooks to log metrics for each transformer block.

    Args:
        model: The model to register hooks on.
        batch_idx: The current batch index.
        metrics: Dictionary to store computed metrics.

    Returns:
        List of hook handles that can be used to unregister the hooks.

    """
    hook_storage: dict[
        str,
        Float[torch.Tensor, "batch_size sequence_length features"],
    ] = {}

    hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    match model:
        case FalconForCausalLM():
            for layer_idx, layer in enumerate(model.transformer.h):
                assert isinstance(layer, FalconDecoderLayer)

                hook_handles.extend(
                    register_hooks(
                        layer,
                        layer.self_attention,
                        batch_idx=batch_idx,
                        layer_idx=layer_idx,
                        get_attention=lambda output: output[0],
                        get_residual=lambda input_data: input_data[0],
                        hook_storage=hook_storage,
                        metrics=metrics,
                    ),
                )
        case BloomForCausalLM():
            for layer_idx, layer in enumerate(model.transformer.h):
                assert isinstance(layer, BloomBlock)

                # Make sure that the path that involves dense is used:
                assert not (
                    layer.self_attention.pretraining_tp > 1
                    and layer.self_attention.slow_but_exact
                ), (
                    "Dense layer not used in this configuration. Can not use hook to get attention output for BLOOM model."
                )

                hook_handles.extend(
                    register_hooks(
                        layer,
                        layer.self_attention.dense,
                        batch_idx=batch_idx,
                        layer_idx=layer_idx,
                        get_attention=lambda output: output,
                        get_residual=lambda input_data: input_data[0],
                        hook_storage=hook_storage,
                        metrics=metrics,
                    ),
                )
        case MptForCausalLM():
            for layer_idx, layer in enumerate(model.transformer.blocks):
                assert isinstance(layer, MptBlock)

                hook_handles.extend(
                    register_hooks(
                        layer,
                        layer.attn,
                        batch_idx=batch_idx,
                        layer_idx=layer_idx,
                        get_attention=lambda output: output[0],
                        get_residual=lambda input_data: input_data[0],
                        hook_storage=hook_storage,
                        metrics=metrics,
                    ),
                )
        case _:
            raise TypeError(
                f"Model type {type(model)} not supported for hook registration.",
            )

    return hook_handles


def unregister_hooks(hook_handles: list[torch.utils.hooks.RemovableHandle]) -> None:
    """
    Unregister all hooks.

    Args:
        hook_handles: List of hook handles returned by register_metric_hooks.

    """
    for handle in hook_handles:
        handle.remove()
