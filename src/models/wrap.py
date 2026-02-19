"""Functions to wrap attention layers in models."""

from pathlib import Path

import structlog
import torch
from torch import nn
from transformers.models.bloom.modeling_bloom import BloomBlock, BloomForCausalLM
from transformers.models.falcon.modeling_falcon import (
    FalconDecoderLayer,
    FalconForCausalLM,
)
from transformers.models.mpt.modeling_mpt import MptBlock, MptForCausalLM

from src.models.bloom import BloomAttentionWrapper
from src.models.falcon import FalconAttentionWrapper
from src.models.mpt import MptAttentionWrapper

logger = structlog.get_logger()


def wrap_attention_layers(model: nn.Module, sequence_length: int) -> None:
    """
    Replace supported model attention layers with wrapper implementations.

    Traverses the model and replaces backbone-specific attention layers with corresponding wrapper classes configured to collect statistics.

    Args:
        model: The model to modify in-place.
        sequence_length: Sequence length used to allocate wrapper buffers.

    """
    match model:
        case BloomForCausalLM():
            for layer_idx, layer in enumerate(model.transformer.h):
                assert isinstance(layer, BloomBlock)

                attn_weights = layer.self_attention.state_dict()
                device = layer.self_attention.query_key_value.weight.device
                dtype = layer.self_attention.query_key_value.weight.dtype
                layer.self_attention = BloomAttentionWrapper(
                    config=model.config,
                    layer_idx=layer_idx,
                    sequence_length=sequence_length,
                ).to(device=device, dtype=dtype)
                layer.self_attention.load_state_dict(attn_weights)
        case FalconForCausalLM():
            for layer_idx, layer in enumerate(model.transformer.h):
                assert isinstance(layer, FalconDecoderLayer)

                attn_weights = layer.self_attention.state_dict()
                device = layer.self_attention.query_key_value.weight.device
                dtype = layer.self_attention.query_key_value.weight.dtype
                layer.self_attention = FalconAttentionWrapper(
                    config=model.config,
                    layer_idx=layer_idx,
                    sequence_length=sequence_length,
                ).to(device=device, dtype=dtype)
                layer.self_attention.load_state_dict(attn_weights)
        case MptForCausalLM():
            for layer_idx, layer in enumerate(model.transformer.blocks):
                assert isinstance(layer, MptBlock)

                attn_weights = layer.attn.state_dict()
                device = layer.attn.Wqkv.weight.device
                dtype = layer.attn.Wqkv.weight.dtype
                layer.attn = MptAttentionWrapper(
                    config=model.config,
                    layer_idx=layer_idx,
                    sequence_length=sequence_length,
                ).to(device=device, dtype=dtype)
                layer.attn.load_state_dict(attn_weights)
        case _:
            logger.warning(
                f"Model type {type(model)} not supported for wrapping attention layers.",
            )


def _compute_layer_stats(
    qk_sum: torch.Tensor,
    qk_sum_sq: torch.Tensor,
    qk_count: int,
    attention_probs: torch.Tensor,
    attention_probs_sq: torch.Tensor,
    attention_probs_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qk_mean = qk_sum / qk_count
    qk_var = qk_sum_sq / qk_count - qk_mean**2

    prob_mean = attention_probs / attention_probs_count
    prob_var = attention_probs_sq / attention_probs_count - prob_mean**2

    return qk_mean, qk_var, prob_mean, prob_var


def _save_wrapper_stats_to_file(
    save_path: Path,
    *,
    global_qk_mean: torch.Tensor,
    global_qk_var: torch.Tensor,
    global_prob_mean: torch.Tensor,
    global_prob_var: torch.Tensor,
) -> None:
    torch.save(
        global_qk_mean,
        save_path / "qk_mean.pt",
    )
    torch.save(
        global_qk_var,
        save_path / "qk_var.pt",
    )
    torch.save(
        global_prob_mean,
        save_path / "attention_probs_mean.pt",
    )
    torch.save(
        global_prob_var,
        save_path / "attention_probs_var.pt",
    )


def save_wrapper_stats(
    model: nn.Module,
    save_path: Path,
    sequence_length: int,
) -> None:
    """
    Save wrapper statistics aggregated across layers to disk.

    Extracts per-layer qk mean and variance from wrapped attention layers and saves them as tensors under the provided path.

    Args:
        model: The model containing wrapped attention layers.
        save_path: Directory where statistics will be saved.
        sequence_length: Sequence length used to shape the output tensors.

    """
    match model:
        case BloomForCausalLM():
            shape = (
                len(model.transformer.h),
                model.config.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            global_qk_mean = torch.zeros(shape)
            global_qk_var = torch.zeros(shape)
            global_prob_mean = torch.zeros(shape)
            global_prob_var = torch.zeros(shape)

            for layer_idx, layer in enumerate(model.transformer.h):
                assert isinstance(layer, BloomBlock)
                assert isinstance(layer.self_attention, BloomAttentionWrapper)

                logger.info(f"Saving stats for layer {layer_idx}")
                (
                    global_qk_mean[layer_idx],
                    global_qk_var[layer_idx],
                    global_prob_mean[layer_idx],
                    global_prob_var[layer_idx],
                ) = _compute_layer_stats(
                    qk_sum=layer.self_attention.qk_sum,
                    qk_sum_sq=layer.self_attention.qk_sum_sq,
                    qk_count=layer.self_attention.qk_count,
                    attention_probs=layer.self_attention.attention_probs,
                    attention_probs_sq=layer.self_attention.attention_probs_sq,
                    attention_probs_count=layer.self_attention.attention_probs_count,
                )

            _save_wrapper_stats_to_file(
                save_path=save_path,
                global_qk_mean=global_qk_mean,
                global_qk_var=global_qk_var,
                global_prob_mean=global_prob_mean,
                global_prob_var=global_prob_var,
            )
        case FalconForCausalLM():
            shape = (
                len(model.transformer.h),
                model.config.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            global_qk_mean = torch.zeros(shape)
            global_qk_var = torch.zeros(shape)
            global_prob_mean = torch.zeros(shape)
            global_prob_var = torch.zeros(shape)

            for layer_idx, layer in enumerate(model.transformer.h):
                assert isinstance(layer, FalconDecoderLayer)
                assert isinstance(layer.self_attention, FalconAttentionWrapper)

                logger.info(f"Saving stats for layer {layer_idx}")
                (
                    global_qk_mean[layer_idx],
                    global_qk_var[layer_idx],
                    global_prob_mean[layer_idx],
                    global_prob_var[layer_idx],
                ) = _compute_layer_stats(
                    qk_sum=layer.self_attention.qk_sum,
                    qk_sum_sq=layer.self_attention.qk_sum_sq,
                    qk_count=layer.self_attention.qk_count,
                    attention_probs=layer.self_attention.attention_probs,
                    attention_probs_sq=layer.self_attention.attention_probs_sq,
                    attention_probs_count=layer.self_attention.attention_probs_count,
                )

            _save_wrapper_stats_to_file(
                save_path=save_path,
                global_qk_mean=global_qk_mean,
                global_qk_var=global_qk_var,
                global_prob_mean=global_prob_mean,
                global_prob_var=global_prob_var,
            )
        case MptForCausalLM():
            shape = (
                len(model.transformer.blocks),
                model.config.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            global_qk_mean = torch.zeros(shape)
            global_qk_var = torch.zeros(shape)
            global_prob_mean = torch.zeros(shape)
            global_prob_var = torch.zeros(shape)

            for layer_idx, layer in enumerate(model.transformer.blocks):
                assert isinstance(layer, MptBlock)
                assert isinstance(layer.attn, MptAttentionWrapper)

                logger.info(f"Saving stats for layer {layer_idx}")
                (
                    global_qk_mean[layer_idx],
                    global_qk_var[layer_idx],
                    global_prob_mean[layer_idx],
                    global_prob_var[layer_idx],
                ) = _compute_layer_stats(
                    qk_sum=layer.attn.qk_sum,
                    qk_sum_sq=layer.attn.qk_sum_sq,
                    qk_count=layer.attn.qk_count,
                    attention_probs=layer.attn.attention_probs,
                    attention_probs_sq=layer.attn.attention_probs_sq,
                    attention_probs_count=layer.attn.attention_probs_count,
                )

            _save_wrapper_stats_to_file(
                save_path=save_path,
                global_qk_mean=global_qk_mean,
                global_qk_var=global_qk_var,
                global_prob_mean=global_prob_mean,
                global_prob_var=global_prob_var,
            )
        case _:
            logger.warning(
                f"Model type {type(model)} not supported for saving wrapper statistics.",
            )


def save_attribution_stats(
    attribution: torch.Tensor,
    save_path: Path,
) -> None:
    """
    Save attribution statistics to disk.

    Writes the attribution tensor to disk if it is non-empty, otherwise logs a warning and skips saving.

    Args:
        attribution: Attribution tensor to save.
        save_path: Directory to save the tensor to.

    """
    if attribution.numel() == 0:
        logger.warning("Attribution tensor is empty. Skipping save.")
        return
    torch.save(attribution, save_path / "input_grad_l2_mean.pt")
