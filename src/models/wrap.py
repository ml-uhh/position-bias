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
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2DecoderLayer,
    Gemma2ForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer,
    MistralForCausalLM,
)
from transformers.models.mpt.modeling_mpt import MptBlock, MptForCausalLM
from transformers.models.olmo3.modeling_olmo3 import Olmo3DecoderLayer, Olmo3ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM
from transformers.models.smollm3.modeling_smollm3 import (
    SmolLM3DecoderLayer,
    SmolLM3ForCausalLM,
)

from src.models.bloom import BloomAttentionWrapper
from src.models.falcon import FalconAttentionWrapper
from src.models.gemma import Gemma2AttentionWrapper
from src.models.llama import LlamaAttentionWrapper
from src.models.mistral import MistralAttentionWrapper
from src.models.mpt import MptAttentionWrapper
from src.models.olmo3 import Olmo3AttentionWrapper
from src.models.qwen2 import Qwen2AttentionWrapper
from src.models.qwen3 import Qwen3AttentionWrapper
from src.models.smollm3 import SmolLM3AttentionWrapper

logger = structlog.get_logger()


def wrap_attention_layers(model: nn.Module, sequence_length: int) -> None:  # noqa: C901, PLR0912, PLR0915
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
        case Gemma2ForCausalLM():
            for layer_idx, layer in enumerate(model.model.layers):
                assert isinstance(layer, Gemma2DecoderLayer)

                attn_weights = layer.self_attn.state_dict()
                device = layer.self_attn.q_proj.weight.device
                dtype = layer.self_attn.q_proj.weight.dtype
                layer.self_attn = Gemma2AttentionWrapper(
                    config=model.config,
                    layer_idx=layer_idx,
                    sequence_length=sequence_length,
                ).to(device=device, dtype=dtype)
                layer.self_attn.load_state_dict(attn_weights)
        case LlamaForCausalLM():
            for layer_idx, layer in enumerate(model.model.layers):
                assert isinstance(layer, LlamaDecoderLayer)

                attn_weights = layer.self_attn.state_dict()
                device = layer.self_attn.q_proj.weight.device
                dtype = layer.self_attn.q_proj.weight.dtype
                layer.self_attn = LlamaAttentionWrapper(
                    config=model.config,
                    layer_idx=layer_idx,
                    sequence_length=sequence_length,
                ).to(device=device, dtype=dtype)
                layer.self_attn.load_state_dict(attn_weights)
        case MistralForCausalLM():
            for layer_idx, layer in enumerate(model.model.layers):
                assert isinstance(layer, MistralDecoderLayer)

                attn_weights = layer.self_attn.state_dict()
                device = layer.self_attn.q_proj.weight.device
                dtype = layer.self_attn.q_proj.weight.dtype
                layer.self_attn = MistralAttentionWrapper(
                    config=model.config,
                    layer_idx=layer_idx,
                    sequence_length=sequence_length,
                ).to(device=device, dtype=dtype)
                layer.self_attn.load_state_dict(attn_weights)
        case Qwen2ForCausalLM():
            for layer_idx, layer in enumerate(model.model.layers):
                assert isinstance(layer, Qwen2DecoderLayer)

                attn_weights = layer.self_attn.state_dict()
                device = layer.self_attn.q_proj.weight.device
                dtype = layer.self_attn.q_proj.weight.dtype
                layer.self_attn = Qwen2AttentionWrapper(
                    config=model.config,
                    layer_idx=layer_idx,
                    sequence_length=sequence_length,
                ).to(device=device, dtype=dtype)
                layer.self_attn.load_state_dict(attn_weights)
        case Olmo3ForCausalLM():
            for layer_idx, layer in enumerate(model.model.layers):
                assert isinstance(layer, Olmo3DecoderLayer)

                attn_weights = layer.self_attn.state_dict()
                device = layer.self_attn.q_proj.weight.device
                dtype = layer.self_attn.q_proj.weight.dtype
                layer.self_attn = Olmo3AttentionWrapper(
                    config=model.config,
                    layer_idx=layer_idx,
                    sequence_length=sequence_length,
                ).to(device=device, dtype=dtype)
                layer.self_attn.load_state_dict(attn_weights)
        case Qwen3ForCausalLM():
            for layer_idx, layer in enumerate(model.model.layers):
                assert isinstance(layer, Qwen3DecoderLayer)

                attn_weights = layer.self_attn.state_dict()
                device = layer.self_attn.q_proj.weight.device
                dtype = layer.self_attn.q_proj.weight.dtype
                layer.self_attn = Qwen3AttentionWrapper(
                    config=model.config,
                    layer_idx=layer_idx,
                    sequence_length=sequence_length,
                ).to(device=device, dtype=dtype)
                layer.self_attn.load_state_dict(attn_weights)
        case SmolLM3ForCausalLM():
            for layer_idx, layer in enumerate(model.model.layers):
                assert isinstance(layer, SmolLM3DecoderLayer)

                attn_weights = layer.self_attn.state_dict()
                device = layer.self_attn.q_proj.weight.device
                dtype = layer.self_attn.q_proj.weight.dtype
                layer.self_attn = SmolLM3AttentionWrapper(
                    config=model.config,
                    layer_idx=layer_idx,
                    sequence_length=sequence_length,
                ).to(device=device, dtype=dtype)
                layer.self_attn.load_state_dict(attn_weights)
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


def save_wrapper_stats(  # noqa: C901, PLR0912, PLR0915
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
        case Gemma2ForCausalLM():
            global_qk_mean = torch.zeros(
                len(model.model.layers),
                model.config.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            global_qk_var = torch.zeros(
                len(model.model.layers),
                model.config.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            for layer_idx, layer in enumerate(model.model.layers):
                assert isinstance(layer, Gemma2DecoderLayer)
                assert isinstance(layer.self_attn, Gemma2AttentionWrapper)

                logger.info(f"Saving stats for layer {layer_idx}")
                qk_mean = layer.self_attn.qk_sum / layer.self_attn.qk_count
                qk_var = (
                    layer.self_attn.qk_sum_sq / layer.self_attn.qk_count - qk_mean**2
                )
                global_qk_mean[layer_idx] = qk_mean
                global_qk_var[layer_idx] = qk_var
            torch.save(
                global_qk_mean,  # pyright: ignore[reportAttributeAccessIssue]
                save_path / "qk_mean.pt",
            )
            torch.save(
                global_qk_var,
                save_path / "qk_var.pt",
            )
        case LlamaForCausalLM():
            global_qk_mean = torch.zeros(
                len(model.model.layers),
                model.config.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            global_qk_var = torch.zeros(
                len(model.model.layers),
                model.config.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            for layer_idx, layer in enumerate(model.model.layers):
                assert isinstance(layer, LlamaDecoderLayer)
                assert isinstance(layer.self_attn, LlamaAttentionWrapper)

                logger.info(f"Saving stats for layer {layer_idx}")
                qk_mean = layer.self_attn.qk_sum / layer.self_attn.qk_count
                qk_var = (
                    layer.self_attn.qk_sum_sq / layer.self_attn.qk_count - qk_mean**2
                )
                global_qk_mean[layer_idx] = qk_mean
                global_qk_var[layer_idx] = qk_var
            torch.save(
                global_qk_mean,  # pyright: ignore[reportAttributeAccessIssue]
                save_path / "qk_mean.pt",
            )
            torch.save(
                global_qk_var,
                save_path / "qk_var.pt",
            )
        case MistralForCausalLM():
            global_qk_mean = torch.zeros(
                len(model.model.layers),
                model.config.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            global_qk_var = torch.zeros(
                len(model.model.layers),
                model.config.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            for layer_idx, layer in enumerate(model.model.layers):
                assert isinstance(layer, MistralDecoderLayer)
                assert isinstance(layer.self_attn, MistralAttentionWrapper)

                logger.info(f"Saving stats for layer {layer_idx}")
                qk_mean = layer.self_attn.qk_sum / layer.self_attn.qk_count
                qk_var = (
                    layer.self_attn.qk_sum_sq / layer.self_attn.qk_count - qk_mean**2
                )
                global_qk_mean[layer_idx] = qk_mean
                global_qk_var[layer_idx] = qk_var
            torch.save(
                global_qk_mean,  # pyright: ignore[reportAttributeAccessIssue]
                save_path / "qk_mean.pt",
            )
            torch.save(
                global_qk_var,
                save_path / "qk_var.pt",
            )
        case Qwen2ForCausalLM():
            global_qk_mean = torch.zeros(
                len(model.model.layers),
                model.config.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            global_qk_var = torch.zeros(
                len(model.model.layers),
                model.config.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            global_prob_mean = torch.zeros(
                len(model.model.layers),
                model.config.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            for layer_idx, layer in enumerate(model.model.layers):
                assert isinstance(layer, Qwen2DecoderLayer)
                assert isinstance(layer.self_attn, Qwen2AttentionWrapper)

                logger.info(f"Saving stats for layer {layer_idx}")
                qk_mean = layer.self_attn.qk_sum / layer.self_attn.qk_count
                qk_var = (
                    layer.self_attn.qk_sum_sq / layer.self_attn.qk_count - qk_mean**2
                )
                global_qk_mean[layer_idx] = qk_mean
                global_qk_var[layer_idx] = qk_var
                global_prob_mean[layer_idx] = (
                    layer.self_attn.attention_probs_sum
                    / layer.self_attn.attention_probs_count
                )
            torch.save(
                global_qk_mean,  # pyright: ignore[reportAttributeAccessIssue]
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
        case Olmo3ForCausalLM():
            global_qk_mean = torch.zeros(
                len(model.model.layers),
                model.config.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            global_qk_var = torch.zeros(
                len(model.model.layers),
                model.config.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            for layer_idx, layer in enumerate(model.model.layers):
                assert isinstance(layer, Olmo3DecoderLayer)
                assert isinstance(layer.self_attn, Olmo3AttentionWrapper)

                logger.info(f"Saving stats for layer {layer_idx}")
                qk_mean = layer.self_attn.qk_sum / layer.self_attn.qk_count
                qk_var = (
                    layer.self_attn.qk_sum_sq / layer.self_attn.qk_count - qk_mean**2
                )
                global_qk_mean[layer_idx] = qk_mean
                global_qk_var[layer_idx] = qk_var
            torch.save(
                global_qk_mean,  # pyright: ignore[reportAttributeAccessIssue]
                save_path / "qk_mean.pt",
            )
            torch.save(
                global_qk_var,
                save_path / "qk_var.pt",
            )
        case Qwen3ForCausalLM():
            global_qk_mean = torch.zeros(
                len(model.model.layers),
                model.config.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            global_qk_var = torch.zeros(
                len(model.model.layers),
                model.config.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            for layer_idx, layer in enumerate(model.model.layers):
                assert isinstance(layer, Qwen3DecoderLayer)
                assert isinstance(layer.self_attn, Qwen3AttentionWrapper)

                logger.info(f"Saving stats for layer {layer_idx}")
                qk_mean = layer.self_attn.qk_sum / layer.self_attn.qk_count
                qk_var = (
                    layer.self_attn.qk_sum_sq / layer.self_attn.qk_count - qk_mean**2
                )
                global_qk_mean[layer_idx] = qk_mean
                global_qk_var[layer_idx] = qk_var
            torch.save(
                global_qk_mean,  # pyright: ignore[reportAttributeAccessIssue]
                save_path / "qk_mean.pt",
            )
            torch.save(
                global_qk_var,
                save_path / "qk_var.pt",
            )
        case SmolLM3ForCausalLM():
            global_qk_mean = torch.zeros(
                len(model.model.layers),
                model.config.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            global_qk_var = torch.zeros(
                len(model.model.layers),
                model.config.num_attention_heads,
                sequence_length,
                sequence_length,
            )
            for layer_idx, layer in enumerate(model.model.layers):
                assert isinstance(layer, SmolLM3DecoderLayer)
                assert isinstance(layer.self_attn, SmolLM3AttentionWrapper)

                logger.info(f"Saving stats for layer {layer_idx}")
                qk_mean = layer.self_attn.qk_sum / layer.self_attn.qk_count
                qk_var = (
                    layer.self_attn.qk_sum_sq / layer.self_attn.qk_count - qk_mean**2
                )
                global_qk_mean[layer_idx] = qk_mean
                global_qk_var[layer_idx] = qk_var
            torch.save(
                global_qk_mean,  # pyright: ignore[reportAttributeAccessIssue]
                save_path / "qk_mean.pt",
            )
            torch.save(
                global_qk_var,
                save_path / "qk_var.pt",
            )
        case _:
            logger.warning(
                f"Model type {type(model)} not supported for saving wrapper statistics.",
            )


def save_attribution_stats(
    attribution: torch.Tensor,
    save_path: Path,
    per_head_attribution: torch.Tensor | None = None,
) -> None:
    """
    Save attribution statistics to disk.

    Writes the attribution tensor to disk if it is non-empty, otherwise logs
    a warning and skips saving.

    Args:
        attribution: Attribution tensor of shape (num_layers, seq_len) giving,
            for each decoder layer, the L2 norm over hidden of
            d(||layer_output[:, -1, :]||_2) / d(inputs_embeds) averaged over
            batches per input-token position.
        save_path: Directory to save the tensor to.
        per_head_attribution: Optional per-head attribution tensor of shape
            (num_layers, seq_len, num_heads) giving, for each decoder layer
            and attention head, the L2 norm over head_dim of
            d(||final_hidden[:, -1, :]||_2) / d(o_proj_input_head) averaged
            over batches per input-token position.

    """
    if attribution.numel() == 0:
        logger.warning("Attribution tensor is empty. Skipping save.")
        return
    torch.save(attribution, save_path / "input_grad_l2_mean.pt")
    if per_head_attribution is not None:
        if per_head_attribution.numel() == 0:
            logger.warning("Per-head attribution tensor is empty. Skipping save.")
            return
        torch.save(per_head_attribution, save_path / "input_grad_l2_per_head.pt")
