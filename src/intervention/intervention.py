r"""
local beta intervention on the attention branch, generalized across architectures.

Provides ``BetaScale``, a differentiable context manager that scales every supported model's
attention-branch output tensor by a fixed ``beta`` immediately before the residual addition, leaving
the residual stream, MLP, attention probabilities, and full block output otherwise untouched. Used to
test a pre-specified causal prediction of the rollout theory without fitting or recalibrating the
rollout against influence curves.

Requires a module whose own forward output *is* the attention-branch value immediately before the
residual add -- scaling its output there is exactly ``beta * Attn_t(X_t)``, whether that module is
the whole attention submodule (its residual add happens externally, in the block/layer's own
forward) or a child submodule inside it (if the parent adds the residual internally, but a child
computes the pre-residual value before that happens). ``_make_hook`` handles both a tuple-returning
module (scales element 0, leaves the rest -- e.g. attention weights -- untouched) and a
bare-tensor-returning module identically. Supported:

- MPT (``transformer.blocks[i].attn``): residual add is external, in ``MptBlock.forward``.
- Falcon, non-parallel/non-new-decoder-architecture only, e.g. falcon-rw-7b
  (``transformer.h[i].self_attention``): residual add is external, in
  ``FalconDecoderLayer.forward`` via ``dropout_add(attention_output, residual, ...)`` called *after*
  ``self_attention`` returns.
- Bloom, default (non-tensor-parallel) path only (``transformer.h[i].self_attention.dense``):
  ``BloomAttention.forward`` adds the residual *internally*
  (``dropout_add(self.dense(context_layer), residual, ...)``), so hooking ``self_attention`` itself
  would scale the already-residual-added value -- wrong. But ``self.dense``'s own output is exactly
  the attention branch's final projection, computed *before* that internal residual add, so hooking
  ``self_attention.dense`` instead achieves the same ``x + beta*Attn(x)`` semantics as MPT/Falcon.
  Same mechanism already used and validated for Bloom's Q1/Q2 capture in
  the branch-capture implementation. ``pretraining_tp>1`` with ``slow_but_exact``
  bypasses ``self.dense`` with a manual tensor-parallel sum and is rejected.
- Decoder architectures with ``model.layers[i].self_attn`` whose residual addition is performed
  by the decoder layer after ``self_attn`` returns. This covers the Qwen2, Qwen3, and SmolLM3
  implementations used in the RoPE experiments; the hooked tuple's first element is the
  post-output-projection attention branch.

Falcon's parallel-attention and new-decoder-architecture variants are **not** supported: unlike
Bloom, their attention output is summed with the MLP output *before* the single external residual
add, so there is no submodule whose own output isolates just the attention branch's contribution.
"""

from collections.abc import Callable
from types import TracebackType
from typing import Self

import structlog
import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.falcon.modeling_falcon import FalconForCausalLM
from transformers.models.mpt.modeling_mpt import MptForCausalLM

from src.util.configs import EvalConfig

logger = structlog.get_logger()


def load_causal_model(eval_config: EvalConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a causal LM for the Q3 beta intervention, for any architecture ``BetaScale`` can hook.

    Slimmed-down copy of the standard causal-model loading path (no
    attribution/QK-wrapper branches), generalized beyond that function's MPT-only restriction to
    every architecture ``_attention_modules`` supports.

    Args:
        eval_config: Evaluation config providing ``hf_model_config``.

    Returns:
        The loaded, eval-mode model and its tokenizer.

    """
    model_config = eval_config.hf_model_config
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_id,
        padding_side="left",
        cache_dir=model_config.cache_dir,
        device_map=model_config.device_map or "auto",
        max_memory=model_config.max_memory,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_id,
        cache_dir=model_config.cache_dir,
        device_map=model_config.device_map or "auto",
        max_memory=model_config.max_memory,
        dtype=torch.bfloat16,
    ).eval()
    model.requires_grad_(requires_grad=False)
    model.config.use_cache = False

    # Fail fast on an unsupported architecture rather than silently registering no hooks.
    _attention_modules(model)

    return model, tokenizer


def _architecture_layers(model: PreTrainedModel) -> tuple[list[nn.Module], list[nn.Module]]:
    """
    Return (decoder blocks, attention submodules) for every architecture this module supports.

    Single source of truth for architecture dispatch -- ``_attention_modules`` and
    ``_decoder_blocks`` are both thin wrappers over this, so the two lists can never drift out of
    sync (same length, same layer order, built from the same underlying block list).
    """
    # Every branch below is duck-typed on purpose (real transformers modules and the toy modules in
    # tests/intervention_test.py alike); pyright only sees nn.Module's ambiguous
    # __getattr__ fallback type here, hence the blanket ignores.
    match model:
        case MptForCausalLM():
            blocks = list(model.transformer.blocks)  # pyright: ignore[reportAttributeAccessIssue]
            return blocks, [block.attn for block in blocks]  # pyright: ignore[reportReturnType]
        case FalconForCausalLM():
            if model.config.new_decoder_architecture or model.config.parallel_attn:
                raise NotImplementedError(
                    "This module only supports Falcon's non-parallel, non-new-decoder-architecture "
                    "variant (e.g. falcon-rw-7b), where the residual add happens outside "
                    "self_attention; got new_decoder_architecture="
                    f"{model.config.new_decoder_architecture}, parallel_attn={model.config.parallel_attn}.",
                )
            blocks = list(model.transformer.h)  # pyright: ignore[reportAttributeAccessIssue]
            return blocks, [layer.self_attention for layer in blocks]  # pyright: ignore[reportReturnType]
        case BloomForCausalLM():
            if model.config.pretraining_tp > 1 and model.config.slow_but_exact:
                raise NotImplementedError(
                    "BetaScale only supports Bloom's default dense-projection path; "
                    "pretraining_tp>1 with slow_but_exact bypasses self_attention.dense with a "
                    "manual tensor-parallel sum, so the attention branch could not be scaled there.",
                )
            blocks = list(model.transformer.h)  # pyright: ignore[reportAttributeAccessIssue]
            return blocks, [block.self_attention.dense for block in blocks]  # pyright: ignore[reportReturnType, reportAttributeAccessIssue]
        case _ if hasattr(model, "model") and hasattr(model.model, "layers"):
            blocks = list(model.model.layers)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
            if not blocks or not all(hasattr(block, "self_attn") for block in blocks):
                raise TypeError(
                    f"{type(model)} exposes model.layers but not one self_attn module per layer",
                )
            return blocks, [block.self_attn for block in blocks]  # pyright: ignore[reportGeneralTypeIssues]
        case _ if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
            # Duck-typed fallback for the toy models in tests/intervention_test.py --
            # any object with an MPT-shaped ``.transformer.blocks[i].attn`` works, real model or not.
            blocks = list(model.transformer.blocks)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
            return blocks, [block.attn for block in blocks]  # pyright: ignore[reportGeneralTypeIssues]
        case _:
            raise TypeError(
                f"This module does not support {type(model)}; see this module's docstring for why "
                "Bloom and Falcon's parallel/new-decoder variants are excluded.",
            )


def _attention_modules(model: PreTrainedModel) -> list[nn.Module]:
    """Return each layer's attention submodule, in the residual-external module path BetaScale requires."""
    return _architecture_layers(model)[1]


def _decoder_blocks(model: PreTrainedModel) -> list[nn.Module]:
    """Return each layer's whole decoder-block module, in the same order as ``_attention_modules``."""
    return _architecture_layers(model)[0]


class BetaScale:
    """Scale every supported model's attention branch output by ``beta``, before the residual add."""

    def __init__(self, model: PreTrainedModel, beta: float) -> None:
        """
        Prepare (but do not yet register) a beta-scaling intervention for ``model``.

        Args:
            model: Causal LM whose attention submodules (see module docstring for the supported
                architectures) will be hooked.
            beta: Scale factor applied to each attention module's output tensor.

        """
        self.model = model
        self.beta = beta
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(
        self,
    ) -> Callable[[nn.Module, object, torch.Tensor | tuple[torch.Tensor, ...]], torch.Tensor | tuple[torch.Tensor, ...]]:
        beta = self.beta

        def hook(
            _module: nn.Module,
            _input_data: object,
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> torch.Tensor | tuple[torch.Tensor, ...]:
            if isinstance(output, tuple):
                return (beta * output[0], *output[1:])
            return beta * output

        return hook

    def __enter__(self) -> Self:
        """Register the beta-scaling forward hook on every attention module."""
        for attn in _attention_modules(self.model):
            self._handles.append(attn.register_forward_hook(self._make_hook()))  # pyright: ignore[reportAttributeAccessIssue]
        logger.info(f"BetaScale entered: beta={self.beta}, {len(self._handles)} hooks registered")
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Remove every hook registered by ``__enter__``."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
