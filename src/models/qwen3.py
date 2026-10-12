"""Qwen3 attention wrapper utilities."""

from typing import TYPE_CHECKING, override

import einops
import structlog
import torch
from transformers import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)
from transformers.processing_utils import Unpack
from transformers.utils.deprecation import deprecate_kwarg

from src.models.source_contributions import (
    capture_source_contribution_state,
    clear_source_contribution_state,
    initialize_source_contribution_state,
    value_source_removal_scores,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = structlog.get_logger()


class Qwen3AttentionWrapper(Qwen3Attention):  # type: ignore[misc]
    """Wrapper around Qwen3Attention to store statistics."""

    def __init__(
        self,
        config: Qwen3Config,
        sequence_length: int,
        layer_idx: int,
    ) -> None:
        """
        Initialize the Qwen3AttentionWrapper.

        Args:
            config: Qwen3 model configuration.
            sequence_length: Maximum sequence length for which buffers are allocated.
            layer_idx: Layer index in the model.

        """
        super().__init__(config, layer_idx)

        self.qk_sum = torch.zeros(
            (self.config.num_attention_heads, sequence_length, sequence_length),
        )
        self.qk_sum_sq = torch.zeros(
            (self.config.num_attention_heads, sequence_length, sequence_length),
        )
        self.qk_count = 0
        self.last_attention_probs: torch.Tensor | None = None
        initialize_source_contribution_state(self)

    def clear_source_projection_state(self) -> None:
        """Release current-batch tensors retained for a forward hook."""
        clear_source_contribution_state(self)

    def value_source_removal_scores(
        self,
        decompositions: list[tuple[torch.Tensor | None, torch.Tensor]],
        query_chunk_size: int = 16,
    ) -> list[torch.Tensor]:
        """Return forward-only ALTI scores for exact source vectors."""
        return value_source_removal_scores(
            self,
            decompositions,
            query_chunk_size=query_chunk_size,
        )

    @override
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")  # type: ignore[misc]
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape),
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape),
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
            )

        key_states_full = repeat_kv(key_states, self.num_key_value_groups)
        qk_prod = torch.matmul(
            query_states.detach().cpu(),
            key_states_full.detach().cpu().transpose(2, 3),
        )
        qk_prod = einops.reduce(qk_prod, "b h q k -> h q k", "mean")
        self.qk_sum += qk_prod
        self.qk_sum_sq += qk_prod**2
        self.qk_count += 1

        attention_interface: Callable = eager_attention_forward  # type: ignore[type-arg]
        if self.config._attn_implementation != "eager":  # noqa: SLF001
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation  # noqa: SLF001
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        if self.capture_source_projection_state:
            if attn_weights is None:
                raise RuntimeError("Eager attention did not return attention weights")
            self.last_attention_probs = attn_weights.mean(dim=1).detach()
            capture_source_contribution_state(
                self,
                attn_weights,
                repeat_kv(value_states, self.num_key_value_groups),
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
