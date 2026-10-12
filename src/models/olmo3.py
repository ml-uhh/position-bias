"""Olmo3 attention wrapper utilities."""

from typing import TYPE_CHECKING, override

import einops
import structlog
import torch
from transformers import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.olmo3.modeling_olmo3 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)
from transformers.models.olmo3.configuration_olmo3 import Olmo3Config
from transformers.models.olmo3.modeling_olmo3 import Olmo3Attention
from transformers.processing_utils import Unpack
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import TransformersKwargs

if TYPE_CHECKING:
    from collections.abc import Callable

logger = structlog.get_logger()


class Olmo3AttentionWrapper(Olmo3Attention):  # type: ignore[misc]
    """Olmo3Attention wrapper with attention heatmap tracking."""

    def __init__(
        self,
        config: Olmo3Config,
        sequence_length: int,
        layer_idx: int,
    ) -> None:
        """
        Initialize the Olmo3AttentionWrapper.

        Args:
            config: Olmo3 model configuration.
            sequence_length: Sequence length used to allocate buffers.
            layer_idx: Layer index passed to parent.

        """
        super().__init__(config, layer_idx)

        self.qk_sum = torch.zeros(
            (self.config.num_attention_heads, sequence_length, sequence_length),
        )
        self.qk_sum_sq = torch.zeros(
            (self.config.num_attention_heads, sequence_length, sequence_length),
        )
        self.qk_count = 0

    @override
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")  # type: ignore[misc]
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

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
            query_states.detach(),
            key_states_full.detach().transpose(2, 3),
        ).cpu()
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
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
