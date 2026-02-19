"""MPT attention wrapper utilities."""

from typing import override

import einops
import structlog
import torch
from torch import nn
from transformers import Cache
from transformers.models.mpt.configuration_mpt import MptConfig
from transformers.models.mpt.modeling_mpt import MptAttention
from transformers.utils.deprecation import deprecate_kwarg

logger = structlog.get_logger()


class MptAttentionWrapper(MptAttention):  # type: ignore[misc]
    """MptAttentionWrapper with attention heatmap tracking."""

    def __init__(
        self,
        config: MptConfig,
        sequence_length: int,
        layer_idx: int | None = None,
    ) -> None:
        """
        Initialize the MptAttentionWrapper.

        Args:
            config: Mpt model configuration.
            sequence_length: Sequence length used to allocate buffers.
            layer_idx: Optional layer index passed to parent.

        """
        super().__init__(config, layer_idx)

        self.qk_sum = torch.zeros((self.n_heads, sequence_length, sequence_length))
        self.qk_sum_sq = torch.zeros((self.n_heads, sequence_length, sequence_length))
        self.qk_count = 0

        self.attention_probs = torch.zeros(
            (self.n_heads, sequence_length, sequence_length),
        )
        self.attention_probs_sq = torch.zeros(
            (self.n_heads, sequence_length, sequence_length),
        )
        self.attention_probs_count = 0

    @override
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")  # type: ignore[misc]
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_bias: torch.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = hidden_states.shape[:2]

        mixed_qkv = self.Wqkv(hidden_states)
        if self.clip_qkv:
            mixed_qkv = mixed_qkv.clamp(min=-self.clip_qkv, max=self.clip_qkv)

        query_states, key_states, value_states = mixed_qkv.chunk(3, dim=2)
        query_states = query_states.reshape(
            batch_size,
            seq_length,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)
        key_states = key_states.reshape(
            batch_size,
            seq_length,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)
        value_states = value_states.reshape(
            batch_size,
            seq_length,
            self.n_heads,
            self.head_dim,
        ).transpose(1, 2)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,  # pyright: ignore[reportArgumentType]
                cache_kwargs,
            )

        attention_scores = (  # pyright: ignore[reportOperatorIssue]
            torch.matmul(query_states, key_states.transpose(-1, -2))
            * self.softmax_scale
        )
        query_length = (
            seq_length
            if past_key_values is None
            else seq_length + past_key_values.get_seq_length()
        )

        qk_prod = torch.matmul(
            query_states.detach(),
            key_states.detach().transpose(-1, -2),
        ).cpu()
        qk_prod = einops.reduce(qk_prod, "b h q k -> h q k", "mean")
        self.qk_sum += qk_prod
        self.qk_sum_sq += qk_prod**2
        self.qk_count += 1

        if position_bias is not None:
            if len(position_bias.shape) != 3:  # noqa: PLR2004
                raise ValueError(
                    f"Expecting position_bias shape to be 3 dimensions, got {len(position_bias.shape)}",
                )
            key_length = key_states.shape[-2]

            position_bias_query_index = max(0, position_bias.size(1) - query_length)
            position_bias_key_index = max(0, position_bias.size(2) - key_length)

            position_bias = position_bias[
                :,
                position_bias_query_index:,
                position_bias_key_index:,
            ]

            attention_scores = attention_scores + position_bias

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask,
                torch.finfo(query_states.dtype).min,
            )

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(attention_scores.float(), dim=-1).to(
            value_states.dtype,
        )

        cpu_attention_probs = einops.reduce(
            attn_weights.detach().cpu(),
            "b h q k -> h q k",
            "mean",
        )
        self.attention_probs += cpu_attention_probs
        self.attention_probs_sq += cpu_attention_probs**2
        self.attention_probs_count += 1

        attn_weights = nn.functional.dropout(
            attn_weights,
            p=self.attn_dropout_p,
            training=self.training,
        )

        context_states = torch.matmul(attn_weights, value_states)
        context_states = (
            context_states.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_length, -1)
        )
        attn_output = self.out_proj(context_states)

        return attn_output, attn_weights
