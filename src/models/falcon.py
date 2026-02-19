"""Falcon attention wrapper utilities."""

from typing import override

import einops
import structlog
import torch
import torch.nn.functional as F  # noqa: N812
from transformers import Cache
from transformers.models.falcon.configuration_falcon import FalconConfig
from transformers.models.falcon.modeling_falcon import (
    FalconAttention,
    apply_rotary_pos_emb,
)

logger = structlog.get_logger()


class FalconAttentionWrapper(FalconAttention):  # type: ignore[misc]
    """Wrapper around FalconAttention to store statistics."""

    def __init__(
        self,
        config: FalconConfig,
        sequence_length: int,
        layer_idx: int | None = None,
    ) -> None:
        """
        Initialize the FalconAttentionWrapper.

        Args:
            config: Falcon model configuration.
            sequence_length: Maximum sequence length for which buffers are allocated.
            layer_idx: Optional layer index passed to the parent constructor.

        """
        super().__init__(config, layer_idx)

        self.qk_sum = torch.zeros((self.num_heads, sequence_length, sequence_length))
        self.qk_sum_sq = torch.zeros((self.num_heads, sequence_length, sequence_length))
        self.qk_count = 0

        self.attention_probs = torch.zeros(
            (self.num_heads, sequence_length, sequence_length),
        )
        self.attention_probs_sq = torch.zeros(
            (self.num_heads, sequence_length, sequence_length),
        )
        self.attention_probs_count = 0

    @override
    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor | None,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        layer_past: Cache | None = None,
        head_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]
        | None = None,  # necessary, but kept here for BC
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        fused_qkv = self.query_key_value(
            hidden_states,
        )  # [batch_size, seq_length, 3 x hidden_size]
        num_kv_heads = (
            self.num_heads if self.new_decoder_architecture else self.num_kv_heads
        )
        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        batch_size, query_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(
            batch_size,
            self.num_heads,
            query_length,
            self.head_dim,
        )
        key_layer = key_layer.transpose(1, 2).reshape(
            batch_size,
            num_kv_heads,
            query_length,
            self.head_dim,
        )
        value_layer = value_layer.transpose(1, 2).reshape(
            batch_size,
            num_kv_heads,
            query_length,
            self.head_dim,
        )

        if alibi is None:
            cos, sin = position_embeddings  # type: ignore[misc] # pyright: ignore[reportGeneralTypeIssues]
            query_layer, key_layer = apply_rotary_pos_emb(
                query_layer,
                key_layer,
                cos,
                sin,
            )

        if layer_past is not None:
            cache_kwargs = {"cache_position": cache_position}
            if alibi is None:
                cache_kwargs.update({"sin": sin, "cos": cos})  # pyright: ignore[reportCallIssue, reportArgumentType, reportPossiblyUnboundVariable]
            key_layer, value_layer = layer_past.update(
                key_layer,
                value_layer,
                self.layer_idx,  # pyright: ignore[reportArgumentType]
                cache_kwargs,
            )

        kv_length = key_layer.shape[-2]
        if (
            self.config._attn_implementation == "sdpa"  # noqa: SLF001
            and query_layer.device.type == "cuda"
            and attention_mask is not None
        ):
            # For torch<=2.1.2, SDPA with memory-efficient backend is bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, : key_layer.shape[-2]]

        if alibi is None:
            raise ValueError("FalconAttentionWrapper requires alibi tensor")

        matmul_result = query_layer @ key_layer.transpose(-1, -2)

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(
            batch_size,
            self.num_heads,
            query_length,
            kv_length,
        )

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16 or input_dtype == torch.bfloat16:  # noqa: PLR1714
            attention_scores = attention_scores.to(torch.float32)

        qk_prod = attention_scores.detach().cpu()
        qk_prod = einops.reduce(qk_prod, "b h q k -> h q k", "mean")
        self.qk_sum += qk_prod
        self.qk_sum_sq += qk_prod**2
        self.qk_count += 1

        attention_logits = attention_scores + alibi.view(
            batch_size,
            self.num_heads,
            1,
            -1,
        )
        attention_logits *= self.inv_norm_factor
        attention_probs = F.softmax(
            attention_logits + attention_mask,
            dim=-1,
            dtype=hidden_states.dtype,
        )

        cpu_attention_probs = einops.reduce(
            attention_probs.detach().cpu(),
            "b h q k -> h q k",
            "mean",
        )
        self.attention_probs += cpu_attention_probs
        self.attention_probs_sq += cpu_attention_probs**2
        self.attention_probs_count += 1

        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size, num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(
            batch_size,
            self.num_heads,
            query_length,
            kv_length,
        )

        # matmul: [batch_size * num_heads, q_length, head_dim]
        attn_output = (attention_probs_reshaped @ value_layer).flatten(0, 1)

        # change view [batch_size, q_length, num_heads * head_dim]
        attn_output = self._merge_heads(attn_output)

        attn_output = self.dense(attn_output)

        return attn_output, attention_probs
