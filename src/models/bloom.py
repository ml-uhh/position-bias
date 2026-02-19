"""Bloom attention wrapper utilities."""

from typing import override

import einops
import structlog
import torch
import torch.nn.functional as F  # noqa: N812
from transformers import Cache
from transformers.models.bloom.configuration_bloom import BloomConfig
from transformers.models.bloom.modeling_bloom import BloomAttention, dropout_add

logger = structlog.get_logger()


class BloomAttentionWrapper(BloomAttention):  # type: ignore[misc]
    """Wrapper around BloomAttention to store statistics."""

    def __init__(
        self,
        config: BloomConfig,
        sequence_length: int,
        layer_idx: int | None = None,
    ) -> None:
        """
        Initialize the BloomAttentionWrapper.

        Args:
            config: Bloom model configuration.
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
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Cache | None = None,
        head_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, q_length, _ = hidden_states.shape
        fused_qkv = self.query_key_value(
            hidden_states,
        )  # [batch_size, seq_length, 3 x hidden_size]
        # 3 x [batch_size, num_heads, seq_length, head_dim]
        query_layer, key_layer, value_layer = self._reshape(fused_qkv)

        if layer_past is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_layer, value_layer = layer_past.update(
                key_layer,
                value_layer,
                self.layer_idx,  # pyright: ignore[reportArgumentType]
                cache_kwargs,
            )  # pyright: ignore[reportArgumentType]

        # reshape qkv for further computations
        query_layer = query_layer.reshape(
            batch_size * self.num_heads,
            -1,
            self.head_dim,
        )
        key_layer = key_layer.reshape(
            batch_size * self.num_heads,
            -1,
            self.head_dim,
        ).transpose(-1, -2)
        value_layer = value_layer.reshape(
            batch_size * self.num_heads,
            -1,
            self.head_dim,
        )

        qk_prod = torch.bmm(query_layer.detach().cpu(), key_layer.detach().cpu())
        qk_prod = einops.rearrange(
            qk_prod,
            "(b h) q k -> b h q k",
            b=batch_size,
            h=self.num_heads,
        )
        qk_prod = einops.reduce(qk_prod, "b h q k -> h q k", "mean")
        self.qk_sum += qk_prod
        self.qk_sum_sq += qk_prod**2
        self.qk_count += 1

        # [batch_size * num_heads, q_length, kv_length]
        attention_scores = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        # change view to [batch_size, num_heads, q_length, kv_length]
        attn_weights = attention_scores.view(batch_size, self.num_heads, q_length, -1)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_layer.shape[-1]]
            attn_weights = attn_weights + causal_mask

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype
        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_layer.dtype,
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

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(
            batch_size * self.num_heads,
            q_length,
            -1,
        )

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        # change view [batch_size, q_length, num_heads * head_dim]
        context_layer = self._merge_heads(context_layer)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(
            output_tensor,
            residual,
            self.hidden_dropout,
            self.training,
        )
        return output_tensor, attention_probs
