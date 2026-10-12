"""Qwen2 attention wrapper utilities."""

from typing import TYPE_CHECKING, override

import einops
import structlog
import torch
from transformers import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
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


class Qwen2AttentionWrapper(Qwen2Attention):  # type: ignore[misc]
    """Wrapper around Qwen2Attention to store statistics."""

    def __init__(
        self,
        config: Qwen2Config,
        sequence_length: int,
        layer_idx: int,
    ) -> None:
        """
        Initialize the Qwen2AttentionWrapper.

        Args:
            config: Qwen2 model configuration.
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
        self.attention_probs_sum = torch.zeros(
            (self.config.num_attention_heads, sequence_length, sequence_length),
        )
        self.attention_probs_count = 0
        self.last_attention_probs: torch.Tensor | None = None
        self.capture_value_contributions = False
        self.last_value_output_contributions: torch.Tensor | None = None
        self.capture_value_norm_kernels = False
        self.last_value_norm_attention: torch.Tensor | None = None
        self.last_value_o_norm_attention: torch.Tensor | None = None
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
    def forward(  # noqa: PLR0915
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass with attention statistics collection.

        Args:
            hidden_states: Input hidden states.
            position_embeddings: Position embeddings (cos, sin).
            attention_mask: Attention mask.
            past_key_values: Cached key-values for generation.
            cache_position: Cache position for static cache.
            **kwargs: Additional keyword arguments for flash attention.

        Returns:
            Attention output and attention weights.

        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        key_states_full = repeat_kv(key_states, self.num_key_value_groups)
        qk_prod_batch = torch.matmul(
            query_states.detach().cpu(),
            key_states_full.detach().cpu().transpose(2, 3),
        )
        qk_prod = einops.reduce(qk_prod_batch, "b h q k -> h q k", "mean")
        self.qk_sum += qk_prod
        self.qk_sum_sq += qk_prod**2
        self.qk_count += 1

        # Rollout is defined in terms of the expected attention kernel
        # E[softmax(L)], not softmax(E[L]). Accumulate the former directly,
        # including Qwen2's logit scaling and the exact additive mask supplied
        # to this layer. Keeping only the first moment avoids another large
        # (layers, heads, sequence, sequence) variance buffer.
        attention_logits = qk_prod_batch.to(torch.float32) * self.scaling
        if attention_mask is not None:
            mask = attention_mask.detach().to(device="cpu", dtype=torch.float32)
            attention_logits = (
                attention_logits
                + mask[
                    :,
                    :,
                    : attention_logits.shape[-2],
                    : attention_logits.shape[-1],
                ]
            )
        else:
            # Match the SDPA fast path. Transformers omits the materialized
            # additive mask there and communicates causality via ``is_causal``
            # (falling back to the attention module's decoder flag).
            is_causal = kwargs.get("is_causal")
            if is_causal is None:
                is_causal = getattr(self, "is_causal", True)
            query_length = attention_logits.shape[-2]
            key_length = attention_logits.shape[-1]
            if query_length > 1 and is_causal:
                query_positions = torch.arange(query_length).unsqueeze(1)
                key_positions = torch.arange(key_length).unsqueeze(0)
                causal_offset = key_length - query_length
                causal_mask = key_positions > query_positions + causal_offset
                attention_logits.masked_fill_(
                    causal_mask.unsqueeze(0).unsqueeze(0),
                    -torch.inf,
                )
        attention_probs = torch.softmax(attention_logits, dim=-1)
        # Keep the head-averaged kernels from the current batch so callers can
        # form a complete rollout for each sample before averaging over the
        # corpus. This is deliberately not accumulated across batches.
        self.last_attention_probs = attention_probs.mean(dim=1)
        if self.capture_value_contributions:
            # Decompose the last query's projected attention output into one
            # vector per source token. Summing these vectors reproduces the
            # last-query attention output after the O projection (Qwen's
            # projection has no bias). Their norms provide a V/O-aware local
            # transport proxy without retaining an autograd graph.
            value_states_full = repeat_kv(value_states, self.num_key_value_groups)
            last_query_probs = attention_probs[:, :, -1, :].to(
                device=value_states_full.device,
                dtype=value_states_full.dtype,
            )
            source_head_outputs = value_states_full * last_query_probs.unsqueeze(-1)
            source_outputs = source_head_outputs.transpose(1, 2).reshape(
                value_states_full.shape[0],
                value_states_full.shape[2],
                -1,
            )
            projected_sources = F.linear(
                source_outputs,
                self.o_proj.weight,
                bias=None,
            )
            self.last_value_output_contributions = (
                projected_sources.detach().cpu().float()
            )
        if self.capture_value_norm_kernels:
            value_states_full = repeat_kv(value_states, self.num_key_value_groups)
            value_norms = torch.linalg.vector_norm(
                value_states_full.detach().cpu().float(),
                dim=-1,
            )
            value_weighted = attention_probs * value_norms.unsqueeze(-2)
            value_kernel = value_weighted.mean(dim=1)
            self.last_value_norm_attention = value_kernel / value_kernel.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-30)

            output_weight = self.o_proj.weight.detach().float()
            output_weight = output_weight.view(
                output_weight.shape[0],
                self.config.num_attention_heads,
                self.head_dim,
            )
            head_output_norms = torch.linalg.vector_norm(
                output_weight,
                dim=(0, 2),
            ).cpu()
            value_o_weighted = value_weighted * head_output_norms[None, :, None, None]
            value_o_kernel = value_o_weighted.mean(dim=1)
            self.last_value_o_norm_attention = value_o_kernel / value_o_kernel.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-30)
        attention_probs_mean = einops.reduce(
            attention_probs,
            "b h q k -> h q k",
            "mean",
        )
        self.attention_probs_sum += attention_probs_mean
        self.attention_probs_count += 1

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
        if self.capture_source_projection_state:
            source_attention_probs = (
                attn_weights
                if attn_weights is not None
                else attention_probs.to(device=value_states.device)
            )
            capture_source_contribution_state(
                self,
                source_attention_probs,
                repeat_kv(value_states, self.num_key_value_groups),
            )
        return attn_output, attn_weights
