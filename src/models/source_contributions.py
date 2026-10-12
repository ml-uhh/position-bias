"""Forward-only source-vector attribution utilities for attention wrappers."""

from __future__ import annotations

from typing import Protocol

import torch
from torch import nn


class SourceContributionAttention(Protocol):
    """Structural interface required by the source-contribution routines."""

    o_proj: nn.Linear
    capture_source_projection_state: bool
    last_source_attention_probs: torch.Tensor | None
    last_source_value_states: torch.Tensor | None


def _project_values_by_head(
    module: SourceContributionAttention,
    value_states: torch.Tensor,
) -> torch.Tensor:
    """Project each head/value pair through its block of the output map."""
    if module.o_proj.bias is not None:
        raise NotImplementedError(
            "Source-vector decomposition currently requires a bias-free output projection",
        )
    head_count = value_states.shape[1]
    head_dimension = value_states.shape[-1]
    output_dimension, input_dimension = module.o_proj.weight.shape
    if input_dimension != head_count * head_dimension:
        raise ValueError("Output projection shape does not match the captured heads")
    head_projection_weights = module.o_proj.weight.reshape(
        output_dimension,
        head_count,
        head_dimension,
    )
    return torch.einsum(
        "bhkd,ohd->bhko",
        value_states,
        head_projection_weights,
    )


def initialize_source_contribution_state(
    module: SourceContributionAttention,
) -> None:
    """Initialize the transient tensors used by forward contribution hooks."""
    module.capture_source_projection_state = False
    module.last_source_attention_probs = None
    module.last_source_value_states = None


def capture_source_contribution_state(
    module: SourceContributionAttention,
    attention_probs: torch.Tensor,
    expanded_value_states: torch.Tensor,
) -> None:
    """Retain exact attention probabilities and repeated value vectors."""
    if not module.capture_source_projection_state:
        return
    module.last_source_attention_probs = attention_probs.to(
        device=expanded_value_states.device,
        dtype=expanded_value_states.dtype,
    ).detach()
    module.last_source_value_states = expanded_value_states.detach()


def clear_source_contribution_state(module: SourceContributionAttention) -> None:
    """Release tensors retained for the current forward-hook invocation."""
    module.last_source_attention_probs = None
    module.last_source_value_states = None


def value_source_removal_scores(
    module: SourceContributionAttention,
    decompositions: list[tuple[torch.Tensor | None, torch.Tensor]],
    query_chunk_size: int = 16,
) -> list[torch.Tensor]:
    """
    Return ALTI scores for exact per-source ``A V W_O`` vectors.

    Each decomposition contains an optional vector added to source ``j=i``
    and the corresponding local output vector. The score is the positive
    reduction in the output's L1 norm when that source contribution is
    removed. This retains direction and cancellation without gradients.
    """
    attention_probs = module.last_source_attention_probs
    value_states = module.last_source_value_states
    if attention_probs is None or value_states is None:
        raise RuntimeError("Source-projection state was not captured")
    batch_size, _, query_count, key_count = attention_probs.shape
    # By linearity, project each head/value pair once instead of applying the
    # full output matrix separately to every query-source pair.
    projected_values = _project_values_by_head(module, value_states)
    results = [
        torch.empty(
            batch_size,
            query_count,
            key_count,
            device=value_states.device,
            dtype=torch.float32,
        )
        for _ in decompositions
    ]
    for query_start in range(0, query_count, query_chunk_size):
        query_stop = min(query_start + query_chunk_size, query_count)
        probabilities = attention_probs[:, :, query_start:query_stop, :]
        projected_sources = torch.einsum(
            "bhik,bhko->biko",
            probabilities,
            projected_values,
        )
        local_queries = torch.arange(
            query_stop - query_start,
            device=value_states.device,
        )
        global_queries = torch.arange(
            query_start,
            query_stop,
            device=value_states.device,
        )
        for result, (diagonal_addition, target) in zip(
            results,
            decompositions,
            strict=True,
        ):
            if diagonal_addition is None:
                contribution = projected_sources
            else:
                contribution = projected_sources.clone()
                contribution[:, local_queries, global_queries] += diagonal_addition[
                    :,
                    global_queries,
                ].to(dtype=contribution.dtype)
            target_chunk = target[:, query_start:query_stop].float()
            target_norm = torch.linalg.vector_norm(
                target_chunk,
                ord=1,
                dim=-1,
                keepdim=True,
            )
            removed_norm = torch.linalg.vector_norm(
                target_chunk.unsqueeze(2) - contribution.float(),
                ord=1,
                dim=-1,
            )
            result[:, query_start:query_stop] = (target_norm - removed_norm).clamp_min(
                0.0,
            )
    return results


def value_source_cancellation_sums(
    module: SourceContributionAttention,
    query_chunk_size: int = 16,
) -> dict[str, torch.Tensor]:
    """Return forward-only masses used to quantify head and source cancellation.

    Values are returned per batch item. Ratios should be formed only after
    summing these masses over the corpus, so negligible token pairs do not
    receive disproportionate weight.
    """
    attention_probs = module.last_source_attention_probs
    value_states = module.last_source_value_states
    if attention_probs is None or value_states is None:
        raise RuntimeError("Source-projection state was not captured")
    batch_size, _, query_count, _ = attention_probs.shape
    projected_values = _project_values_by_head(module, value_states)
    projected_value_norms = torch.linalg.vector_norm(
        projected_values.float(),
        ord=1,
        dim=-1,
    )
    names = (
        "raw_mass",
        "raw_token0_mass",
        "head_pre_l1",
        "head_pre_token0_l1",
        "head_post_l1",
        "head_post_token0_l1",
        "source_post_l1",
    )
    sums = {
        name: torch.zeros(batch_size, device=value_states.device, dtype=torch.float64)
        for name in names
    }
    for query_start in range(0, query_count, query_chunk_size):
        query_stop = min(query_start + query_chunk_size, query_count)
        probabilities = attention_probs[:, :, query_start:query_stop, :].float()
        projected_sources = torch.einsum(
            "bhik,bhko->biko",
            probabilities,
            projected_values.float(),
        )
        head_pre = torch.einsum(
            "bhik,bhk->bik",
            probabilities,
            projected_value_norms,
        )
        head_post = torch.linalg.vector_norm(
            projected_sources,
            ord=1,
            dim=-1,
        )
        sums["raw_mass"] += probabilities.sum(dim=(1, 2, 3)).double()
        sums["raw_token0_mass"] += probabilities[:, :, :, 0].sum(
            dim=(1, 2),
        ).double()
        sums["head_pre_l1"] += head_pre.sum(dim=(1, 2)).double()
        sums["head_pre_token0_l1"] += torch.einsum(
            "bhi,bh->bi",
            probabilities[:, :, :, 0],
            projected_value_norms[:, :, 0],
        ).sum(dim=1).double()
        sums["head_post_l1"] += head_post.sum(dim=(1, 2)).double()
        sums["head_post_token0_l1"] += head_post[:, :, 0].sum(dim=1).double()
        sums["source_post_l1"] += torch.linalg.vector_norm(
            projected_sources.sum(dim=2),
            ord=1,
            dim=-1,
        ).sum(dim=1).double()
    return sums
