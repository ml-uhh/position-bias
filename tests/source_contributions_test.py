"""Tests for forward-only attention source contributions."""

import torch
from torch import nn

from src.models.source_contributions import (
    capture_source_contribution_state,
    clear_source_contribution_state,
    initialize_source_contribution_state,
    value_source_cancellation_sums,
    value_source_removal_scores,
)


class _AttentionStub:
    def __init__(self) -> None:
        self.o_proj = nn.Linear(4, 4, bias=False)
        self.capture_source_projection_state = False
        self.last_source_attention_probs: torch.Tensor | None = None
        self.last_source_value_states: torch.Tensor | None = None


def test_removal_scores_match_brute_force_source_vectors() -> None:
    """The vectorized implementation matches explicit source construction."""
    torch.manual_seed(3)
    module = _AttentionStub()
    initialize_source_contribution_state(module)
    module.capture_source_projection_state = True

    probabilities = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0], [0.4, 0.6, 0.0], [0.2, 0.3, 0.5]],
                [[1.0, 0.0, 0.0], [0.1, 0.9, 0.0], [0.5, 0.1, 0.4]],
            ],
        ],
    )
    values = torch.randn(1, 2, 3, 2)
    residual = torch.randn(1, 3, 4)
    capture_source_contribution_state(module, probabilities, values)

    source_vectors = []
    for query_index in range(3):
        query_sources = []
        for key_index in range(3):
            head_values = (
                probabilities[0, :, query_index, key_index, None]
                * values[
                    0,
                    :,
                    key_index,
                ]
            )
            query_sources.append(module.o_proj(head_values.reshape(-1)))
        source_vectors.append(torch.stack(query_sources))
    projected = torch.stack(source_vectors).unsqueeze(0)
    target = projected.sum(dim=2) + residual

    actual = value_source_removal_scores(
        module,
        [(residual, target)],
        query_chunk_size=2,
    )[0]
    expected_contributions = projected.clone()
    diagonal = torch.arange(3)
    expected_contributions[:, diagonal, diagonal] += residual[:, diagonal]
    target_norm = torch.linalg.vector_norm(target, ord=1, dim=-1, keepdim=True)
    expected = (
        target_norm
        - torch.linalg.vector_norm(
            target.unsqueeze(2) - expected_contributions,
            ord=1,
            dim=-1,
        )
    ).clamp_min(0.0)

    torch.testing.assert_close(actual, expected)
    clear_source_contribution_state(module)
    assert module.last_source_attention_probs is None
    assert module.last_source_value_states is None


def test_cancellation_sums_match_brute_force_source_vectors() -> None:
    """Cancellation masses agree with an explicit head/source decomposition."""
    torch.manual_seed(7)
    module = _AttentionStub()
    initialize_source_contribution_state(module)
    module.capture_source_projection_state = True
    probabilities = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.4, 0.6]],
                [[1.0, 0.0], [0.2, 0.8]],
            ],
        ],
    )
    values = torch.randn(1, 2, 2, 2)
    capture_source_contribution_state(module, probabilities, values)

    weight = module.o_proj.weight.reshape(4, 2, 2)
    projected_values = torch.einsum("bhkd,ohd->bhko", values, weight)
    head_sources = probabilities[..., None] * projected_values[:, :, None]
    source_vectors = head_sources.sum(dim=1)
    expected = {
        "raw_mass": probabilities.sum(dim=(1, 2, 3)).double(),
        "raw_token0_mass": probabilities[:, :, :, 0].sum(dim=(1, 2)).double(),
        "head_pre_l1": torch.linalg.vector_norm(
            head_sources,
            ord=1,
            dim=-1,
        ).sum(dim=(1, 2, 3)).double(),
        "head_pre_token0_l1": torch.linalg.vector_norm(
            head_sources[:, :, :, 0],
            ord=1,
            dim=-1,
        ).sum(dim=(1, 2)).double(),
        "head_post_l1": torch.linalg.vector_norm(
            source_vectors,
            ord=1,
            dim=-1,
        ).sum(dim=(1, 2)).double(),
        "head_post_token0_l1": torch.linalg.vector_norm(
            source_vectors[:, :, 0],
            ord=1,
            dim=-1,
        ).sum(dim=1).double(),
        "source_post_l1": torch.linalg.vector_norm(
            source_vectors.sum(dim=2),
            ord=1,
            dim=-1,
        ).sum(dim=1).double(),
    }

    actual = value_source_cancellation_sums(module, query_chunk_size=1)
    for name, expected_value in expected.items():
        torch.testing.assert_close(actual[name], expected_value)
