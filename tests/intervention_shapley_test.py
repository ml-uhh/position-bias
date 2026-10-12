"""Tests for the norm-Shapley beta-intervention prediction."""
# ruff: noqa: INP001

import pytest
import torch
from transformers import BloomConfig, BloomForCausalLM

from src.intervention.shapley_baseline import (
    capture_branch_statistics,
    mean_shapley_schedule,
    shapley_lambdas_from_statistics,
)


_TINY_BLOOM_CONFIG = BloomConfig(
    vocab_size=32,
    hidden_size=16,
    n_layer=3,
    n_head=2,
    pretraining_tp=1,
    slow_but_exact=False,
)


def test_capture_branch_statistics_on_bloom() -> None:
    model = BloomForCausalLM(_TINY_BLOOM_CONFIG).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    statistics = capture_branch_statistics(model, input_ids, attention_mask)

    assert statistics.shape == (3, 3)
    assert torch.isfinite(statistics).all()
    assert torch.all(statistics[:, :2] >= 0)


def test_shapley_beta_formula_for_orthogonal_branches() -> None:
    statistics = torch.tensor([[[1.0, 1.0, 0.0]]], dtype=torch.float64)

    at_one = shapley_lambdas_from_statistics(statistics, 1.0).item()
    below = shapley_lambdas_from_statistics(statistics, 0.5).item()
    above = shapley_lambdas_from_statistics(statistics, 2.0).item()

    assert at_one == pytest.approx(0.5)
    assert below == pytest.approx(0.27639320225)
    assert above == pytest.approx(0.72360679775)


def test_shapley_beta_formula_matches_definition() -> None:
    residual = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    attention = torch.tensor([-0.25, 0.75, 1.5], dtype=torch.float64)
    statistics = torch.tensor(
        [[[
            torch.sum(residual * residual),
            torch.sum(attention * attention),
            torch.sum(residual * attention),
        ]]],
        dtype=torch.float64,
    )

    for beta in (0.6, 0.8, 1.0, 1.2, 1.4):
        summed = residual + beta * attention
        expected = (summed.norm() + beta * attention.norm() - residual.norm()) / (2.0 * summed.norm())
        actual = shapley_lambdas_from_statistics(statistics, beta).item()
        assert actual == pytest.approx(expected.item())


def test_shapley_schedule_is_monotone_and_averages_over_prompts() -> None:
    statistics = torch.tensor(
        [
            [[4.0, 1.0, 0.5], [1.0, 9.0, -2.0]],
            [[2.0, 3.0, -1.0], [5.0, 2.0, 1.5]],
        ],
        dtype=torch.float64,
    )
    low = torch.tensor(mean_shapley_schedule(statistics, 0.8))
    reference = torch.tensor(mean_shapley_schedule(statistics, 1.0))
    high = torch.tensor(mean_shapley_schedule(statistics, 1.2))

    assert torch.all(low <= reference)
    assert torch.all(reference <= high)
    assert torch.all((low >= 0.0) & (high <= 1.0))


def test_shapley_beta_rejects_nonpositive_scale() -> None:
    statistics = torch.ones((1, 1, 3), dtype=torch.float64)
    with pytest.raises(ValueError, match="positive"):
        shapley_lambdas_from_statistics(statistics, 0.0)
