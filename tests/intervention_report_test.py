"""Sanity checks for the attention-scaling intervention report."""
# ruff: noqa: INP001 (test module, intentionally not a package)

import numpy as np
import pytest

from src.intervention.beta_report import (
    PREFIX_RECENCY_M,
    _build_final_row,
    _compute_beta_sweep_trend,
    _paired_bootstrap_ci,
    _parse_betas,
    _predicted_profile,
    _prefix_mass,
    _recency_mass,
)
from src.rollout.attention import AttentionConfig
from src.rollout.content_prior import ContentPriorConfig
from src.rollout.lambda_schedule import CustomLambdaScheduleConfig
from src.rollout.main import ModelConfig

ATOL = 1e-6


def _tiny_config(lambdas: list[float]) -> ModelConfig:
    return ModelConfig(
        sequence_length=4,
        num_layers=len(lambdas),
        attention_config=AttentionConfig(num_heads=1, alibi_slopes=[[0.5]]),
        content_config=ContentPriorConfig(diagonal_scores=[[0.0]], base_scores=[[0.0]]),
        lambda_schedule_config=CustomLambdaScheduleConfig(lambdas_custom=lambdas),
    )


def test_predicted_profile_at_beta_one_matches_unmodified_final_row() -> None:
    """beta=1.0 leaves lambda_t(beta) unchanged, so _predicted_profile == _build_final_row(lambdas)."""
    lambdas = [0.3, 0.6]
    config = _tiny_config(lambdas)

    predicted = _predicted_profile(config, lambdas, beta=1.0)
    unmodified = _build_final_row(config, lambdas, label="test")

    assert np.allclose(predicted, unmodified, atol=ATOL)


def test_prefix_and_recency_mass_pick_correct_positions() -> None:
    """prefix_mass sums positions 0..25, recency_mass sums positions 230..255, on a length-256 profile."""
    profile = np.arange(256, dtype=float)

    expected_prefix = float(np.sum(np.arange(PREFIX_RECENCY_M)))
    expected_recency = float(np.sum(np.arange(256 - PREFIX_RECENCY_M, 256)))

    assert _prefix_mass(profile) == expected_prefix
    assert _recency_mass(profile) == expected_recency


def test_paired_bootstrap_ci_is_deterministic_and_pairs_indices() -> None:
    """Same seed gives identical CIs; the resample must apply the same drawn indices to both inputs."""
    rng = np.random.default_rng(0)
    n = 16
    raw_beta = rng.uniform(size=(n, 256))
    raw_one = rng.uniform(size=(n, 256))

    ci_a = _paired_bootstrap_ci(raw_beta, raw_one, _prefix_mass, n_resamples=200, seed=42)
    ci_b = _paired_bootstrap_ci(raw_beta, raw_one, _prefix_mass, n_resamples=200, seed=42)
    assert ci_a == ci_b

    # If raw_beta and raw_one are identical, every paired resample's difference is exactly 0,
    # regardless of which indices are drawn (that's what "paired" buys us).
    ci_zero = _paired_bootstrap_ci(raw_beta, raw_beta, _prefix_mass, n_resamples=200, seed=1)
    assert ci_zero == (0.0, 0.0)


def test_results_table_reports_only_same_beta_w1() -> None:
    """The assembled per-beta W1 dict has exactly one entry per beta, never a cross-beta pair."""
    lambdas = [0.3, 0.6]
    config = _tiny_config(lambdas)
    betas = (0.8, 1.0, 1.2)

    w1_keys = {beta: _predicted_profile(config, lambdas, beta).shape for beta in betas}
    assert set(w1_keys.keys()) == set(betas)
    assert len(w1_keys) == len(betas)


def test_parse_betas_sorts_dedupes_and_requires_reference_point() -> None:
    """_parse_betas sorts and dedupes the comma-separated list, and rejects a set missing 1.0."""
    assert _parse_betas("1.2,0.8,1.0,0.8") == (0.8, 1.0, 1.2)
    assert _parse_betas("0.6,0.8,1.0,1.2,1.4") == (0.6, 0.8, 1.0, 1.2, 1.4)
    with pytest.raises(ValueError, match=r"1\.0"):
        _parse_betas("0.8,1.2")


def _profile_with_prefix_recency_mass(prefix_mass: float, recency_mass: float) -> np.ndarray:
    """Length-256 profile with an exact prefix mass and recency mass, uniform in between."""
    profile = np.zeros(256)
    profile[:PREFIX_RECENCY_M] = prefix_mass / PREFIX_RECENCY_M
    profile[-PREFIX_RECENCY_M:] = recency_mass / PREFIX_RECENCY_M
    middle = 256 - 2 * PREFIX_RECENCY_M
    profile[PREFIX_RECENCY_M:-PREFIX_RECENCY_M] = max(1.0 - prefix_mass - recency_mass, 0.0) / middle
    return profile


def test_compute_beta_sweep_trend_detects_perfect_monotonic_agreement() -> None:
    """
    When measured deltas track predicted deltas in rank order across beta, Spearman rho is exactly 1.0.

    Every prompt row is identical (zero variance across prompts), so the paired bootstrap CI must
    collapse to a single point at the same value -- a strong, exact check rather than an approximate
    range check.
    """
    betas = (1.4, 0.6, 1.0, 0.8, 1.2)  # deliberately unsorted
    n_prompts = 6
    prefix_mass_by_beta = {0.6: 0.05, 0.8: 0.08, 1.0: 0.10, 1.2: 0.13, 1.4: 0.17}  # increasing in beta
    recency_mass_by_beta = {0.6: 0.17, 0.8: 0.13, 1.0: 0.10, 1.2: 0.08, 1.4: 0.05}  # decreasing in beta
    raw = {
        beta: np.tile(
            _profile_with_prefix_recency_mass(prefix_mass_by_beta[beta], recency_mass_by_beta[beta]),
            (n_prompts, 1),
        )
        for beta in betas
    }
    delta_prefix_measured = {beta: prefix_mass_by_beta[beta] - prefix_mass_by_beta[1.0] for beta in betas}
    delta_recency_measured = {beta: recency_mass_by_beta[beta] - recency_mass_by_beta[1.0] for beta in betas}
    # Predicted deltas: same rank order as measured, different magnitudes -- only rank order matters.
    delta_prefix_predicted = {0.6: -0.06, 0.8: -0.03, 1.0: 0.0, 1.2: 0.02, 1.4: 0.05}
    delta_recency_predicted = {beta: -delta_prefix_predicted[beta] for beta in betas}

    stats = _compute_beta_sweep_trend(
        raw,
        delta_prefix_predicted,
        delta_recency_predicted,
        delta_prefix_measured,
        delta_recency_measured=delta_recency_measured,
        betas=betas,
        bootstrap_samples=25,
        seed=0,
    )

    assert stats.prefix_trend == pytest.approx(1.0)
    assert stats.recency_trend == pytest.approx(1.0)
    assert stats.prefix_trend_ci == pytest.approx((1.0, 1.0))
    assert stats.recency_trend_ci == pytest.approx((1.0, 1.0))


def test_compute_beta_sweep_trend_ci_is_deterministic() -> None:
    """Same seed gives an identical beta-sweep trend CI."""
    betas = (0.6, 0.8, 1.0, 1.2, 1.4)
    n_prompts = 8
    rng = np.random.default_rng(1)
    raw = {beta: rng.uniform(size=(n_prompts, 256)) for beta in betas}
    delta_prefix_measured = {beta: _prefix_mass(raw[beta].mean(axis=0)) - _prefix_mass(raw[1.0].mean(axis=0)) for beta in betas}
    delta_recency_measured = {
        beta: _recency_mass(raw[beta].mean(axis=0)) - _recency_mass(raw[1.0].mean(axis=0)) for beta in betas
    }
    delta_prefix_predicted = {0.6: -0.06, 0.8: -0.03, 1.0: 0.0, 1.2: 0.02, 1.4: 0.05}
    delta_recency_predicted = {beta: -delta_prefix_predicted[beta] for beta in betas}

    kwargs = {
        "delta_prefix_measured": delta_prefix_measured,
        "delta_recency_measured": delta_recency_measured,
        "betas": betas,
        "bootstrap_samples": 50,
        "seed": 7,
    }
    stats_a = _compute_beta_sweep_trend(raw, delta_prefix_predicted, delta_recency_predicted, **kwargs)
    stats_b = _compute_beta_sweep_trend(raw, delta_prefix_predicted, delta_recency_predicted, **kwargs)

    assert stats_a == stats_b
