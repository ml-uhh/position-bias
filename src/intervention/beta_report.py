r"""
assemble the no-refit rollout prediction vs. measured-gradient beta table.

Pure CPU / config-driven -- no model forward passes here. Predicted profiles use the paper's
content-free ALiBi kernel and norm-Shapley schedules computed from baseline forward branch
statistics. Measured profiles come from the raw per-prompt gradient tensors already saved on disk
by ``src.intervention.beta_gradient``. No parameter is fitted to the measured profiles.

Run, e.g.::

    uv run -m src.intervention.beta_report \
        --root results/intervention \
        --baseline-rollout-config config/rollout/alibi/no_content/mpt-7b.yaml \
        --bootstrap-samples 1000 --seed 42

"""

from collections.abc import Callable
import csv
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.stats import spearmanr
import structlog
import torch
import typer

from src.metrics.metrics import normalized_wasserstein_1
from src.intervention.shapley_baseline import mean_shapley_schedule
from src.rollout.lambda_schedule import CustomLambdaScheduleConfig
from src.rollout.main import ModelConfig, compute_model_results
from src.util.pydantic import load_config

logger = structlog.get_logger()

DEFAULT_ROOT = Path("results/intervention")
DEFAULT_BASELINE_ROLLOUT_CONFIG = Path("config/rollout/alibi/no_content/mpt-7b.yaml")
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_BETAS = (0.8, 1.0, 1.2)
PREFIX_RECENCY_M = 26  # ceil(0.1 * 256)
GRADIENT_VS_BLANKOUT_W1_REFERENCE = 0.0528
NEGATIVE_ENTRY_TOL = -1e-6
CAUSAL_LEAK_TOL = 1e-4
ROW_STOCHASTIC_TOL = 1e-4


def identity_baseline(n: int) -> np.ndarray:
    return np.eye(n)


def first_token_baseline(n: int) -> np.ndarray:
    matrix = np.zeros((n, n))
    matrix[:, 0] = 1.0
    return matrix


def uniform_causal_baseline(n: int) -> np.ndarray:
    matrix = np.zeros((n, n))
    for row in range(n):
        matrix[row, : row + 1] = 1.0 / (row + 1)
    return matrix


def validate_rollout(matrix: np.ndarray, name: str) -> None:
    """Check nonnegativity, causality, and row normalization."""
    future = np.triu(np.ones(matrix.shape, dtype=bool), k=1)
    causal_leak = float(np.abs(matrix[future]).max()) if future.any() else 0.0
    row_error = float(np.abs(matrix.sum(axis=1) - 1.0).max())
    if matrix.min() < NEGATIVE_ENTRY_TOL or causal_leak > CAUSAL_LEAK_TOL or row_error > ROW_STOCHASTIC_TOL:
        raise ValueError(
            f"{name} is not a causal stochastic rollout: min={matrix.min():.3e}, "
            f"leak={causal_leak:.3e}, row_error={row_error:.3e}",
        )


def _final_row(scores_product: np.ndarray) -> np.ndarray:
    row = scores_product[-1].copy()
    total = row.sum()
    if total > 0:
        row = row / total
    return row


def _load_shapley_schedules(branch_stats: Path, betas: tuple[float, ...]) -> dict[float, list[float]]:
    """Compute corpus-mean counterfactual schedules from unmodified forward statistics."""
    payload = torch.load(branch_stats, map_location="cpu", weights_only=False)
    if payload.get("definition") != "frobenius_norm_shapley":
        raise ValueError(f"unsupported branch-statistics definition in {branch_stats}")
    statistics = payload["statistics"].to(torch.float64)
    schedules = {beta: mean_shapley_schedule(statistics, beta) for beta in betas}
    if not np.all(np.diff(np.asarray([schedules[beta] for beta in sorted(betas)]), axis=0) >= -1e-12):
        raise ValueError("norm-Shapley schedules are not monotone in beta")
    return schedules


def _build_final_row(base_config: ModelConfig, lambdas: list[float], *, label: str) -> np.ndarray:
    config = base_config.model_copy(
        update={"lambda_schedule_config": CustomLambdaScheduleConfig(lambdas_custom=lambdas)},
    )
    results = compute_model_results(config)
    validate_rollout(results.scores_product, label)
    return _final_row(results.scores_product)


def _predicted_profile(
    base_config: ModelConfig,
    transformed_schedule: list[float],
    beta: float,
) -> np.ndarray:
    """Build a prediction from the supplied norm-Shapley schedule."""
    return _build_final_row(
        base_config,
        transformed_schedule,
        label=f"predicted P(beta={beta})",
    )


def _normalize(profile: np.ndarray) -> np.ndarray:
    return profile / profile.sum()


def _prefix_mass(profile: np.ndarray) -> float:
    return float(np.sum(profile[:PREFIX_RECENCY_M]))


def _recency_mass(profile: np.ndarray) -> float:
    return float(np.sum(profile[-PREFIX_RECENCY_M:]))


def _paired_bootstrap_ci(
    raw_beta: np.ndarray,
    raw_one: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap CI for stat_fn(g_beta) - stat_fn(g_one), pairing the same resampled prompts."""
    rng = np.random.default_rng(seed)
    n = raw_beta.shape[0]
    diffs = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        g_beta = _normalize(raw_beta[idx].mean(axis=0))
        g_one = _normalize(raw_one[idx].mean(axis=0))
        diffs[b] = stat_fn(g_beta) - stat_fn(g_one)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _kl_divergence(logits_beta: torch.Tensor, logits_one: torch.Tensor) -> float:
    """Mean per-prompt KL(P_beta || P_1) over the next-token distribution at the final position."""
    log_p_beta = torch.log_softmax(logits_beta, dim=-1)
    log_p_one = torch.log_softmax(logits_one, dim=-1)
    p_beta = log_p_beta.exp()
    kl = (p_beta * (log_p_beta - log_p_one)).sum(dim=-1)
    return float(kl.mean())


def _direction_label(delta: float, *, increase_word: str, decrease_word: str) -> str:
    return increase_word if delta > 0 else decrease_word


def _load_base_config(baseline_rollout_config: Path) -> ModelConfig:
    base_config = load_config(baseline_rollout_config, ModelConfig)
    lambda_schedule = base_config.lambda_schedule_config
    if not isinstance(lambda_schedule, CustomLambdaScheduleConfig):
        raise TypeError(
            f"{baseline_rollout_config} must use a custom lambda schedule, got {type(lambda_schedule)}",
        )
    return base_config


def _load_beta_outputs(
    root: Path,
    base_config: ModelConfig,
    betas: tuple[float, ...],
    shapley_schedules: dict[float, list[float]],
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], dict[float, torch.Tensor], dict[float, np.ndarray]]:
    """Load raw/aggregated gradient tensors per beta, and compute+save each beta's predicted profile."""
    raw: dict[float, np.ndarray] = {}
    grad_profile: dict[float, np.ndarray] = {}
    final_logits: dict[float, torch.Tensor] = {}
    predicted: dict[float, np.ndarray] = {}
    for beta in betas:
        beta_dir = root / f"beta_{beta}"
        raw[beta] = torch.load(beta_dir / "raw_gradient_profiles.pt", map_location="cpu", weights_only=True).numpy()
        grad_profile[beta] = torch.load(
            beta_dir / "gradient_profile.pt",
            map_location="cpu",
            weights_only=True,
        ).numpy()
        final_logits[beta] = torch.load(beta_dir / "final_logits.pt", map_location="cpu", weights_only=True)
        predicted[beta] = _predicted_profile(
            base_config,
            shapley_schedules[beta],
            beta,
        )
        torch.save(torch.from_numpy(predicted[beta]), beta_dir / "predicted_rollout.pt")
    return raw, grad_profile, final_logits, predicted


def _compute_deltas(
    predicted: dict[float, np.ndarray],
    grad_profile: dict[float, np.ndarray],
    betas: tuple[float, ...],
) -> tuple[dict[float, float], dict[float, float], dict[float, float], dict[float, float]]:
    delta_prefix_predicted = {beta: _prefix_mass(predicted[beta]) - _prefix_mass(predicted[1.0]) for beta in betas}
    delta_recency_predicted = {beta: _recency_mass(predicted[beta]) - _recency_mass(predicted[1.0]) for beta in betas}
    delta_prefix_measured = {beta: _prefix_mass(grad_profile[beta]) - _prefix_mass(grad_profile[1.0]) for beta in betas}
    delta_recency_measured = {
        beta: _recency_mass(grad_profile[beta]) - _recency_mass(grad_profile[1.0]) for beta in betas
    }
    return delta_prefix_predicted, delta_recency_predicted, delta_prefix_measured, delta_recency_measured


def _compute_cis(
    raw: dict[float, np.ndarray],
    betas: tuple[float, ...],
    *,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[float, tuple[float, float]], dict[float, tuple[float, float]]]:
    ci_prefix: dict[float, tuple[float, float]] = {}
    ci_recency: dict[float, tuple[float, float]] = {}
    for beta in betas:
        if beta == 1.0:
            continue
        ci_prefix[beta] = _paired_bootstrap_ci(raw[beta], raw[1.0], _prefix_mass, n_resamples=bootstrap_samples, seed=seed)
        ci_recency[beta] = _paired_bootstrap_ci(
            raw[beta],
            raw[1.0],
            _recency_mass,
            n_resamples=bootstrap_samples,
            seed=seed,
        )
    return ci_prefix, ci_recency


def _delta_profile(profile: dict[float, np.ndarray], beta: float) -> np.ndarray:
    return profile[beta] - profile[1.0]


def _sign_agreement(delta_predicted: np.ndarray, delta_measured: np.ndarray) -> float:
    """
    Fraction of positions where the measured delta has the same sign as the predicted delta.

    Positions where the predicted delta is exactly zero make no directional claim and are excluded
    from both the numerator and denominator.
    """
    nonzero = delta_predicted != 0
    if not np.any(nonzero):
        return float("nan")
    agree = np.sign(delta_predicted[nonzero]) == np.sign(delta_measured[nonzero])
    return float(np.mean(agree))


def _spearman_corr(delta_predicted: np.ndarray, delta_measured: np.ndarray) -> float:
    return float(spearmanr(delta_predicted, delta_measured)[0])  # pyright: ignore[reportArgumentType]


def _paired_profile_bootstrap_ci(
    raw_beta: np.ndarray,
    raw_one: np.ndarray,
    delta_predicted: np.ndarray,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for stat_fn(delta_predicted, delta_measured_resampled).

    The predicted delta profile is a fixed, config-derived quantity (no prompt dependence), so only
    the measured side is resampled, pairing the same drawn prompt indices for beta and beta=1.0 --
    matching the convention already used by ``_paired_bootstrap_ci``.
    """
    rng = np.random.default_rng(seed)
    n = raw_beta.shape[0]
    values = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        g_beta = _normalize(raw_beta[idx].mean(axis=0))
        g_one = _normalize(raw_one[idx].mean(axis=0))
        values[b] = stat_fn(delta_predicted, g_beta - g_one)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


class ProfileTrendStats(NamedTuple):
    """Whole-profile trend statistics for one beta, point estimate plus paired-bootstrap CI."""

    sign_agreement: float
    sign_agreement_ci: tuple[float, float]
    spearman: float
    spearman_ci: tuple[float, float]

    def as_dict(self) -> dict[str, float | tuple[float, float]]:
        """Convert to a plain dict for JSON serialization (namedtuples serialize as lists otherwise)."""
        return self._asdict()


def _compute_profile_trend(
    raw: dict[float, np.ndarray],
    predicted: dict[float, np.ndarray],
    grad_profile: dict[float, np.ndarray],
    betas: tuple[float, ...],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[float, ProfileTrendStats]:
    """Whole-profile (all 256 positions) trend statistics, supplementary to the prefix/recency table."""
    trend: dict[float, ProfileTrendStats] = {}
    for beta in betas:
        if beta == 1.0:
            continue
        delta_predicted = _delta_profile(predicted, beta)
        delta_measured = _delta_profile(grad_profile, beta)
        trend[beta] = ProfileTrendStats(
            sign_agreement=_sign_agreement(delta_predicted, delta_measured),
            sign_agreement_ci=_paired_profile_bootstrap_ci(
                raw[beta],
                raw[1.0],
                delta_predicted,
                _sign_agreement,
                n_resamples=bootstrap_samples,
                seed=seed,
            ),
            spearman=_spearman_corr(delta_predicted, delta_measured),
            spearman_ci=_paired_profile_bootstrap_ci(
                raw[beta],
                raw[1.0],
                delta_predicted,
                _spearman_corr,
                n_resamples=bootstrap_samples,
                seed=seed,
            ),
        )
    return trend


class BetaSweepTrendStats(NamedTuple):
    """Cross-beta trend statistics (predicted vs. measured delta as a function of beta itself)."""

    prefix_trend: float
    prefix_trend_ci: tuple[float, float]
    recency_trend: float
    recency_trend_ci: tuple[float, float]

    def as_dict(self) -> dict[str, float | tuple[float, float]]:
        """Convert to a plain dict for JSON serialization (namedtuples serialize as lists otherwise)."""
        return self._asdict()


def _paired_sweep_bootstrap_ci(
    raw: dict[float, np.ndarray],
    betas_sorted: list[float],
    predicted_delta: dict[float, float],
    mass_fn: Callable[[np.ndarray], float],
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for spearman(predicted_delta_by_beta, measured_delta_by_beta_resampled).

    All betas share the same underlying prompt set (one manifest reused across every beta run), so
    each resample draws one set of prompt indices and applies it consistently to every beta -- the
    same paired-resampling convention as ``_paired_bootstrap_ci``/``_paired_profile_bootstrap_ci``,
    extended from a single beta-vs-beta=1.0 comparison to the full beta sweep at once.
    """
    rng = np.random.default_rng(seed)
    n = raw[1.0].shape[0]
    pred = np.array([predicted_delta[beta] for beta in betas_sorted])
    values = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        g_one = _normalize(raw[1.0][idx].mean(axis=0))
        mass_one = mass_fn(g_one)
        measured = np.array([mass_fn(_normalize(raw[beta][idx].mean(axis=0))) - mass_one for beta in betas_sorted])
        values[i] = _spearman_corr(pred, measured)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _compute_beta_sweep_trend(
    raw: dict[float, np.ndarray],
    delta_prefix_predicted: dict[float, float],
    delta_recency_predicted: dict[float, float],
    delta_prefix_measured: dict[float, float],
    *,
    delta_recency_measured: dict[float, float],
    betas: tuple[float, ...],
    bootstrap_samples: int,
    seed: int,
) -> BetaSweepTrendStats:
    """
    Compute trend statistics across the beta axis.

    Tests whether the measured delta tracks the predicted delta as beta varies. Unlike the primary
    table's per-beta pairwise tests (each beta compared to beta=1.0 in isolation)
    or the full-profile statistics (per-beta, across positions), this correlates the *sequence* of
    deltas across all betas at once, a monotonic-trend
    test can find power in the shared direction across betas even where no single pairwise comparison
    clears significance on its own.
    """
    betas_sorted = sorted(betas)
    pred_prefix = np.array([delta_prefix_predicted[beta] for beta in betas_sorted])
    meas_prefix = np.array([delta_prefix_measured[beta] for beta in betas_sorted])
    pred_recency = np.array([delta_recency_predicted[beta] for beta in betas_sorted])
    meas_recency = np.array([delta_recency_measured[beta] for beta in betas_sorted])
    return BetaSweepTrendStats(
        prefix_trend=_spearman_corr(pred_prefix, meas_prefix),
        prefix_trend_ci=_paired_sweep_bootstrap_ci(
            raw,
            betas_sorted,
            delta_prefix_predicted,
            _prefix_mass,
            n_resamples=bootstrap_samples,
            seed=seed,
        ),
        recency_trend=_spearman_corr(pred_recency, meas_recency),
        recency_trend_ci=_paired_sweep_bootstrap_ci(
            raw,
            betas_sorted,
            delta_recency_predicted,
            _recency_mass,
            n_resamples=bootstrap_samples,
            seed=seed,
        ),
    )


def _compute_baseline_w1(
    base_config: ModelConfig,
    num_layers: int,
    grad_profile_beta_one: np.ndarray,
) -> dict[str, float]:
    identity_profile = identity_baseline(256)[-1]
    first_token_profile = first_token_baseline(256)[-1]
    uniform_profile = uniform_causal_baseline(256)[-1]
    attention_only_profile = _build_final_row(
        base_config,
        [1.0] * num_layers,
        label="attention-only (lambda=1) baseline",
    )
    return {
        "First-token": normalized_wasserstein_1(first_token_profile, grad_profile_beta_one),
        "Identity/last-token": normalized_wasserstein_1(identity_profile, grad_profile_beta_one),
        "Uniform": normalized_wasserstein_1(uniform_profile, grad_profile_beta_one),
        "Attention-only (lambda_t=1)": normalized_wasserstein_1(attention_only_profile, grad_profile_beta_one),
    }


def _compute_stability(final_logits: dict[float, torch.Tensor], betas: tuple[float, ...]) -> dict[float, dict[str, float]]:
    stability: dict[float, dict[str, float]] = {}
    for beta in betas:
        if beta == 1.0:
            continue
        kl = _kl_divergence(final_logits[beta], final_logits[1.0])
        top1_beta = final_logits[beta].argmax(dim=-1)
        top1_one = final_logits[1.0].argmax(dim=-1)
        top1_unchanged_fraction = float((top1_beta == top1_one).float().mean())
        stability[beta] = {"mean_kl": kl, "top1_unchanged_fraction": top1_unchanged_fraction}
    return stability


def _build_table_rows(
    w1: dict[float, float],
    delta_prefix_predicted: dict[float, float],
    delta_recency_predicted: dict[float, float],
    delta_prefix_measured: dict[float, float],
    betas: tuple[float, ...],
    *,
    delta_recency_measured: dict[float, float],
    ci_prefix: dict[float, tuple[float, float]],
    ci_recency: dict[float, tuple[float, float]],
) -> tuple[list[str], list[list[str]]]:
    header = [
        "beta",
        "Predicted prefix change",
        "Measured prefix change [95% CI]",
        "Predicted recency change",
        "Measured recency change [95% CI]",
        "Rollout vs. gradient W1",
    ]
    rows = []
    for beta in betas:
        if beta == 1.0:
            rows.append([str(beta), "reference", "reference", "reference", "reference", f"{w1[beta]:.4f}"])
            continue
        prefix_dir = _direction_label(delta_prefix_predicted[beta], increase_word="increase", decrease_word="decrease")
        recency_dir = _direction_label(delta_recency_predicted[beta], increase_word="increase", decrease_word="decrease")
        prefix_lo, prefix_hi = ci_prefix[beta]
        recency_lo, recency_hi = ci_recency[beta]
        rows.append(
            [
                str(beta),
                prefix_dir,
                f"{delta_prefix_measured[beta]:+.4f} [{prefix_lo:+.4f}, {prefix_hi:+.4f}]",
                recency_dir,
                f"{delta_recency_measured[beta]:+.4f} [{recency_lo:+.4f}, {recency_hi:+.4f}]",
                f"{w1[beta]:.4f}",
            ],
        )
    return header, rows


def _write_csv(root: Path, stem: str, header: list[str], rows: list[list[str]]) -> None:
    with (root / f"{stem}.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_markdown(
    root: Path,
    stem: str,
    header: list[str],
    rows: list[list[str]],
    delta_prefix_predicted: dict[float, float],
    *,
    betas: tuple[float, ...],
    delta_recency_predicted: dict[float, float],
    baseline_w1: dict[str, float],
    stability: dict[float, dict[str, float]],
    profile_trend: dict[float, ProfileTrendStats],
    beta_sweep_trend: BetaSweepTrendStats | None,
    bootstrap_samples: int,
    seed: int,
    num_prompts: int,
    calibration: str,
) -> None:
    is_sweep = beta_sweep_trend is not None
    betas_str = ", ".join(str(b) for b in betas)
    primary_table_note = (
        "Predicted columns report only the pre-specified direction (no post-intervention rollout "
        "remeasurement or curve fitting); measured columns are paired prompt-bootstrap "
        f"(seed {seed}, {bootstrap_samples} resamples) 95% CIs. `beta=1.0` is the fixed reference "
        "point (delta=0 by construction). `W1` compares only the predicted rollout and measured "
        "gradient profile for the *same* beta -- no cross-beta W1 is reported."
    )
    baseline_note = (
        "Normalized W1 lies in [0,1]; at n=256, `255*W1` is the corresponding average token "
        "transport. The existing mean gradient-versus-blank-out distance "
        f"`{GRADIENT_VS_BLANKOUT_W1_REFERENCE}` is cited here as an empirical cross-proxy "
        "reference (from prior blank-out/gradient agreement), not a mathematical lower bound, "
        "and is not recomputed by this script."
    )
    disclosure_prompts = (
        f"- The same {num_prompts} prompts and the same fixed baseline target y(x) (from the "
        f"unmodified, beta=1.0 model) are used across all {len(betas)} beta conditions."
    )
    disclosure_no_refit = (
        "- Each predicted profile uses the paper's content-free ALiBi kernel and the "
        "norm-Shapley schedule computed from residual/attention geometry measured once in the "
        "unmodified model. No quantity is remeasured after intervention, and no influence "
        "measurement enters the prediction."
    )
    disclosure_stability = (
        "- The model-stability KL/top-1 diagnostics above are reported for transparency only and "
        "were not used to choose or adjust the beta values."
    )
    profile_trend_note = (
        "Supplementary to the primary table: the primary prefix/recency masses each sum only 26 of "
        "256 positions (~10% each per tail); these statistics use the full predicted and measured "
        "delta-per-position profiles (`predicted[beta] - predicted[1.0]` and "
        "`gradient_profile[beta] - gradient_profile[1.0]`, all 256 positions). Sign agreement is the "
        "fraction of positions where the measured delta has the same sign as the predicted delta "
        "(chance level 0.50); Spearman rho is the rank correlation between the two full delta "
        "profiles. Both use the same paired prompt-bootstrap "
        f"(seed {seed}, {bootstrap_samples} resamples) convention as the primary table. Added after "
        "the primary table and "
        "not used to revise the beta values or the primary table's conclusions."
    )
    beta_sweep_note = (
        "This report extends the pre-registered beta={0.8,1.0,1.2} comparison with additional beta "
        f"values ({betas_str}) for the configured intervention values, independent of any "
        "sweep results. It does **not** supersede the original three-beta `beta_results.md`/`.csv` in "
        "this directory, which remain unchanged and are the primary pre-registered result. The trend "
        "statistic below is the Spearman rank correlation, across all betas in this sweep, between "
        "each beta's predicted delta (prefix or recency mass vs. beta=1.0) and its measured delta -- "
        "testing whether the two track together as beta varies, rather than treating each beta as an "
        "isolated pairwise comparison. Its bootstrap CI resamples prompts once per resample and "
        "applies the same resampled indices to every beta (the shared prompt manifest is reused "
        "unmodified across the whole sweep), consistent with the paired-resampling convention used "
        "throughout this script."
    )

    md_lines = [
        f"# Attention-scaling intervention results{' -- beta sweep' if is_sweep else ''}",
        "",
        "## Primary table",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    md_lines.extend("| " + " | ".join(row) + " |" for row in rows)
    md_lines.extend(
        [
            "",
            primary_table_note,
            "",
            "### Predicted delta magnitudes (no-refit rollout, for context; direction is the pre-registered claim)",
            "",
            "| beta | Predicted Delta_prefix | Predicted Delta_recency |",
            "|---:|---:|---:|",
        ],
    )
    md_lines.extend(
        f"| {beta} | {delta_prefix_predicted[beta]:+.4f} | {delta_recency_predicted[beta]:+.4f} |" for beta in betas
    )
    md_lines.extend(
        [
            "",
            "## Metric scale and baselines (beta=1.0 gradient profile vs. reference distributions)",
            "",
            "| Baseline | W1 |",
            "|---|---:|",
        ],
    )
    md_lines.extend(f"| {name} | {value:.4f} |" for name, value in baseline_w1.items())
    md_lines.extend(
        [
            "",
            baseline_note,
            "",
            "## Model-stability sanity check (diagnostic only -- not used to select or revise beta)",
            "",
            "| beta | Mean next-token KL vs. beta=1.0 | Fraction top-1 unchanged |",
            "|---:|---:|---:|",
        ],
    )
    md_lines.extend(
        f"| {beta} | {stability[beta]['mean_kl']:.4f} | {stability[beta]['top1_unchanged_fraction']:.4f} |"
        for beta in betas
        if beta != 1.0
    )
    md_lines.extend(
        [
            "",
            "## Full-profile trend statistics (all 256 positions, supplementary to the primary table)",
            "",
            "| beta | Sign agreement [95% CI] | Spearman rho (predicted vs. measured delta) [95% CI] |",
            "|---:|---|---|",
        ],
    )
    for beta in betas:
        if beta == 1.0:
            continue
        stats = profile_trend[beta]
        sign_lo, sign_hi = stats.sign_agreement_ci
        rho_lo, rho_hi = stats.spearman_ci
        md_lines.append(
            f"| {beta} | {stats.sign_agreement:.4f} [{sign_lo:.4f}, {sign_hi:.4f}] "
            f"| {stats.spearman:+.4f} [{rho_lo:+.4f}, {rho_hi:+.4f}] |",
        )
    md_lines.extend(["", profile_trend_note])
    if beta_sweep_trend is not None:
        prefix_lo, prefix_hi = beta_sweep_trend.prefix_trend_ci
        recency_lo, recency_hi = beta_sweep_trend.recency_trend_ci
        md_lines.extend(
            [
                "",
                f"## Beta-sweep trend test (betas: {betas_str})",
                "",
                "| Mass | Spearman rho (predicted vs. measured delta, across betas) [95% CI] |",
                "|---|---|",
                f"| Prefix | {beta_sweep_trend.prefix_trend:+.4f} [{prefix_lo:+.4f}, {prefix_hi:+.4f}] |",
                f"| Recency | {beta_sweep_trend.recency_trend:+.4f} [{recency_lo:+.4f}, {recency_hi:+.4f}] |",
                "",
                beta_sweep_note,
            ],
        )
    md_lines.extend(
        [
            "",
            "## Disclosures",
            "",
            f"- Betas fixed at {betas_str} before inspecting any results; not revised afterward.",
            disclosure_prompts,
            disclosure_no_refit,
            disclosure_stability,
        ],
    )
    (root / f"{stem}.md").write_text("\n".join(md_lines) + "\n")


def _parse_betas(betas: str) -> tuple[float, ...]:
    parsed = tuple(sorted({float(b) for b in betas.split(",")}))
    if 1.0 not in parsed:
        raise ValueError(f"betas must include 1.0 (the fixed reference point), got {betas}")
    return parsed


def main(
    root: Path = DEFAULT_ROOT,
    baseline_rollout_config: Path = DEFAULT_BASELINE_ROLLOUT_CONFIG,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 42,
    betas: str = "0.8,1.0,1.2",
    branch_stats: Path = Path("results/intervention/baseline_branch_stats.pt"),
) -> None:
    """
    Assemble the Q3 beta-intervention results table from saved raw tensors. CPU-only.

    Args:
        root: Directory containing ``prompt_manifest.pt`` and ``beta_{beta}/`` subdirectories
            written by ``make_manifest``/``beta_gradient``.
        baseline_rollout_config: Path to the paper's variant (b) rollout config.
        bootstrap_samples: Paired prompt-bootstrap resamples for measured-delta 95% CIs.
        seed: Bootstrap seed.
        betas: Comma-separated beta values to report on; must include 1.0. The default
            (0.8, 1.0, 1.2) writes ``beta_results.md/.csv`` (the pre-registered primary report, never
            overwritten by a larger sweep). Any other set writes ``beta_sweep_results.md/.csv``
            instead and adds the beta-sweep trend-test section, for the configured beta sweep
            option 3 -- it does not touch or supersede the primary report's files.
        branch_stats: Baseline branch-statistics file written by ``shapley_baseline``.

    """
    beta_values = _parse_betas(betas)
    is_sweep = beta_values != DEFAULT_BETAS
    stem = "beta_sweep_results" if is_sweep else "beta_results"

    base_config = _load_base_config(baseline_rollout_config)
    shapley_schedules = _load_shapley_schedules(branch_stats, beta_values)
    calibration = "norm_shapley"
    raw, grad_profile, final_logits, predicted = _load_beta_outputs(
        root,
        base_config,
        beta_values,
        shapley_schedules,
    )

    w1 = {beta: normalized_wasserstein_1(predicted[beta], grad_profile[beta]) for beta in beta_values}
    delta_prefix_predicted, delta_recency_predicted, delta_prefix_measured, delta_recency_measured = _compute_deltas(
        predicted,
        grad_profile,
        beta_values,
    )
    ci_prefix, ci_recency = _compute_cis(raw, beta_values, bootstrap_samples=bootstrap_samples, seed=seed)
    baseline_w1 = _compute_baseline_w1(base_config, base_config.num_layers, grad_profile[1.0])
    stability = _compute_stability(final_logits, beta_values)
    profile_trend = _compute_profile_trend(
        raw,
        predicted,
        grad_profile,
        beta_values,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    beta_sweep_trend = (
        _compute_beta_sweep_trend(
            raw,
            delta_prefix_predicted,
            delta_recency_predicted,
            delta_prefix_measured,
            delta_recency_measured=delta_recency_measured,
            betas=beta_values,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        if is_sweep
        else None
    )

    header, rows = _build_table_rows(
        w1,
        delta_prefix_predicted,
        delta_recency_predicted,
        delta_prefix_measured,
        beta_values,
        delta_recency_measured=delta_recency_measured,
        ci_prefix=ci_prefix,
        ci_recency=ci_recency,
    )

    root.mkdir(parents=True, exist_ok=True)
    _write_csv(root, stem, header, rows)
    _write_markdown(
        root,
        stem,
        header,
        rows,
        delta_prefix_predicted,
        betas=beta_values,
        delta_recency_predicted=delta_recency_predicted,
        baseline_w1=baseline_w1,
        stability=stability,
        profile_trend=profile_trend,
        beta_sweep_trend=beta_sweep_trend,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        num_prompts=raw[1.0].shape[0],
        calibration=calibration,
    )

    metrics_out = {
        "betas": beta_values,
        "w1": w1,
        "delta_prefix_predicted": delta_prefix_predicted,
        "delta_recency_predicted": delta_recency_predicted,
        "delta_prefix_measured": delta_prefix_measured,
        "delta_recency_measured": delta_recency_measured,
        "ci_prefix": ci_prefix,
        "ci_recency": ci_recency,
        "baseline_w1": baseline_w1,
        "stability": stability,
        "profile_trend": {beta: stats.as_dict() for beta, stats in profile_trend.items()},
        "beta_sweep_trend": beta_sweep_trend.as_dict() if beta_sweep_trend is not None else None,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "calibration": calibration,
        "lambda_schedules": shapley_schedules,
    }
    (root / f"{stem}_metrics.json").write_text(json.dumps(metrics_out, indent=2))
    logger.info(f"Wrote {stem}.md/csv", root=str(root))


if __name__ == "__main__":
    typer.run(main)
