"""No-refit intervention report for projection-aware RoPE transport kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.metrics.metrics import normalized_wasserstein_1
from src.intervention.beta_report import (
    _compute_cis,
    _compute_deltas,
    _compute_profile_trend,
    _compute_stability,
    _load_shapley_schedules,
)


def _rollout(attention: np.ndarray, lambdas: list[float]) -> np.ndarray:
    """Return the final-token rollout through fixed layerwise transport kernels."""
    profile = np.zeros(attention.shape[-1], dtype=np.float64)
    profile[-1] = 1.0
    for layer in range(attention.shape[0] - 1, -1, -1):
        transported = profile @ attention[layer].astype(np.float64)
        profile = (1.0 - lambdas[layer]) * profile + lambdas[layer] * transported
    return profile / profile.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--kernel-npz", required=True, type=Path)
    parser.add_argument("--branch-stats", required=True, type=Path)
    parser.add_argument("--betas", default="0.8,1.0,1.2")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    betas = tuple(float(value) for value in args.betas.split(","))
    if 1.0 not in betas:
        raise ValueError("--betas must contain 1.0")
    with np.load(args.kernel_npz) as artifact:
        attention = np.asarray(artifact["mean_forward_alti_attention"])
    schedules = _load_shapley_schedules(args.branch_stats, betas)
    predicted = {beta: _rollout(attention, schedules[beta]) for beta in betas}
    raw = {
        beta: torch.load(
            args.root / f"beta_{beta}" / "raw_gradient_profiles.pt", map_location="cpu", weights_only=True
        ).numpy()
        for beta in betas
    }
    measured = {
        beta: torch.load(
            args.root / f"beta_{beta}" / "gradient_profile.pt", map_location="cpu", weights_only=True
        ).numpy()
        for beta in betas
    }
    logits = {
        beta: torch.load(args.root / f"beta_{beta}" / "final_logits.pt", map_location="cpu", weights_only=True)
        for beta in betas
    }
    w1 = {beta: normalized_wasserstein_1(predicted[beta], measured[beta]) for beta in betas}
    delta_prefix_predicted, delta_recency_predicted, delta_prefix_measured, delta_recency_measured = _compute_deltas(
        predicted, measured, betas
    )
    ci_prefix, ci_recency = _compute_cis(raw, betas, bootstrap_samples=args.bootstrap, seed=42)
    stability = _compute_stability(logits, betas)
    profile_trend = _compute_profile_trend(
        raw, predicted, measured, betas, bootstrap_samples=args.bootstrap, seed=42
    )
    result = {
        "betas": betas,
        "w1": w1,
        "delta_prefix_predicted": delta_prefix_predicted,
        "delta_recency_predicted": delta_recency_predicted,
        "delta_prefix_measured": delta_prefix_measured,
        "delta_recency_measured": delta_recency_measured,
        "ci_prefix": ci_prefix,
        "ci_recency": ci_recency,
        "stability": stability,
        "profile_trend": {beta: value.as_dict() for beta, value in profile_trend.items()},
        "lambda_schedules": schedules,
        "kernel_source": str(args.kernel_npz),
        "calibration": "norm_shapley",
    }
    output = args.root / "rope_beta_results_metrics.json"
    output.write_text(json.dumps(result, indent=2))
    for beta in betas:
        torch.save(torch.from_numpy(predicted[beta]), args.root / f"beta_{beta}" / "predicted_rollout.pt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
