# ruff: noqa: INP001, T201
"""Report paper-style stochastic-monotonicity violations for layer kernels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _measure_kernel(kernel: np.ndarray, threshold: float) -> dict[str, float | int]:
    cumulative = np.cumsum(kernel.astype(np.float64), axis=-1)
    triple_count = 0
    literal_count = 0
    literal_sum = 0.0
    threshold_count = 0
    threshold_sum = 0.0
    maximum = 0.0
    for row_index in range(cumulative.shape[0] - 1):
        gaps = (
            cumulative[row_index + 1 :]
            - cumulative[row_index][None, :]
        ).ravel()
        positive = gaps[gaps > 0.0]
        above_threshold = gaps[gaps > threshold]
        triple_count += gaps.size
        literal_count += positive.size
        literal_sum += positive.sum()
        threshold_count += above_threshold.size
        threshold_sum += above_threshold.sum()
        if positive.size:
            maximum = max(maximum, float(positive.max()))
    return {
        "triple_count": triple_count,
        "literal_count": literal_count,
        "literal_sum": literal_sum,
        "threshold_count": threshold_count,
        "threshold_sum": threshold_sum,
        "max_gap": maximum,
    }


def _finalize(
    values: dict[str, float | int],
    *,
    layer: str | int,
) -> dict[str, float | int | str]:
    triples = int(values["triple_count"])
    literal_count = int(values["literal_count"])
    threshold_count = int(values["threshold_count"])
    literal_sum = float(values["literal_sum"])
    threshold_sum = float(values["threshold_sum"])
    return {
        "layer": layer,
        "triple_count": triples,
        "literal_violation_rate": literal_count / triples,
        "literal_conditional_mean": literal_sum / max(literal_count, 1),
        "literal_unconditional_mean": literal_sum / triples,
        "threshold_violation_rate": threshold_count / triples,
        "threshold_conditional_mean": threshold_sum / max(threshold_count, 1),
        "threshold_unconditional_mean": threshold_sum / triples,
        "max_gap": float(values["max_gap"]),
    }


def main() -> None:
    """Analyze layer kernels stored in an NPZ file or defined by an ALiBi config."""
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--profiles", type=Path)
    source.add_argument("--rollout-config", type=Path)
    parser.add_argument(
        "--kernel-key",
        default="mean_forward_alti_attention",
        help="NPZ key for the unmixed transport kernels A^(t)",
    )
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.profiles is not None:
        with np.load(args.profiles) as archive:
            if args.kernel_key not in archive:
                available = ", ".join(archive.files)
                raise KeyError(f"{args.kernel_key!r} not found; available keys: {available}")
            kernels = archive[args.kernel_key]
    else:
        from src.rollout.main import ModelConfig, compute_model_results
        from src.util.pydantic import load_config

        config = load_config(args.rollout_config, ModelConfig)
        kernels = np.stack(compute_model_results(config).attention_scores_list)
    if kernels.ndim != 3 or kernels.shape[-1] != kernels.shape[-2]:
        raise ValueError("Expected kernels with shape [layers, positions, positions]")

    raw_rows = [_measure_kernel(kernel, args.threshold) for kernel in kernels]
    totals = {
        name: sum(float(row[name]) for row in raw_rows)
        for name in (
            "triple_count",
            "literal_count",
            "literal_sum",
            "threshold_count",
            "threshold_sum",
        )
    }
    totals["max_gap"] = max(float(row["max_gap"]) for row in raw_rows)
    rows = [
        _finalize(row, layer=layer_index)
        for layer_index, row in enumerate(raw_rows)
    ]
    rows.append(_finalize(totals, layer="all"))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"threshold={args.threshold:g}")
    for name, value in rows[-1].items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
