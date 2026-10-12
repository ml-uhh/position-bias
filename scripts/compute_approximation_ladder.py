"""Compute corpus-rollout approximation errors from saved profile bundles."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / values.sum()


def _w1(left: np.ndarray, right: np.ndarray) -> float:
    positions = np.arange(left.size, dtype=np.float64)
    return float(
        wasserstein_distance(
            positions,
            positions,
            u_weights=_normalize(left),
            v_weights=_normalize(right),
        )
        / (left.size - 1)
    )


def compute(result_dir: Path) -> list[dict[str, str | float]]:
    with (result_dir / "forward_source_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        metric_rows = list(csv.DictReader(stream))

    aggregations = list(dict.fromkeys(row["aggregation"] for row in metric_rows))
    if len(aggregations) < 3:
        raise ValueError(f"{result_dir}: expected three aggregation levels")
    variants = {
        aggregation: [
            row["variant"] for row in metric_rows if row["aggregation"] == aggregation
        ]
        for aggregation in aggregations
    }
    common = set(variants[aggregations[0]])
    for aggregation in aggregations[1:]:
        common &= set(variants[aggregation])

    bundle = np.load(result_dir / "forward_source_profiles.npz")
    profiles: dict[tuple[str, str], np.ndarray] = {}
    for aggregation_index, aggregation in enumerate(aggregations):
        for variant_index, variant in enumerate(variants[aggregation]):
            profiles[aggregation, variant] = bundle[
                f"profile_{aggregation_index}_{variant_index}"
            ]

    joint = "Product of corpus-mean kernels"
    input_average = "Mean of per-sample rollouts"
    mean_field = "Separate means"
    required = {joint, input_average, mean_field}
    if not required.issubset(aggregations):
        missing = ", ".join(sorted(required.difference(aggregations)))
        raise ValueError(f"{result_dir}: missing aggregation levels: {missing}")

    rows: list[dict[str, str | float]] = []
    pairs = (
        (input_average, joint, "cross_layer"),
        (joint, mean_field, "within_layer"),
        (input_average, mean_field, "combined"),
    )
    for variant in variants[mean_field]:
        if variant not in common:
            continue
        row: dict[str, str | float] = {"variant": variant}
        for left, right, label in pairs:
            row[label] = _w1(profiles[left, variant], profiles[right, variant])
        rows.append(row)

    output = result_dir / "approximation_ladder_metrics.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    for result_dir in args.result_dirs:
        rows = compute(result_dir)
        for row in rows:
            print(result_dir, row, sep="\t")


if __name__ == "__main__":
    main()
