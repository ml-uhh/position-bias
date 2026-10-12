"""Render paper tables and figures from canonical result directories."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = ROOT / "paper_outputs"

VARIANT_LABELS = {
    "attention_only": "Wu (lambda=1)",
    "fixed_half": "Abnar (lambda=1/2)",
    "adaptive": "Ours (adaptive)",
}


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / values.sum()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def rollout_metrics(model: str, family: str, directory: Path) -> dict[str, float]:
    rows = read_csv(directory / ("metrics.csv" if family == "alibi" else "forward_source_metrics.csv"))
    if family == "alibi":
        aliases = {
            "attention_only": "attention_only",
            "fixed_half": "fixed_half",
            "adaptive": "three_norm_frobenius",
        }
        by_name = {row["variant"]: row for row in rows}
        return {
            key: float(by_name[value]["wasserstein_normalized"])
            for key, value in aliases.items()
        }
    target_aggregation = "Separate means"
    by_name = {
        row["variant"]: row
        for row in rows
        if row["aggregation"] == target_aggregation
    }
    aliases = {
        "attention_only": "Forward ALTI attention only",
        "fixed_half": "Forward ALTI attention kernel plus fixed-half lambda",
        "adaptive": "Forward ALTI attention kernel plus three-norm Frobenius lambda",
    }
    return {
        key: float(by_name[value]["wasserstein_normalized"])
        for key, value in aliases.items()
    }


def rollout_profiles(family: str, directory: Path) -> dict[str, np.ndarray]:
    filename = "alibi_three_norm_results.npz" if family == "alibi" else "forward_source_profiles.npz"
    with np.load(directory / filename) as data:
        if family == "alibi":
            return {
                "measured": normalize(data["measured"]),
                "attention_only": normalize(data["profile_attention_only"]),
                "fixed_half": normalize(data["profile_fixed_half"]),
                "adaptive": normalize(data["profile_three_norm_frobenius"]),
                "lambda": np.asarray(data["lambda_three_norm_frobenius"]),
                "lambda_ci95": np.asarray(data["lambda_ci95_three_norm_frobenius"]),
            }
        variants = [str(value) for value in data["variant_names"]]
        indices = {name: index for index, name in enumerate(variants)}
        aliases = {
            "adaptive": "Forward ALTI attention kernel plus three-norm Frobenius lambda",
            "fixed_half": "Forward ALTI attention kernel plus fixed-half lambda",
            "attention_only": "Forward ALTI attention only",
        }
        profiles = {
            key: normalize(data[f"profile_2_{indices[value]}"])
            for key, value in aliases.items()
        }
        profiles["measured"] = normalize(data["measured"])
        profiles["lambda"] = np.asarray(data["mean_three_norm_lambda"])
        profiles["lambda_ci95"] = np.asarray(data["three_norm_lambda_ci95"])
        return profiles


def write_table(stem: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    table_dir = OUTPUT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (table_dir / f"{stem}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [" & ".join(fields) + " \\\\", r"\midrule"]
    for row in rows:
        formatted = []
        for field in fields:
            value = row[field]
            formatted.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append(" & ".join(formatted) + " \\\\")
    (table_dir / f"{stem}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_rollout_table(manifest: dict[str, Any], length: int, datasets: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for model, spec in manifest["models"].items():
        for dataset in datasets:
            directory = RESULTS / dataset / str(length) / model / "rollout"
            try:
                metrics = rollout_metrics(model, spec["family"], directory)
            except (FileNotFoundError, KeyError):
                continue
            rows.append({"model": model, "dataset": dataset, **metrics})
    write_table(f"wasserstein_{length}", rows)


def render_blankout_table(manifest: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for model in manifest["models"]:
        path = RESULTS / "fineweb-edu" / "256" / model / "blankout" / "metrics.csv"
        if not path.exists():
            continue
        values = {row["profile"]: row for row in read_csv(path)}
        rows.append(
            {
                "model": model,
                "attention_only": float(values["attention_only"]["w1_to_blankout"]),
                "fixed_half": float(values["fixed_half"]["w1_to_blankout"]),
                "adaptive": float(values["adaptive"]["w1_to_blankout"]),
                "blankout_vs_gradient": float(
                    values["matched_probability_gradient"]["w1_to_blankout"]
                ),
            }
        )
    write_table("blankout", rows)


def render_approximation_table(manifest: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for model, spec in manifest["models"].items():
        for dataset in manifest["main"]["datasets"]:
            directory = RESULTS / dataset / "256" / model / "rollout"
            path = directory / (
                "approximation_metrics.csv"
                if spec["family"] == "alibi"
                else "approximation_ladder_metrics.csv"
            )
            if not path.exists():
                continue
            values = read_csv(path)
            if spec["family"] == "alibi":
                by_name = {row["comparison"]: row for row in values}
                row = {
                    "model": model,
                    "dataset": dataset,
                    "cross_layer": float(by_name["input_vs_joint"]["wasserstein_normalized"]),
                    "within_layer": float(by_name["joint_vs_mean_field"]["wasserstein_normalized"]),
                    "combined": float(by_name["input_vs_mean_field"]["wasserstein_normalized"]),
                }
            else:
                target = next(
                    item
                    for item in values
                    if "three-norm Frobenius" in item["variant"]
                )
                row = {
                    "model": model,
                    "dataset": dataset,
                    "cross_layer": float(target["cross_layer"]),
                    "within_layer": float(target["within_layer"]),
                    "combined": float(target["combined"]),
                }
            rows.append(row)
    write_table("approximation_errors", rows)


def render_model_table(manifest: dict[str, Any]) -> None:
    rows = [
        {
            "model": model,
            "encoding": spec["family"].upper() if spec["family"] == "alibi" else "RoPE",
            "parameters": spec["parameters"],
            "depth": spec["depth"],
        }
        for model, spec in manifest["models"].items()
    ]
    write_table("models", rows)


def render_mixing_table(manifest: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for model, spec in manifest["models"].items():
        main_dir = RESULTS / "fineweb-edu" / "256" / model / "rollout"
        try:
            baseline = rollout_metrics(model, spec["family"], main_dir)
        except (FileNotFoundError, KeyError):
            continue
        if spec["family"] == "alibi":
            metrics = {row["variant"]: row for row in read_csv(main_dir / "metrics.csv")}
            relative = float(metrics["original"]["wasserstein_normalized"])
        else:
            diagnostic = main_dir.parent / "rollout_diagnostic" / "forward_source_metrics.csv"
            if not diagnostic.exists():
                continue
            metrics = {
                row["variant"]: row
                for row in read_csv(diagnostic)
                if row["aggregation"] == "Separate means"
            }
            relative = float(
                metrics["Forward ALTI attention kernel plus measured lambda"]
                ["wasserstein_normalized"]
            )
        rows.append(
            {
                "model": model,
                "attention_only": baseline["attention_only"],
                "fixed_half": baseline["fixed_half"],
                "relative_norm": relative,
                "norm_shapley": baseline["adaptive"],
            }
        )
    write_table("mixing_ablation", rows)


def render_monotonicity_table(manifest: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for model in manifest["models"]:
        for dataset in manifest["main"]["datasets"]:
            root = RESULTS / dataset / "256" / model
            if manifest["models"][model]["family"] == "rope":
                candidates = list(
                    (root / "monotonicity").glob("samples-*/monotonicity.csv")
                )
                if candidates:
                    path = max(
                        candidates,
                        key=lambda item: int(
                            item.parent.name.removeprefix("samples-")
                        ),
                    )
                else:
                    path = root / "monotonicity" / "missing.csv"
            else:
                path = root / "monotonicity.csv"
            if not path.exists():
                continue
            aggregate = read_csv(path)[-1]
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "violation_rate": float(aggregate["threshold_violation_rate"]),
                    "conditional_shortfall": float(aggregate["threshold_conditional_mean"]),
                    "unconditional_shortfall": float(aggregate["threshold_unconditional_mean"]),
                }
            )
    write_table("monotonicity", rows)


def render_intervention_table(manifest: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for model in manifest["intervention"]["models"]:
        root = RESULTS / "intervention" / model
        path = root / (
            "beta_results_metrics.json"
            if manifest["models"][model]["family"] == "alibi"
            else "rope_beta_results_metrics.json"
        )
        if not path.exists():
            continue
        values = json.loads(path.read_text(encoding="utf-8"))
        for beta in (0.8, 1.2):
            key = str(beta)
            prefix_ci = values["ci_prefix"][key]
            recency_ci = values["ci_recency"][key]
            stability = values["stability"][key]
            rows.append(
                {
                    "model": model,
                    "beta": beta,
                    "predicted_prefix": float(values["delta_prefix_predicted"][key]),
                    "measured_prefix": float(values["delta_prefix_measured"][key]),
                    "prefix_ci_low": float(prefix_ci[0]),
                    "prefix_ci_high": float(prefix_ci[1]),
                    "predicted_recency": float(values["delta_recency_predicted"][key]),
                    "measured_recency": float(values["delta_recency_measured"][key]),
                    "recency_ci_low": float(recency_ci[0]),
                    "recency_ci_high": float(recency_ci[1]),
                    "kl": float(stability["mean_kl"]),
                    "top1": float(stability["top1_unchanged_fraction"]),
                }
            )
    write_table("intervention", rows)


def render_profiles_and_lambdas(manifest: dict[str, Any]) -> None:
    figure_root = OUTPUT / "figures"
    profile_root = figure_root / "profiles"
    profile_root.mkdir(parents=True, exist_ok=True)
    colors = {
        "attention_only": "#4477AA",
        "fixed_half": "#EE6677",
        "adaptive": "#228833",
        "measured": "#AA3377",
    }
    for dataset in manifest["main"]["datasets"]:
        schedules: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {
            "alibi": [],
            "rope": [],
        }
        for model, spec in manifest["models"].items():
            directory = RESULTS / dataset / "256" / model / "rollout"
            try:
                profiles = rollout_profiles(spec["family"], directory)
            except (FileNotFoundError, KeyError):
                continue
            fig, axis = plt.subplots(figsize=(4.4, 2.8))
            positions = np.arange(1, profiles["measured"].size + 1)
            for name in ("attention_only", "fixed_half", "adaptive", "measured"):
                axis.plot(positions, profiles[name], color=colors[name], label=name.replace("_", " "))
            axis.set_xlabel("Input-token position")
            axis.set_ylabel("Normalized influence")
            axis.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            stem = profile_root / f"{model}_{dataset}_256"
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
            fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
            schedules[spec["family"]].append(
                (model, profiles["lambda"], profiles["lambda_ci95"])
            )
        for family, entries in schedules.items():
            if not entries:
                continue
            fig, axis = plt.subplots(figsize=(5.2, 3.0))
            for model, values, ci in entries:
                depth = np.arange(1, values.size + 1)
                axis.plot(depth, values, label=model)
                axis.fill_between(depth, values - ci, values + ci, alpha=0.15)
            axis.set_xlabel("Layer")
            axis.set_ylabel("Mixing coefficient")
            axis.legend(frameon=False, fontsize=7, ncol=2)
            fig.tight_layout()
            stem = figure_root / f"lambda_{family}_{dataset}_256"
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
            fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
            plt.close(fig)


def render_context_profiles(manifest: dict[str, Any]) -> None:
    figure_root = OUTPUT / "figures"
    entries: dict[str, list[tuple[str, np.ndarray]]] = {"alibi": [], "rope": []}
    for model, spec in manifest["models"].items():
        path = (
            RESULTS
            / "wikipedia"
            / "2048"
            / model
            / "attribution"
            / "output_prob_input_grad_l2_mean.npy"
        )
        if path.exists():
            entries[spec["family"]].append((model, normalize(np.load(path))))
    for family, profiles in entries.items():
        if not profiles:
            continue
        fig, axes = plt.subplots(1, len(profiles), figsize=(3.0 * len(profiles), 2.5), squeeze=False)
        for axis, (model, profile) in zip(axes[0], profiles, strict=True):
            axis.plot(np.arange(1, profile.size + 1), profile, color="#AA3377")
            axis.set_title(model, fontsize=9)
            axis.set_xlabel("Position")
        axes[0][0].set_ylabel("Normalized influence")
        fig.tight_layout()
        stem = figure_root / f"gradient_influence_{family}_wikipedia_2048"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    manifest = json.loads((ROOT / "config" / "paper.json").read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    render_model_table(manifest)
    render_rollout_table(manifest, 256, manifest["main"]["datasets"])
    render_rollout_table(manifest, 2048, manifest["context"]["datasets"])
    render_blankout_table(manifest)
    render_approximation_table(manifest)
    render_mixing_table(manifest)
    render_monotonicity_table(manifest)
    render_intervention_table(manifest)
    render_profiles_and_lambdas(manifest)
    render_context_profiles(manifest)
    print(f"Rendered available tables and figures under {OUTPUT}")


if __name__ == "__main__":
    main()
