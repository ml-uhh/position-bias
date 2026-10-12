# ruff: noqa: INP001, T201
"""Compare the original and three-norm lambda definitions on ALiBi models."""

from __future__ import annotations

import argparse
import csv
import gc
from itertools import islice
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
import torch
from torch import nn
from tqdm import tqdm

from scripts.measure_per_sample_rope_rollout import _load_measured_profile, _normalize
from src.main import close_dataloader_iterator, load_model, setup_dataset
from src.rollout.main import ModelConfig, compute_model_results
from src.util.configs import EvalConfig
from src.util.pydantic import load_config


def _rollout_last_row(attention: torch.Tensor, lambdas: torch.Tensor) -> torch.Tensor:
    """Return the last row of the factorized residual-aware rollout."""
    num_layers, sequence_length, _ = attention.shape
    profile = torch.zeros(sequence_length, dtype=torch.float64)
    profile[-1] = 1.0
    for layer_idx in range(num_layers - 1, -1, -1):
        attended = profile @ attention[layer_idx].double()
        strength = lambdas[layer_idx].double()
        profile = (1.0 - strength) * profile + strength * attended
    return profile / profile.sum().clamp_min(1e-300)


def _rollout_last_rows(
    attention: torch.Tensor,
    lambdas: torch.Tensor,
) -> torch.Tensor:
    """Return one content-free rollout profile for each per-input schedule."""
    batch_size, num_layers = lambdas.shape
    sequence_length = attention.shape[-1]
    profiles = torch.zeros(batch_size, sequence_length, dtype=torch.float64)
    profiles[:, -1] = 1.0
    attention = attention.double()
    for layer_idx in range(num_layers - 1, -1, -1):
        attended = profiles @ attention[layer_idx]
        strength = lambdas[:, layer_idx].double().unsqueeze(1)
        profiles = (1.0 - strength) * profiles + strength * attended
    return profiles / profiles.sum(dim=1, keepdim=True).clamp_min(1e-300)


def _normalized_w1(left: np.ndarray, right: np.ndarray) -> float:
    positions = np.arange(left.size, dtype=np.float64)
    return float(
        wasserstein_distance(positions, positions, u_weights=left, v_weights=right)
        / (left.size - 1)
    )


def _lambda_values(
    residual: torch.Tensor,
    attention_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Frobenius relative norm and three norm-Shapley aggregations."""
    residual = residual.float()
    attention_output = attention_output.to(residual.device).float()
    combined = residual + attention_output

    residual_frobenius = torch.linalg.vector_norm(residual, dim=(-2, -1))
    attention_frobenius = torch.linalg.vector_norm(attention_output, dim=(-2, -1))
    combined_frobenius = torch.linalg.vector_norm(combined, dim=(-2, -1))

    residual_token_norm = torch.linalg.vector_norm(residual, dim=-1)
    attention_token_norm = torch.linalg.vector_norm(attention_output, dim=-1)
    combined_token_norm = torch.linalg.vector_norm(combined, dim=-1)

    original = (
        attention_frobenius
        / (residual_frobenius + attention_frobenius).clamp_min(1e-30)
    )
    token_three_norm = (
        combined_token_norm + attention_token_norm - residual_token_norm
    ) / (2.0 * combined_token_norm).clamp_min(1e-30)
    token_three_norm = torch.where(
        combined_token_norm > 1e-30,
        token_three_norm,
        torch.full_like(token_three_norm, 0.5),
    ).clamp(0.0, 1.0).mean(dim=-1)

    mean_residual_norm = residual_token_norm.mean(dim=-1)
    mean_attention_norm = attention_token_norm.mean(dim=-1)
    mean_combined_norm = combined_token_norm.mean(dim=-1)
    ratio_of_mean_norms = (
        mean_combined_norm + mean_attention_norm - mean_residual_norm
    ) / (2.0 * mean_combined_norm).clamp_min(1e-30)
    ratio_of_mean_norms = torch.where(
        mean_combined_norm > 1e-30,
        ratio_of_mean_norms,
        torch.full_like(ratio_of_mean_norms, 0.5),
    ).clamp(0.0, 1.0)

    sequence_three_norm = (
        combined_frobenius + attention_frobenius - residual_frobenius
    ) / (2.0 * combined_frobenius).clamp_min(1e-30)
    sequence_three_norm = torch.where(
        combined_frobenius > 1e-30,
        sequence_three_norm,
        torch.full_like(sequence_three_norm, 0.5),
    ).clamp(0.0, 1.0)
    return original, sequence_three_norm, token_three_norm, ratio_of_mean_norms


def _model_layout(model: nn.Module) -> tuple[list[nn.Module], list[nn.Module]]:
    """Return transformer blocks and modules producing attention updates."""
    model_type = model.config.model_type  # type: ignore[attr-defined]
    if model_type == "falcon":
        layers = list(model.transformer.h)  # type: ignore[attr-defined]
        return layers, [layer.self_attention for layer in layers]  # type: ignore[attr-defined]
    if model_type == "mpt":
        layers = list(model.transformer.blocks)  # type: ignore[attr-defined]
        return layers, [layer.attn for layer in layers]  # type: ignore[attr-defined]
    if model_type == "bloom":
        layers = list(model.transformer.h)  # type: ignore[attr-defined]
        return layers, [layer.self_attention.dense for layer in layers]  # type: ignore[attr-defined]
    raise TypeError(f"Unsupported ALiBi model type: {model_type}")


def _register_hooks(
    layers: list[nn.Module],
    attention_modules: list[nn.Module],
) -> tuple[
    list[torch.utils.hooks.RemovableHandle],
    list[torch.Tensor | None],
    list[torch.Tensor | None],
]:
    residuals: list[torch.Tensor | None] = [None] * len(layers)
    attention_outputs: list[torch.Tensor | None] = [None] * len(layers)
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer_idx, (layer, attention_module) in enumerate(
        zip(layers, attention_modules, strict=True),
    ):
        def capture_residual(
            _module: nn.Module,
            args: tuple[Any, ...],
            *,
            index: int = layer_idx,
        ) -> None:
            residuals[index] = args[0]

        def capture_attention(
            _module: nn.Module,
            _args: tuple[Any, ...],
            output: Any,
            *,
            index: int = layer_idx,
        ) -> None:
            attention_outputs[index] = output[0] if isinstance(output, tuple) else output

        handles.append(layer.register_forward_pre_hook(capture_residual))
        handles.append(attention_module.register_forward_hook(capture_attention))
    return handles, residuals, attention_outputs


def _load_attention(path: Path, sequence_length: int) -> torch.Tensor:
    loaded = torch.load(path, map_location="cpu").float()
    if loaded.ndim == 4:
        loaded = loaded.mean(dim=1)
    if loaded.ndim != 3 or loaded.shape[-2:] != (sequence_length, sequence_length):
        raise ValueError(f"Unexpected attention-kernel shape: {tuple(loaded.shape)}")
    return loaded.double()


def _load_rollout_attention(path: Path, sequence_length: int) -> torch.Tensor:
    rollout_config = load_config(path, ModelConfig)
    if rollout_config.sequence_length != sequence_length:
        raise ValueError("Evaluation and rollout sequence lengths differ")
    results = compute_model_results(rollout_config)
    return torch.from_numpy(np.stack(results.attention_scores_list)).double()


def _metrics(measured: np.ndarray, profile: np.ndarray) -> dict[str, float]:
    positions = np.arange(measured.size, dtype=np.float64)
    distance = wasserstein_distance(
        positions,
        positions,
        u_weights=measured,
        v_weights=profile,
    )
    return {
        "wasserstein_tokens": float(distance),
        "wasserstein_normalized": float(distance / (measured.size - 1)),
        "token0_mass": float(profile[0]),
        "last_token_mass": float(profile[-1]),
    }


def _save_results(
    output_dir: Path,
    measured: np.ndarray,
    schedules: dict[str, np.ndarray],
    profiles: dict[str, np.ndarray],
    confidence_intervals: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    sample_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "alibi_three_norm_results.npz",
        measured=measured,
        **{f"lambda_{name}": value for name, value in schedules.items()},
        **{
            f"lambda_ci95_{name}": value
            for name, value in confidence_intervals.items()
        },
        **{f"profile_{name}": value for name, value in profiles.items()},
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "num_samples": sample_count,
                "metrics": metrics,
                "lambda_schedules": {
                    name: values.tolist() for name, values in schedules.items()
                },
            },
            stream,
            indent=2,
        )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["variant", *next(iter(metrics.values())).keys()],
        )
        writer.writeheader()
        for name, values in metrics.items():
            writer.writerow({"variant": name, **values})

    positions = np.arange(measured.size)
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=200)
    ax.semilogy(positions, measured, color="black", linewidth=2.2, label="Measured")
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for (name, profile), color in zip(profiles.items(), colors, strict=False):
        w1 = metrics[name]["wasserstein_normalized"]
        ax.semilogy(
            positions,
            np.maximum(profile, 1e-12),
            color=color,
            linewidth=1.65,
            label=f"{name} (W1={w1:.3f})",
        )
    ax.set(xlabel="Token position", ylabel="Normalized influence", xlim=(0, measured.size - 1))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "profile_comparison.png", bbox_inches="tight")
    fig.savefig(output_dir / "profile_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    layers = np.arange(1, next(iter(schedules.values())).size + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    for (name, schedule), color in zip(schedules.items(), colors, strict=False):
        ci95 = confidence_intervals[name]
        ax.plot(layers, schedule, color=color, linewidth=1.7, label=name)
        ax.fill_between(layers, schedule - ci95, schedule + ci95, color=color, alpha=0.14)
    ax.set(xlabel="Layer", ylabel=r"Effective attention mixing $\lambda_t$")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "lambda_schedules.png", bbox_inches="tight")
    fig.savefig(output_dir / "lambda_schedules.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-config", type=Path, required=True)
    attention_source = parser.add_mutually_exclusive_group(required=True)
    attention_source.add_argument("--attention-kernel", type=Path)
    attention_source.add_argument("--rollout-config", type=Path)
    parser.add_argument(
        "--measured-profile",
        type=Path,
        help=(
            "Measured influence profile used for rollout comparisons. Omit this "
            "for approximation-only runs."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-batches", type=int)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument(
        "--model-id",
        help="Override the configured model ID, for example with a cached snapshot path.",
    )
    args = parser.parse_args()

    eval_config = load_config(args.eval_config, EvalConfig)
    if args.model_id is not None:
        if eval_config.hf_model_config is None:
            raise ValueError("The evaluation config has no Hugging Face model config")
        eval_config = eval_config.model_copy(
            update={
                "hf_model_config": eval_config.hf_model_config.model_copy(
                    update={"model_id": args.model_id},
                ),
            },
        )
    # Attention is loaded from the paper's saved corpus mean, so wrapping would only
    # duplicate large statistics buffers during the lambda-only forward pass.
    eval_config = eval_config.model_copy(
        update={"enable_qk_stats": False, "num_workers": 0},
    )
    model, tokenizer = load_model(eval_config)
    dataloader = setup_dataset(eval_config, tokenizer)
    layers, attention_modules = _model_layout(model)
    if args.attention_kernel is not None:
        attention = _load_attention(args.attention_kernel, eval_config.sequence_length)
    else:
        attention = _load_rollout_attention(
            args.rollout_config,
            eval_config.sequence_length,
        )
    if attention.shape[0] != len(layers):
        raise ValueError(
            f"Attention has {attention.shape[0]} layers, model has {len(layers)}",
        )

    names = (
        "original",
        "three_norm_frobenius",
        "three_norm_token_mean",
        "three_norm_ratio_of_token_mean_norms",
    )
    sums = {name: torch.zeros(len(layers), dtype=torch.float64) for name in names}
    sum_squares = {name: torch.zeros(len(layers), dtype=torch.float64) for name in names}
    input_rollout_sum = torch.zeros(
        eval_config.sequence_length,
        dtype=torch.float64,
    )
    sample_count = 0
    handles, residuals, attention_outputs = _register_hooks(layers, attention_modules)
    num_batches = args.num_batches or eval_config.num_batches
    data_iterator = iter(dataloader)
    try:
        for inputs in tqdm(islice(data_iterator, num_batches), total=num_batches):
            with torch.inference_mode():
                model(
                    input_ids=inputs["input_ids"].to(model.device),
                    attention_mask=inputs["attention_mask"].to(model.device),
                    use_cache=False,
                )
            if any(value is None for value in residuals + attention_outputs):
                raise RuntimeError("Not all layer activations were captured")
            batch_values = [[], [], [], []]
            for residual, attention_output in zip(
                residuals,
                attention_outputs,
                strict=True,
            ):
                assert residual is not None and attention_output is not None
                for values, value in zip(
                    batch_values,
                    _lambda_values(residual, attention_output),
                    strict=True,
                ):
                    values.append(value.detach().cpu().double())
            batch_size = batch_values[0][0].shape[0]
            for name, values in zip(names, batch_values, strict=True):
                stacked = torch.stack(values, dim=1)
                sums[name] += stacked.sum(dim=0)
                sum_squares[name] += stacked.square().sum(dim=0)
                if name == "three_norm_frobenius":
                    input_rollout_sum += _rollout_last_rows(
                        attention,
                        stacked,
                    ).sum(dim=0)
            sample_count += batch_size
            residuals[:] = [None] * len(layers)
            attention_outputs[:] = [None] * len(layers)
    finally:
        for handle in handles:
            handle.remove()
        close_dataloader_iterator(data_iterator)
        del data_iterator
        del dataloader
        gc.collect()

    if sample_count == 0:
        raise RuntimeError("No samples were evaluated")
    schedules = {name: (sums[name] / sample_count).numpy() for name in names}
    confidence_intervals = {}
    for name in names:
        mean = sums[name] / sample_count
        variance = (sum_squares[name] / sample_count - mean.square()).clamp_min(0.0)
        confidence_intervals[name] = (1.96 * torch.sqrt(variance / sample_count)).numpy()

    if args.measured_profile is None:
        input_profile = _normalize((input_rollout_sum / sample_count).numpy())
        joint_profile = _normalize(
            _rollout_last_row(
                attention,
                torch.from_numpy(schedules["three_norm_frobenius"]),
            ).numpy(),
        )
        approximation_rows = [
            {
                "comparison": "input_vs_joint",
                "wasserstein_normalized": _normalized_w1(
                    input_profile,
                    joint_profile,
                ),
            },
            {
                "comparison": "joint_vs_mean_field",
                "wasserstein_normalized": 0.0,
            },
            {
                "comparison": "input_vs_mean_field",
                "wasserstein_normalized": _normalized_w1(
                    input_profile,
                    joint_profile,
                ),
            },
        ]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output_dir / "approximation_results.npz",
            sample_count=sample_count,
            input_profile=input_profile,
            joint_profile=joint_profile,
            **{f"lambda_{name}": value for name, value in schedules.items()},
            **{
                f"lambda_ci95_{name}": value
                for name, value in confidence_intervals.items()
            },
        )
        with (args.output_dir / "approximation_metrics.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["comparison", "wasserstein_normalized"],
            )
            writer.writeheader()
            writer.writerows(approximation_rows)
        with (args.output_dir / "approximation_summary.json").open(
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(
                {
                    "num_samples": sample_count,
                    "metrics": {
                        row["comparison"]: row["wasserstein_normalized"]
                        for row in approximation_rows
                    },
                },
                stream,
                indent=2,
            )
        for row in approximation_rows:
            print(
                f"{row['comparison']}: "
                f"W1={row['wasserstein_normalized']:.9f}",
            )
        return

    measured = _load_measured_profile(args.measured_profile, args.layer)
    reference_schedule = torch.from_numpy(next(iter(schedules.values())))
    profiles = {
        "attention_only": _normalize(
            _rollout_last_row(attention, torch.ones_like(reference_schedule)).numpy(),
        ),
        "fixed_half": _normalize(
            _rollout_last_row(
                attention,
                torch.full_like(reference_schedule, 0.5),
            ).numpy(),
        ),
        **{
            name: _normalize(
                _rollout_last_row(attention, torch.from_numpy(schedule)).numpy(),
            )
            for name, schedule in schedules.items()
        },
    }
    profiles["three_norm_frobenius_mean_input_rollout"] = _normalize(
        (input_rollout_sum / sample_count).numpy(),
    )
    metrics = {name: _metrics(measured, profile) for name, profile in profiles.items()}
    _save_results(
        args.output_dir,
        measured,
        schedules,
        profiles,
        confidence_intervals,
        metrics,
        sample_count,
    )
    input_profile = profiles["three_norm_frobenius_mean_input_rollout"]
    joint_profile = profiles["three_norm_frobenius"]
    approximation_rows = [
        {
            "comparison": "input_vs_joint",
            "wasserstein_normalized": _normalized_w1(input_profile, joint_profile),
        },
        {
            "comparison": "joint_vs_mean_field",
            "wasserstein_normalized": 0.0,
        },
        {
            "comparison": "input_vs_mean_field",
            "wasserstein_normalized": _normalized_w1(input_profile, joint_profile),
        },
    ]
    with (args.output_dir / "approximation_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["comparison", "wasserstein_normalized"],
        )
        writer.writeheader()
        writer.writerows(approximation_rows)
    for name, values in metrics.items():
        print(f"{name}: W1={values['wasserstein_normalized']:.6f}")


if __name__ == "__main__":
    main()
