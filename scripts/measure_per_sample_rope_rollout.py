"""Measure average complete RoPE rollouts before corpus averaging."""

from __future__ import annotations

import argparse
import csv
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
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from src.main import load_model, setup_dataset
from src.models.qwen2 import Qwen2AttentionWrapper
from src.rollout.lambda_schedule import make_lambda_schedule
from src.rollout.main import ModelConfig
from src.util.configs import EvalConfig
from src.util.pydantic import load_config


def _normalize(values: np.ndarray) -> np.ndarray:
    profile = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(profile)) or np.any(profile < 0):
        raise ValueError("Profile must be finite and non-negative")
    total = profile.sum()
    if total <= 0:
        raise ValueError("Profile must have positive mass")
    return profile / total


def _load_measured_profile(path: Path, layer: int) -> np.ndarray:
    if path.suffix == ".npy":
        values = np.load(path)
    else:
        loaded = torch.load(path, map_location="cpu")
        values = loaded.detach().float().numpy()
    if values.ndim == 2:
        values = values[layer]
    if values.ndim != 1:
        raise ValueError(f"Expected 1D or layer-by-position profile, got {values.shape}")
    return _normalize(values)


def _rollout_last_row(
    attention: torch.Tensor,
    lambdas: torch.Tensor,
) -> torch.Tensor:
    """Return e_n^T R_T ... R_1 for a batch without forming full products."""
    batch_size, num_layers, sequence_length, _ = attention.shape
    profile = torch.zeros(
        batch_size,
        sequence_length,
        device=attention.device,
        dtype=torch.float32,
    )
    profile[:, -1] = 1.0
    for layer_idx in range(num_layers - 1, -1, -1):
        weight = lambdas[:, layer_idx].unsqueeze(1)
        attended = torch.bmm(
            profile.unsqueeze(1),
            attention[:, layer_idx],
        ).squeeze(1)
        profile = (1.0 - weight) * profile + weight * attended
    return profile


def _register_lambda_hooks(
    model: Qwen2ForCausalLM,
) -> tuple[list[torch.utils.hooks.RemovableHandle], list[torch.Tensor | None]]:
    """Capture one effective attention strength per sample and layer."""
    attention_outputs: dict[int, torch.Tensor] = {}
    lambda_values: list[torch.Tensor | None] = [None] * len(model.model.layers)
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer_idx, layer in enumerate(model.model.layers):
        def capture_attention(
            _module: nn.Module,
            _args: tuple[Any, ...],
            output: Any,
            *,
            index: int = layer_idx,
        ) -> None:
            attention_outputs[index] = output[0] if isinstance(output, tuple) else output

        def capture_lambda(
            _module: nn.Module,
            args: tuple[Any, ...],
            _output: Any,
            *,
            index: int = layer_idx,
        ) -> None:
            residual = args[0].float()
            attention_output = attention_outputs.pop(index).float()
            attention_norm = torch.linalg.vector_norm(attention_output, dim=-1)
            residual_norm = torch.linalg.vector_norm(residual, dim=-1).to(
                attention_norm.device,
            )
            strengths = attention_norm / (residual_norm + attention_norm).clamp_min(1e-10)
            lambda_values[index] = strengths.mean(dim=1).detach().cpu()

        handles.append(layer.self_attn.register_forward_hook(capture_attention))
        handles.append(layer.register_forward_hook(capture_lambda))

    return handles, lambda_values


def _profile_metrics(measured: np.ndarray, profile: np.ndarray) -> dict[str, float]:
    predicted = _normalize(profile)
    positions = np.arange(measured.size, dtype=np.float64)
    distance = wasserstein_distance(
        positions,
        positions,
        u_weights=measured,
        v_weights=predicted,
    )
    return {
        "wasserstein_tokens": float(distance),
        "wasserstein_normalized": float(distance / (measured.size - 1)),
        "token0_mass": float(predicted[0]),
        "last_token_mass": float(predicted[-1]),
    }


def _save_results(
    output_dir: Path,
    measured: np.ndarray,
    profiles: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    mean_sample_lambdas: np.ndarray,
    configured_lambdas: np.ndarray,
    num_samples: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "per_sample_rollout_profiles.npz",
        measured=measured,
        configured_lambdas=configured_lambdas,
        mean_sample_lambdas=mean_sample_lambdas,
        **{f"profile_{index}": value for index, value in enumerate(profiles.values())},
    )

    summary = {
        "num_samples": num_samples,
        "mean_sample_lambdas": mean_sample_lambdas.tolist(),
        "configured_lambdas": configured_lambdas.tolist(),
        "profiles": metrics,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["variant", *next(iter(metrics.values())).keys()]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for name, values in metrics.items():
            writer.writerow({"variant": name, **values})

    with (output_dir / "profiles.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["token_position", "measured", *profiles.keys()])
        for index in range(measured.size):
            writer.writerow(
                [index, measured[index], *(profile[index] for profile in profiles.values())],
            )

    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=200)
    positions = np.arange(measured.size)
    ax.plot(positions, measured, color="black", linewidth=2.2, label="Measured output influence")
    styles = [
        ("#D55E00", "-"),
        ("#CC79A7", "--"),
        ("#0072B2", "-"),
        ("#009E73", "--"),
    ]
    for (name, profile), (color, linestyle) in zip(
        profiles.items(),
        styles,
        strict=True,
    ):
        w1 = metrics[name]["wasserstein_normalized"]
        ax.plot(
            positions,
            profile,
            color=color,
            linestyle=linestyle,
            linewidth=1.65,
            label=f"{name} (W1={w1:.3f})",
        )
    ax.set_xlabel("Token position")
    ax.set_ylabel("Normalized influence")
    ax.set_xlim(0, measured.size - 1)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "per_sample_rollout_comparison.png", bbox_inches="tight")
    fig.savefig(output_dir / "per_sample_rollout_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--rollout-config", type=Path, required=True)
    parser.add_argument("--measured-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--num-batches", type=int)
    args = parser.parse_args()

    eval_config = load_config(args.eval_config, EvalConfig)
    rollout_config = load_config(args.rollout_config, ModelConfig)
    if not eval_config.enable_qk_stats:
        raise ValueError("Evaluation config must set enable_qk_stats=true")
    if eval_config.sequence_length != rollout_config.sequence_length:
        raise ValueError("Evaluation and rollout sequence lengths differ")

    model, tokenizer = load_model(eval_config)
    if not isinstance(model, Qwen2ForCausalLM):
        raise TypeError(f"Expected Qwen2ForCausalLM, got {type(model)}")
    if len(model.model.layers) != rollout_config.num_layers:
        raise ValueError("Evaluation model and rollout layer counts differ")
    dataloader = setup_dataset(eval_config, tokenizer)

    configured_lambdas_np = make_lambda_schedule(
        rollout_config.lambda_schedule_config,
        rollout_config.num_layers,
    ).astype(np.float64)
    configured_lambdas = torch.tensor(
        configured_lambdas_np,
        device=model.device,
        dtype=torch.float32,
    )
    num_batches = args.num_batches or eval_config.num_batches
    sequence_length = eval_config.sequence_length
    num_layers = rollout_config.num_layers

    sum_attention = torch.zeros(
        num_layers,
        sequence_length,
        sequence_length,
        dtype=torch.float64,
    )
    sum_joint_attention = torch.zeros_like(sum_attention)
    sum_lambdas = torch.zeros(num_layers, dtype=torch.float64)
    sum_fixed_profiles = torch.zeros(sequence_length, dtype=torch.float64)
    sum_sample_profiles = torch.zeros(sequence_length, dtype=torch.float64)
    sample_count = 0

    handles, captured_lambdas = _register_lambda_hooks(model)
    try:
        batches = islice(dataloader, num_batches)
        for inputs in tqdm(batches, total=num_batches):
            with torch.inference_mode():
                model(
                    input_ids=inputs["input_ids"].to(model.device),
                    attention_mask=inputs["attention_mask"].to(model.device),
                    use_cache=False,
                )

            layer_attention = []
            for layer in model.model.layers:
                wrapper = layer.self_attn
                if not isinstance(wrapper, Qwen2AttentionWrapper):
                    raise TypeError(f"Expected Qwen2AttentionWrapper, got {type(wrapper)}")
                if wrapper.last_attention_probs is None:
                    raise RuntimeError("Current-batch attention probabilities were not captured")
                layer_attention.append(wrapper.last_attention_probs)
            attention_cpu = torch.stack(layer_attention, dim=1).float()

            if any(value is None for value in captured_lambdas):
                raise RuntimeError("Not all per-sample lambdas were captured")
            sample_lambdas = torch.stack(
                [value for value in captured_lambdas if value is not None],
                dim=1,
            ).float()
            batch_size = attention_cpu.shape[0]

            attention_device = attention_cpu.to(model.device)
            sample_lambdas_device = sample_lambdas.to(model.device)
            fixed_batch_lambdas = configured_lambdas.unsqueeze(0).expand(batch_size, -1)
            fixed_profiles = _rollout_last_row(attention_device, fixed_batch_lambdas)
            sample_profiles = _rollout_last_row(
                attention_device,
                sample_lambdas_device,
            )

            sum_attention += attention_cpu.sum(dim=0).to(dtype=torch.float64)
            sum_joint_attention += (
                attention_cpu * sample_lambdas.detach().cpu()[:, :, None, None]
            ).sum(dim=0).to(dtype=torch.float64)
            sum_lambdas += sample_lambdas.detach().cpu().sum(dim=0).to(dtype=torch.float64)
            sum_fixed_profiles += fixed_profiles.detach().cpu().sum(dim=0).to(dtype=torch.float64)
            sum_sample_profiles += sample_profiles.detach().cpu().sum(dim=0).to(dtype=torch.float64)
            sample_count += batch_size
    finally:
        for handle in handles:
            handle.remove()

    if sample_count == 0:
        raise RuntimeError("No samples were evaluated")

    mean_attention = (sum_attention / sample_count).float().to(model.device)
    mean_sample_lambdas = (sum_lambdas / sample_count).float().to(model.device)
    mean_joint_attention = (sum_joint_attention / sample_count).float().to(model.device)

    product_mean_fixed = _rollout_last_row(
        mean_attention.unsqueeze(0),
        configured_lambdas.unsqueeze(0),
    )[0]
    identity_coefficients = 1.0 - mean_sample_lambdas
    mean_joint_residual = (
        mean_joint_attention
        + identity_coefficients[:, None, None]
        * torch.eye(sequence_length, device=model.device).unsqueeze(0)
    )
    product_mean_joint = _rollout_last_row(
        mean_joint_residual.unsqueeze(0),
        torch.ones(1, num_layers, device=model.device),
    )[0]

    profiles = {
        "Product of mean A, fixed lambda": _normalize(product_mean_fixed.cpu().numpy()),
        "Product of mean joint R": _normalize(product_mean_joint.cpu().numpy()),
        "Mean sample rollout, fixed lambda": _normalize(
            (sum_fixed_profiles / sample_count).numpy(),
        ),
        "Mean sample rollout, sample lambda": _normalize(
            (sum_sample_profiles / sample_count).numpy(),
        ),
    }
    measured = _load_measured_profile(args.measured_profile, args.layer)
    metrics = {
        name: _profile_metrics(measured, profile)
        for name, profile in profiles.items()
    }
    _save_results(
        args.output_dir,
        measured,
        profiles,
        metrics,
        mean_sample_lambdas.cpu().numpy(),
        configured_lambdas_np,
        sample_count,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
