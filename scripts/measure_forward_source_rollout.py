# ruff: noqa: INP001, T201
"""Measure forward-only source-vector rollouts for causal RoPE models."""

from __future__ import annotations

import argparse
import csv
import os
import signal
from pathlib import Path
from typing import Any, Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.measure_per_sample_rope_rollout import _load_measured_profile, _normalize
from src.main import setup_dataset
from src.models.source_contributions import SourceContributionAttention
from src.models.wrap import wrap_attention_layers
from src.util.configs import EvalConfig
from src.util.pydantic import load_config


class ForwardSourceAttention(SourceContributionAttention, Protocol):
    """Attention-wrapper interface needed by this experiment."""

    last_attention_probs: torch.Tensor | None

    def clear_source_projection_state(self) -> None:
        """Release source vectors captured for the current sample."""
        ...

    def value_source_removal_scores(
        self,
        decompositions: list[tuple[torch.Tensor | None, torch.Tensor]],
        query_chunk_size: int = 16,
    ) -> list[torch.Tensor]:
        """Calculate forward removal scores for each decomposition."""
        ...


HookOutput = torch.Tensor | tuple[torch.Tensor, ...]
_NORMALIZATION_EPSILON = 1e-30

_DIAGNOSTIC_NAMES = [
    "Standard attention plus measured lambda",
    "Standard attention plus three-norm Frobenius lambda",
    "Standard attention plus three-norm token-mean lambda",
    "Standard attention plus three-norm ratio-of-mean-norms lambda",
    "Forward ALTI attention kernel plus measured lambda",
    "Forward ALTI attention kernel plus three-norm Frobenius lambda",
    "Forward ALTI attention kernel plus three-norm token-mean lambda",
    "Forward ALTI attention kernel plus three-norm ratio-of-mean-norms lambda",
    "Forward ALTI post-attention",
    "Forward ALTI full block",
    "Standard attention only",
    "Forward ALTI attention only",
]
_PAPER_NAMES = [
    "Forward ALTI attention kernel plus three-norm Frobenius lambda",
    "Forward ALTI attention kernel plus fixed-half lambda",
    "Forward ALTI attention only",
]


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a restart checkpoint without exposing a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_input_cache(
    batches: list[dict[str, torch.Tensor]],
    path: Path,
    signature: dict[str, Any],
) -> None:
    """Persist tokenized inputs so subsequent passes do not tokenize raw documents."""
    _atomic_torch_save(
        {
            "signature": signature,
            "batches": [
                {key: value.detach().cpu() for key, value in batch.items()}
                for batch in batches
            ],
        },
        path,
    )


def _load_input_cache(
    path: Path,
    signature: dict[str, Any],
) -> list[dict[str, torch.Tensor]]:
    """Load and validate a tokenized-input cache."""
    if not path.exists():
        return []
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("signature") != signature:
        raise ValueError(f"Input cache does not match this run: {path}")
    return saved["batches"]


def _close_data_iterator(iterator: Any) -> None:
    """Close a partially consumed streaming DataLoader iterator when possible."""
    if iterator is None:
        return
    fetcher = getattr(iterator, "_dataset_fetcher", None)
    dataset_iterator = getattr(fetcher, "dataset_iter", None)
    close = getattr(dataset_iterator, "close", None)
    if callable(close):
        close()
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def _load_eager_model(eval_config: EvalConfig) -> tuple[nn.Module, Any]:
    model_config = eval_config.hf_model_config
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_id,
        padding_side="left",
        cache_dir=model_config.cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_id,
        cache_dir=model_config.cache_dir,
        device_map=model_config.device_map or "auto",
        max_memory=model_config.max_memory,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.requires_grad_(requires_grad=False)
    model.config.use_cache = False
    wrap_attention_layers(model, eval_config.sequence_length)
    model.eval()
    return model, tokenizer


def _row_normalize_with_identity(scores: torch.Tensor) -> torch.Tensor:
    denominator = scores.sum(dim=-1, keepdim=True)
    normalized = scores / denominator.clamp_min(_NORMALIZATION_EPSILON)
    zero_rows = denominator.squeeze(-1) <= _NORMALIZATION_EPSILON
    if torch.any(zero_rows):
        batch_indices, query_indices = zero_rows.nonzero(as_tuple=True)
        normalized[batch_indices, query_indices] = 0.0
        normalized[batch_indices, query_indices, query_indices] = 1.0
    return normalized


def _rollout(kernels: torch.Tensor) -> torch.Tensor:
    batch_size, _, sequence_length, _ = kernels.shape
    profile = torch.zeros(
        batch_size,
        sequence_length,
        device=kernels.device,
        dtype=torch.float64,
    )
    profile[:, -1] = 1.0
    for layer_idx in range(kernels.shape[1] - 1, -1, -1):
        profile = torch.bmm(
            profile.unsqueeze(1),
            kernels[:, layer_idx].double(),
        ).squeeze(1)
    return profile / profile.sum(dim=-1, keepdim=True).clamp_min(1e-300)


def _mix_raw_attention(
    attention: torch.Tensor,
    lambdas: torch.Tensor,
) -> torch.Tensor:
    sequence_length = attention.shape[-1]
    identity = torch.eye(
        sequence_length,
        device=attention.device,
        dtype=attention.dtype,
    )[None, None]
    return (1.0 - lambdas[:, :, None, None]) * identity + lambdas[
        :,
        :,
        None,
        None,
    ] * attention


def _rollout_mixed_attention(
    attention: torch.Tensor,
    lambdas: torch.Tensor,
) -> torch.Tensor:
    """Roll out residual mixing without materializing full mixed kernels."""
    batch_size, _, sequence_length, _ = attention.shape
    profile = torch.zeros(
        batch_size,
        sequence_length,
        device=attention.device,
        dtype=torch.float64,
    )
    profile[:, -1] = 1.0
    for layer_idx in range(attention.shape[1] - 1, -1, -1):
        layer_lambda = lambdas[:, layer_idx].double().unsqueeze(-1)
        transported = torch.bmm(
            profile.unsqueeze(1),
            attention[:, layer_idx].double(),
        ).squeeze(1)
        profile = (1.0 - layer_lambda) * profile + layer_lambda * transported
    return profile / profile.sum(dim=-1, keepdim=True).clamp_min(1e-300)


def _accumulate_mixed_kernel_sum(
    target: torch.Tensor,
    attention: torch.Tensor,
    lambdas: torch.Tensor,
) -> None:
    """Accumulate mixed kernels layerwise on CPU to bound peak GPU memory."""
    cpu_lambdas = lambdas.detach().cpu().double()
    diagonal = torch.arange(attention.shape[-1])
    for layer_idx in range(attention.shape[1]):
        layer_attention = attention[:, layer_idx].detach().cpu().double()
        layer_lambda = cpu_lambdas[:, layer_idx]
        target[layer_idx] += (
            layer_lambda[:, None, None] * layer_attention
        ).sum(dim=0)
        target[layer_idx, diagonal, diagonal] += (1.0 - layer_lambda).sum()


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


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run the forward source-contribution rollout experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--measured-profile", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-batches", type=int, default=1000)
    parser.add_argument("--skip-batches", type=int, default=0)
    parser.add_argument("--query-chunk-size", type=int, default=16)
    parser.add_argument(
        "--variant-set",
        choices=("paper", "diagnostic"),
        default="paper",
        help="Use the three paper variants or the full diagnostic suite.",
    )
    parser.add_argument(
        "--kernel-only",
        action="store_true",
        help=(
            "Only average the unmixed projection-aware transport kernels; "
            "skip lambda, rollout, and influence-profile calculations."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Save restart state after this many processed batches; 0 disables it.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Checkpoint path (default: OUTPUT_DIR/rollout_checkpoint.pt).",
    )
    parser.add_argument(
        "--input-cache",
        type=Path,
        help="Tokenized-input cache (default: OUTPUT_DIR/tokenized_inputs.pt).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing checkpoint and start from the requested offset.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=8,
        help="Maximum PyTorch CPU threads used by this process; 0 leaves it unchanged.",
    )
    args = parser.parse_args()

    if not args.kernel_only and args.measured_profile is None:
        parser.error("--measured-profile is required unless --kernel-only is set")
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be nonnegative")
    if args.cpu_threads < 0:
        parser.error("--cpu-threads must be nonnegative")
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    eval_config = load_config(args.eval_config, EvalConfig)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_path or (
        args.output_dir / "rollout_checkpoint.pt"
    )
    input_cache_path = args.input_cache or (args.output_dir / "tokenized_inputs.pt")
    model, tokenizer = _load_eager_model(eval_config)
    layers = model.model.layers  # type: ignore[attr-defined]
    num_layers = len(layers)
    sequence_length = eval_config.sequence_length
    run_signature = {
        "model_id": eval_config.hf_model_config.model_id,
        "dataset": eval_config.dataset_config.model_dump(mode="json"),
        "sequence_length": sequence_length,
        "skip_batches": args.skip_batches,
        "variant_set": args.variant_set,
        "kernel_only": args.kernel_only,
        "num_layers": num_layers,
    }
    input_signature = {
        "model_id": eval_config.hf_model_config.model_id,
        "dataset": eval_config.dataset_config.model_dump(mode="json"),
        "sequence_length": sequence_length,
        "skip_batches": args.skip_batches,
    }
    cached_inputs = _load_input_cache(input_cache_path, input_signature)

    residual_inputs: list[torch.Tensor | None] = [None] * num_layers
    attention_outputs: list[torch.Tensor | None] = [None] * num_layers
    post_attention: list[torch.Tensor | None] = [None] * num_layers
    attention_scores: list[torch.Tensor | None] = [None] * num_layers
    post_scores: list[torch.Tensor | None] = [None] * num_layers
    full_scores: list[torch.Tensor | None] = [None] * num_layers
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for layer_idx, layer in enumerate(layers):
        wrapper = cast("ForwardSourceAttention", layer.self_attn)
        required = (
            "capture_source_projection_state",
            "value_source_removal_scores",
            "clear_source_projection_state",
        )
        if not all(hasattr(wrapper, name) for name in required):
            raise TypeError(
                f"{type(wrapper).__name__} does not support forward source attribution",
            )
        wrapper.capture_source_projection_state = True

        def capture_residual(
            _module: nn.Module,
            hook_args: tuple[Any, ...],
            *,
            index: int = layer_idx,
        ) -> None:
            residual_inputs[index] = hook_args[0]

        def capture_attention_output(
            _module: nn.Module,
            _hook_args: tuple[Any, ...],
            output: HookOutput,
            *,
            index: int = layer_idx,
        ) -> None:
            attention_output = output[0] if isinstance(output, tuple) else output
            residual = residual_inputs[index]
            if residual is None:
                raise RuntimeError(f"Layer {index} residual was not captured")
            residual = residual.to(attention_output.device, dtype=torch.float32)
            residual_inputs[index] = residual.detach()
            attention_outputs[index] = attention_output.detach()
            post_attention[index] = (
                residual + attention_output.float()
            ).detach()
            if args.variant_set == "paper":
                wrapper = cast("ForwardSourceAttention", layers[index].self_attn)
                (attention_raw,) = wrapper.value_source_removal_scores(
                    [(None, attention_output.float())],
                    query_chunk_size=args.query_chunk_size,
                )
                attention_scores[index] = _row_normalize_with_identity(
                    attention_raw,
                ).detach()
                wrapper.clear_source_projection_state()

        def capture_block_output(
            _module: nn.Module,
            _hook_args: tuple[Any, ...],
            output: HookOutput,
            *,
            index: int = layer_idx,
        ) -> None:
            if args.variant_set == "paper":
                return
            block_output = output[0] if isinstance(output, tuple) else output
            residual = residual_inputs[index]
            attention_output = attention_outputs[index]
            post = post_attention[index]
            if residual is None or attention_output is None or post is None:
                raise RuntimeError(f"Layer {index} forward state was not captured")
            block_float = block_output.float()
            residual = residual.to(block_float.device)
            attention_output = attention_output.to(block_float.device)
            post = post.to(block_float.device)
            mlp_update = block_float - post
            wrapper = cast("ForwardSourceAttention", layers[index].self_attn)
            attention_raw, post_raw, full_raw = wrapper.value_source_removal_scores(
                [
                    (None, attention_output.float()),
                    (residual.float(), post),
                    (residual.float() + mlp_update, block_float),
                ],
                query_chunk_size=args.query_chunk_size,
            )
            attention_scores[index] = _row_normalize_with_identity(
                attention_raw,
            ).detach()
            post_scores[index] = _row_normalize_with_identity(post_raw).detach()
            full_scores[index] = _row_normalize_with_identity(full_raw).detach()
            wrapper.clear_source_projection_state()

        handles.append(layer.register_forward_pre_hook(capture_residual))
        handles.append(layer.self_attn.register_forward_hook(capture_attention_output))
        handles.append(layer.register_forward_hook(capture_block_output))

    names = _PAPER_NAMES if args.variant_set == "paper" else _DIAGNOSTIC_NAMES
    kernel_sums = {
        name: torch.zeros(
            num_layers,
            sequence_length,
            sequence_length,
            dtype=torch.float64,
        )
        for name in names
    }
    profile_sums = {
        name: torch.zeros(sequence_length, dtype=torch.float64) for name in names
    }
    base_kernel_sums = {
        "standard_attention": torch.zeros(
            num_layers,
            sequence_length,
            sequence_length,
            dtype=torch.float64,
        ),
        "forward_alti_attention": torch.zeros(
            num_layers,
            sequence_length,
            sequence_length,
            dtype=torch.float64,
        ),
    }
    lambda_sum = torch.zeros(num_layers, dtype=torch.float64)
    lambda_sq_sum = torch.zeros(num_layers, dtype=torch.float64)
    three_norm_lambda_sum = torch.zeros(num_layers, dtype=torch.float64)
    three_norm_lambda_sq_sum = torch.zeros(num_layers, dtype=torch.float64)
    token_three_norm_lambda_sum = torch.zeros(num_layers, dtype=torch.float64)
    token_three_norm_lambda_sq_sum = torch.zeros(num_layers, dtype=torch.float64)
    mean_norm_three_norm_lambda_sum = torch.zeros(num_layers, dtype=torch.float64)
    mean_norm_three_norm_lambda_sq_sum = torch.zeros(num_layers, dtype=torch.float64)
    sample_count = 0
    batches_processed = 0

    accumulator_names = (
        "kernel_sums",
        "profile_sums",
        "base_kernel_sums",
        "lambda_sum",
        "lambda_sq_sum",
        "three_norm_lambda_sum",
        "three_norm_lambda_sq_sum",
        "token_three_norm_lambda_sum",
        "token_three_norm_lambda_sq_sum",
        "mean_norm_three_norm_lambda_sum",
        "mean_norm_three_norm_lambda_sq_sum",
    )

    if checkpoint_path.exists() and not args.no_resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("signature") != run_signature:
            raise ValueError(f"Checkpoint does not match this run: {checkpoint_path}")
        if checkpoint.get("variant_names") != names:
            raise ValueError(
                f"Checkpoint variants do not match this run: {checkpoint_path}",
            )
        kernel_sums = checkpoint["kernel_sums"]
        profile_sums = checkpoint["profile_sums"]
        base_kernel_sums = checkpoint["base_kernel_sums"]
        lambda_sum = checkpoint["lambda_sum"]
        lambda_sq_sum = checkpoint["lambda_sq_sum"]
        three_norm_lambda_sum = checkpoint["three_norm_lambda_sum"]
        three_norm_lambda_sq_sum = checkpoint["three_norm_lambda_sq_sum"]
        token_three_norm_lambda_sum = checkpoint["token_three_norm_lambda_sum"]
        token_three_norm_lambda_sq_sum = checkpoint[
            "token_three_norm_lambda_sq_sum"
        ]
        mean_norm_three_norm_lambda_sum = checkpoint[
            "mean_norm_three_norm_lambda_sum"
        ]
        mean_norm_three_norm_lambda_sq_sum = checkpoint[
            "mean_norm_three_norm_lambda_sq_sum"
        ]
        sample_count = int(checkpoint["sample_count"])
        batches_processed = int(checkpoint["batches_processed"])
        print(
            f"Resuming from {checkpoint_path}: "
            f"{batches_processed} batches, {sample_count} samples",
        )

    def save_checkpoint() -> None:
        state = {
            "signature": run_signature,
            "variant_names": names,
            "sample_count": sample_count,
            "batches_processed": batches_processed,
        }
        state.update({name: locals_for_checkpoint[name] for name in accumulator_names})
        _atomic_torch_save(state, checkpoint_path)
        _save_input_cache(cached_inputs, input_cache_path, input_signature)

    # Capture references once; calling locals() inside save_checkpoint would inspect
    # that helper's local scope rather than main's accumulators.
    locals_for_checkpoint = {
        "kernel_sums": kernel_sums,
        "profile_sums": profile_sums,
        "base_kernel_sums": base_kernel_sums,
        "lambda_sum": lambda_sum,
        "lambda_sq_sum": lambda_sq_sum,
        "three_norm_lambda_sum": three_norm_lambda_sum,
        "three_norm_lambda_sq_sum": three_norm_lambda_sq_sum,
        "token_three_norm_lambda_sum": token_three_norm_lambda_sum,
        "token_three_norm_lambda_sq_sum": token_three_norm_lambda_sq_sum,
        "mean_norm_three_norm_lambda_sum": mean_norm_three_norm_lambda_sum,
        "mean_norm_three_norm_lambda_sq_sum": mean_norm_three_norm_lambda_sq_sum,
    }

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; checkpointing after the current batch.")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    source_iterator = None
    try:
        if len(cached_inputs) < args.num_batches:
            source_iterator = iter(setup_dataset(eval_config, tokenizer))
            source_skip = args.skip_batches + len(cached_inputs)
            for _ in tqdm(range(source_skip), desc="Skipping cached corpus samples"):
                try:
                    next(source_iterator)
                except StopIteration:
                    source_iterator = None
                    break
        progress = tqdm(
            range(batches_processed, args.num_batches),
            initial=batches_processed,
            total=args.num_batches,
            desc="Corpus samples",
        )
        for batch_index in progress:
            if batch_index < len(cached_inputs):
                inputs = cached_inputs[batch_index]
            else:
                if source_iterator is None:
                    break
                try:
                    inputs = next(source_iterator)
                except StopIteration:
                    break
                cached_inputs.append(
                    {key: value.detach().cpu() for key, value in inputs.items()},
                )
            residual_inputs[:] = [None] * num_layers
            attention_outputs[:] = [None] * num_layers
            post_attention[:] = [None] * num_layers
            attention_scores[:] = [None] * num_layers
            post_scores[:] = [None] * num_layers
            full_scores[:] = [None] * num_layers
            with torch.inference_mode():
                model(
                    input_ids=inputs["input_ids"].to(model.device),
                    attention_mask=inputs["attention_mask"].to(model.device),
                    use_cache=False,
                )

            score_values = [value for value in attention_scores if value is not None]
            if len(score_values) != num_layers:
                raise RuntimeError("Not all projected attention kernels were captured")
            rollout_device = score_values[0].device
            contribution_attention = torch.stack(
                [value.to(rollout_device) for value in score_values],
                dim=1,
            ).float()
            if args.kernel_only:
                batch_size = contribution_attention.shape[0]
                base_kernel_sums["forward_alti_attention"] += (
                    contribution_attention.detach().cpu().double().sum(dim=0)
                )
                sample_count += batch_size
                batches_processed += 1
                if (
                    args.checkpoint_every
                    and batches_processed % args.checkpoint_every == 0
                ):
                    save_checkpoint()
                if stop_requested:
                    save_checkpoint()
                    break
                continue
            raw_probabilities = [
                cast("ForwardSourceAttention", layer.self_attn).last_attention_probs
                for layer in layers
            ]
            if any(value is None for value in raw_probabilities):
                raise RuntimeError("Not all raw attention kernels were captured")
            raw = torch.stack(
                [
                    cast("torch.Tensor", value).to(rollout_device)
                    for value in raw_probabilities
                ],
                dim=1,
            ).float()
            if args.variant_set == "diagnostic":
                post = torch.stack(
                    [
                        value.to(rollout_device)
                        for value in post_scores
                        if value is not None
                    ],
                    dim=1,
                ).float()
                full = torch.stack(
                    [
                        value.to(rollout_device)
                        for value in full_scores
                        if value is not None
                    ],
                    dim=1,
                ).float()
            lambda_layers: list[torch.Tensor] = []
            three_norm_lambda_layers: list[torch.Tensor] = []
            token_three_norm_lambda_layers: list[torch.Tensor] = []
            mean_norm_three_norm_lambda_layers: list[torch.Tensor] = []
            for residual, attention_output in zip(
                residual_inputs,
                attention_outputs,
                strict=True,
            ):
                if residual is None or attention_output is None:
                    raise RuntimeError("Layer state missing while computing lambda")
                residual = residual.to(rollout_device)
                attention_output = attention_output.to(rollout_device)
                residual_norm = torch.linalg.vector_norm(
                    residual.float(),
                    dim=-1,
                )
                attention_norm = torch.linalg.vector_norm(
                    attention_output.float(),
                    dim=-1,
                )
                residual_frobenius = torch.linalg.vector_norm(
                    residual.float(),
                    dim=(-2, -1),
                )
                attention_frobenius = torch.linalg.vector_norm(
                    attention_output.float(),
                    dim=(-2, -1),
                )
                lambda_layers.append(
                    attention_frobenius
                    / (residual_frobenius + attention_frobenius).clamp_min(1e-30),
                )
                combined_norm = torch.linalg.vector_norm(
                    residual.float() + attention_output.float(),
                    dim=-1,
                )
                token_three_norm_strength = (
                    combined_norm + attention_norm - residual_norm
                ) / (2.0 * combined_norm).clamp_min(1e-30)
                token_three_norm_strength = torch.where(
                    combined_norm > 1e-30,
                    token_three_norm_strength,
                    torch.full_like(token_three_norm_strength, 0.5),
                )
                token_three_norm_lambda_layers.append(
                    token_three_norm_strength.clamp(0.0, 1.0).mean(dim=-1),
                )
                mean_residual_norm = residual_norm.mean(dim=-1)
                mean_attention_norm = attention_norm.mean(dim=-1)
                mean_combined_norm = combined_norm.mean(dim=-1)
                mean_norm_three_norm_strength = (
                    mean_combined_norm + mean_attention_norm - mean_residual_norm
                ) / (2.0 * mean_combined_norm).clamp_min(1e-30)
                mean_norm_three_norm_strength = torch.where(
                    mean_combined_norm > 1e-30,
                    mean_norm_three_norm_strength,
                    torch.full_like(mean_norm_three_norm_strength, 0.5),
                )
                mean_norm_three_norm_lambda_layers.append(
                    mean_norm_three_norm_strength.clamp(0.0, 1.0),
                )
                combined_frobenius = torch.linalg.vector_norm(
                    residual.float() + attention_output.float(),
                    dim=(-2, -1),
                )
                three_norm_strength = (
                    combined_frobenius + attention_frobenius - residual_frobenius
                ) / (2.0 * combined_frobenius).clamp_min(1e-30)
                three_norm_strength = torch.where(
                    combined_frobenius > 1e-30,
                    three_norm_strength,
                    torch.full_like(three_norm_strength, 0.5),
                )
                three_norm_lambda_layers.append(
                    three_norm_strength.clamp(0.0, 1.0),
                )
            batch_lambdas = torch.stack(lambda_layers, dim=1)
            batch_three_norm_lambdas = torch.stack(
                three_norm_lambda_layers,
                dim=1,
            )
            batch_token_three_norm_lambdas = torch.stack(
                token_three_norm_lambda_layers,
                dim=1,
            )
            batch_mean_norm_three_norm_lambdas = torch.stack(
                mean_norm_three_norm_lambda_layers,
                dim=1,
            )
            if args.variant_set == "paper":
                fixed_half = torch.full_like(batch_three_norm_lambdas, 0.5)
                paper_variant_lambdas = {
                    names[0]: batch_three_norm_lambdas,
                    names[1]: fixed_half,
                    names[2]: torch.ones_like(batch_three_norm_lambdas),
                }
            else:
                contribution_three_norm = _mix_raw_attention(
                    contribution_attention,
                    batch_three_norm_lambdas,
                )
                standard = _mix_raw_attention(raw, batch_lambdas)
                standard_three_norm = _mix_raw_attention(
                    raw,
                    batch_three_norm_lambdas,
                )
                standard_token_three_norm = _mix_raw_attention(
                    raw,
                    batch_token_three_norm_lambdas,
                )
                standard_mean_norm_three_norm = _mix_raw_attention(
                    raw,
                    batch_mean_norm_three_norm_lambdas,
                )
                contribution_standard = _mix_raw_attention(
                    contribution_attention,
                    batch_lambdas,
                )
                contribution_token_three_norm = _mix_raw_attention(
                    contribution_attention,
                    batch_token_three_norm_lambdas,
                )
                contribution_mean_norm_three_norm = _mix_raw_attention(
                    contribution_attention,
                    batch_mean_norm_three_norm_lambdas,
                )
                variants = {
                    names[0]: standard,
                    names[1]: standard_three_norm,
                    names[2]: standard_token_three_norm,
                    names[3]: standard_mean_norm_three_norm,
                    names[4]: contribution_standard,
                    names[5]: contribution_three_norm,
                    names[6]: contribution_token_three_norm,
                    names[7]: contribution_mean_norm_three_norm,
                    names[8]: post,
                    names[9]: full,
                    names[10]: raw,
                    names[11]: contribution_attention,
                }
            batch_size = raw.shape[0]
            base_kernel_sums["standard_attention"] += (
                raw.detach().cpu().double().sum(dim=0)
            )
            base_kernel_sums["forward_alti_attention"] += (
                contribution_attention.detach().cpu().double().sum(dim=0)
            )
            if args.variant_set == "paper":
                for name, variant_lambdas in paper_variant_lambdas.items():
                    _accumulate_mixed_kernel_sum(
                        kernel_sums[name],
                        contribution_attention,
                        variant_lambdas,
                    )
                    profile_sums[name] += (
                        _rollout_mixed_attention(
                            contribution_attention,
                            variant_lambdas,
                        )
                        .detach()
                        .cpu()
                        .double()
                        .sum(dim=0)
                    )
            else:
                for name, kernels in variants.items():
                    kernel_sums[name] += kernels.detach().cpu().double().sum(dim=0)
                    profile_sums[name] += (
                        _rollout(kernels).detach().cpu().double().sum(dim=0)
                    )
            lambda_sum += batch_lambdas.detach().cpu().double().sum(dim=0)
            lambda_sq_sum += batch_lambdas.detach().cpu().double().square().sum(dim=0)
            three_norm_lambda_sum += (
                batch_three_norm_lambdas.detach().cpu().double().sum(dim=0)
            )
            three_norm_lambda_sq_sum += (
                batch_three_norm_lambdas.detach().cpu().double().square().sum(dim=0)
            )
            token_three_norm_lambda_sum += (
                batch_token_three_norm_lambdas.detach().cpu().double().sum(dim=0)
            )
            token_three_norm_lambda_sq_sum += (
                batch_token_three_norm_lambdas.detach()
                .cpu()
                .double()
                .square()
                .sum(dim=0)
            )
            mean_norm_three_norm_lambda_sum += (
                batch_mean_norm_three_norm_lambdas.detach().cpu().double().sum(dim=0)
            )
            mean_norm_three_norm_lambda_sq_sum += (
                batch_mean_norm_three_norm_lambdas.detach()
                .cpu()
                .double()
                .square()
                .sum(dim=0)
            )
            sample_count += batch_size
            batches_processed += 1
            if (
                args.checkpoint_every
                and batches_processed % args.checkpoint_every == 0
            ):
                save_checkpoint()
            if stop_requested:
                save_checkpoint()
                break
    finally:
        _close_data_iterator(source_iterator)
        for handle in handles:
            handle.remove()

    if stop_requested:
        print(
            f"Stopped after {batches_processed} batches and {sample_count} samples; "
            f"resume state saved to {checkpoint_path}",
        )
        return

    if sample_count == 0:
        raise RuntimeError("No samples were processed")

    if args.kernel_only:
        np.savez(
            args.output_dir / "transport_kernels.npz",
            sample_count=np.asarray(sample_count),
            batches_processed=np.asarray(batches_processed),
            mean_forward_alti_attention=(
                base_kernel_sums["forward_alti_attention"] / sample_count
            ).float().numpy(),
        )
        checkpoint_path.unlink(missing_ok=True)
        print(
            f"Saved corpus-averaged unmixed transport kernels from "
            f"{sample_count} samples to {args.output_dir / 'transport_kernels.npz'}",
        )
        return

    mean_lambda = lambda_sum / sample_count
    lambda_variance = (lambda_sq_sum / sample_count - mean_lambda.square()).clamp_min(
        0.0,
    )
    lambda_ci95 = 1.96 * torch.sqrt(lambda_variance / sample_count)
    mean_three_norm_lambda = three_norm_lambda_sum / sample_count
    three_norm_lambda_variance = (
        three_norm_lambda_sq_sum / sample_count - mean_three_norm_lambda.square()
    ).clamp_min(0.0)
    three_norm_lambda_ci95 = 1.96 * torch.sqrt(
        three_norm_lambda_variance / sample_count,
    )
    mean_token_three_norm_lambda = token_three_norm_lambda_sum / sample_count
    token_three_norm_lambda_variance = (
        token_three_norm_lambda_sq_sum / sample_count
        - mean_token_three_norm_lambda.square()
    ).clamp_min(0.0)
    token_three_norm_lambda_ci95 = 1.96 * torch.sqrt(
        token_three_norm_lambda_variance / sample_count,
    )
    mean_mean_norm_three_norm_lambda = mean_norm_three_norm_lambda_sum / sample_count
    mean_norm_three_norm_lambda_variance = (
        mean_norm_three_norm_lambda_sq_sum / sample_count
        - mean_mean_norm_three_norm_lambda.square()
    ).clamp_min(0.0)
    mean_norm_three_norm_lambda_ci95 = 1.96 * torch.sqrt(
        mean_norm_three_norm_lambda_variance / sample_count,
    )
    profiles: dict[str, dict[str, np.ndarray]] = {
        "Product of corpus-mean kernels": {},
        "Mean of per-sample rollouts": {},
        "Separate means": {},
    }
    mean_kernels: dict[str, np.ndarray] = {}
    for name in names:
        mean_kernel = (kernel_sums[name] / sample_count).float().to(model.device)
        profiles["Product of corpus-mean kernels"][name] = _normalize(
            _rollout(mean_kernel.unsqueeze(0))[0].cpu().numpy(),
        )
        profiles["Mean of per-sample rollouts"][name] = _normalize(
            (profile_sums[name] / sample_count).numpy(),
        )
        mean_kernels[name] = mean_kernel.cpu().numpy()

    mean_standard_attention = (
        base_kernel_sums["standard_attention"] / sample_count
    ).float().to(model.device)
    mean_forward_alti_attention = (
        base_kernel_sums["forward_alti_attention"] / sample_count
    ).float().to(model.device)
    if args.variant_set == "paper":
        fixed_half_mean = torch.full_like(
            mean_three_norm_lambda.unsqueeze(0),
            0.5,
        ).to(model.device)
        separate_mean_variants = {
            names[0]: _mix_raw_attention(
                mean_forward_alti_attention.unsqueeze(0),
                mean_three_norm_lambda.unsqueeze(0).to(model.device),
            )[0],
            names[1]: _mix_raw_attention(
                mean_forward_alti_attention.unsqueeze(0),
                fixed_half_mean,
            )[0],
            names[2]: mean_forward_alti_attention,
        }
    else:
        separate_mean_variants = {
            names[0]: _mix_raw_attention(
                mean_standard_attention.unsqueeze(0),
                mean_lambda.unsqueeze(0).to(model.device),
            )[0],
            names[1]: _mix_raw_attention(
                mean_standard_attention.unsqueeze(0),
                mean_three_norm_lambda.unsqueeze(0).to(model.device),
            )[0],
            names[2]: _mix_raw_attention(
                mean_standard_attention.unsqueeze(0),
                mean_token_three_norm_lambda.unsqueeze(0).to(model.device),
            )[0],
            names[3]: _mix_raw_attention(
                mean_standard_attention.unsqueeze(0),
                mean_mean_norm_three_norm_lambda.unsqueeze(0).to(model.device),
            )[0],
            names[4]: _mix_raw_attention(
                mean_forward_alti_attention.unsqueeze(0),
                mean_lambda.unsqueeze(0).to(model.device),
            )[0],
            names[5]: _mix_raw_attention(
                mean_forward_alti_attention.unsqueeze(0),
                mean_three_norm_lambda.unsqueeze(0).to(model.device),
            )[0],
            names[6]: _mix_raw_attention(
                mean_forward_alti_attention.unsqueeze(0),
                mean_token_three_norm_lambda.unsqueeze(0).to(model.device),
            )[0],
            names[7]: _mix_raw_attention(
                mean_forward_alti_attention.unsqueeze(0),
                mean_mean_norm_three_norm_lambda.unsqueeze(0).to(model.device),
            )[0],
            names[10]: mean_standard_attention,
            names[11]: mean_forward_alti_attention,
        }
    for name, kernels in separate_mean_variants.items():
        profiles["Separate means"][name] = _normalize(
            _rollout(kernels.unsqueeze(0))[0].cpu().numpy(),
        )

    measured = _load_measured_profile(args.measured_profile, -1)
    results: list[dict[str, str | float]] = []
    for aggregation, aggregation_profiles in profiles.items():
        for name, profile in aggregation_profiles.items():
            results.append(
                {
                    "aggregation": aggregation,
                    "variant": name,
                    **_metrics(measured, profile),
                },
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "forward_source_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    np.savetxt(
        args.output_dir / "lambda_schedule.csv",
        np.column_stack(
            (
                np.arange(num_layers),
                mean_lambda.numpy(),
                (mean_lambda - lambda_ci95).numpy(),
                (mean_lambda + lambda_ci95).numpy(),
                mean_three_norm_lambda.numpy(),
                (mean_three_norm_lambda - three_norm_lambda_ci95).numpy(),
                (mean_three_norm_lambda + three_norm_lambda_ci95).numpy(),
                mean_token_three_norm_lambda.numpy(),
                (
                    mean_token_three_norm_lambda - token_three_norm_lambda_ci95
                ).numpy(),
                (
                    mean_token_three_norm_lambda + token_three_norm_lambda_ci95
                ).numpy(),
                mean_mean_norm_three_norm_lambda.numpy(),
                (
                    mean_mean_norm_three_norm_lambda
                    - mean_norm_three_norm_lambda_ci95
                ).numpy(),
                (
                    mean_mean_norm_three_norm_lambda
                    + mean_norm_three_norm_lambda_ci95
                ).numpy(),
            ),
        ),
        delimiter=",",
        header=(
            "layer,norm_ratio_mean,norm_ratio_ci95_lower,norm_ratio_ci95_upper,"
            "three_norm_frobenius_mean,three_norm_frobenius_ci95_lower,"
            "three_norm_frobenius_ci95_upper,three_norm_token_mean,"
            "three_norm_token_ci95_lower,three_norm_token_ci95_upper,"
            "three_norm_ratio_of_token_mean_norms,"
            "three_norm_ratio_of_token_mean_norms_ci95_lower,"
            "three_norm_ratio_of_token_mean_norms_ci95_upper"
        ),
        comments="",
    )
    np.savez(
        args.output_dir / "forward_source_profiles.npz",
        variant_names=np.asarray(names),
        variant_set=np.asarray(args.variant_set),
        sample_count=np.asarray(sample_count),
        batches_processed=np.asarray(batches_processed),
        measured=measured,
        mean_lambda=mean_lambda.numpy(),
        lambda_ci95=lambda_ci95.numpy(),
        mean_three_norm_lambda=mean_three_norm_lambda.numpy(),
        three_norm_lambda_ci95=three_norm_lambda_ci95.numpy(),
        mean_token_three_norm_lambda=mean_token_three_norm_lambda.numpy(),
        token_three_norm_lambda_ci95=token_three_norm_lambda_ci95.numpy(),
        mean_mean_norm_three_norm_lambda=mean_mean_norm_three_norm_lambda.numpy(),
        mean_norm_three_norm_lambda_ci95=mean_norm_three_norm_lambda_ci95.numpy(),
        mean_standard_attention=mean_standard_attention.cpu().numpy(),
        mean_forward_alti_attention=mean_forward_alti_attention.cpu().numpy(),
        **{
            f"profile_{aggregation_index}_{variant_index}": profile
            for aggregation_index, aggregation_profiles in enumerate(profiles.values())
            for variant_index, profile in enumerate(aggregation_profiles.values())
        },
        **{f"kernel_{index}": mean_kernels[name] for index, name in enumerate(names)},
    )

    positions = np.arange(sequence_length)
    colors = [
        "#4477AA",
        "#66CCEE",
        "#009988",
        "#CCBB44",
        "#AA3377",
        "#EE99AA",
        "#228833",
        "#EE6677",
        "#BBBBBB",
        "#000000",
        "#332288",
        "#882255",
    ]
    fig, axes = plt.subplots(
        1,
        len(profiles),
        figsize=(17.25, 4.1),
        dpi=200,
        sharey=True,
    )
    for ax, (aggregation, aggregation_profiles) in zip(
        axes,
        profiles.items(),
        strict=True,
    ):
        ax.plot(positions, measured, color="black", linewidth=2.3, label="Measured")
        for color, (name, profile) in zip(
            colors,
            aggregation_profiles.items(),
            strict=False,
        ):
            metric = _metrics(measured, profile)["wasserstein_normalized"]
            ax.plot(
                positions,
                profile,
                color=color,
                linewidth=1.7,
                label=f"{name} (W1={metric:.3f})",
            )
        ax.set_title(aggregation)
        ax.set_xlabel("Token position")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("Normalized influence")
    fig.tight_layout()
    fig.savefig(args.output_dir / "forward_source_rollout.png", bbox_inches="tight")
    fig.savefig(args.output_dir / "forward_source_rollout.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=200)
    layer_positions = np.arange(num_layers)
    ax.plot(
        layer_positions,
        mean_lambda.numpy(),
        color="#4477AA",
        linewidth=2,
        label="Original norm ratio",
    )
    ax.fill_between(
        layer_positions,
        (mean_lambda - lambda_ci95).numpy(),
        (mean_lambda + lambda_ci95).numpy(),
        color="#4477AA",
        alpha=0.2,
    )
    ax.plot(
        layer_positions,
        mean_token_three_norm_lambda.numpy(),
        color="#009988",
        linewidth=2,
        label="Three-norm token mean",
    )
    ax.fill_between(
        layer_positions,
        (mean_token_three_norm_lambda - token_three_norm_lambda_ci95).numpy(),
        (mean_token_three_norm_lambda + token_three_norm_lambda_ci95).numpy(),
        color="#009988",
        alpha=0.2,
    )
    ax.plot(
        layer_positions,
        mean_mean_norm_three_norm_lambda.numpy(),
        color="#CCBB44",
        linewidth=2,
        label="Three-norm ratio of token-mean norms",
    )
    ax.fill_between(
        layer_positions,
        (
            mean_mean_norm_three_norm_lambda - mean_norm_three_norm_lambda_ci95
        ).numpy(),
        (
            mean_mean_norm_three_norm_lambda + mean_norm_three_norm_lambda_ci95
        ).numpy(),
        color="#CCBB44",
        alpha=0.2,
    )
    ax.plot(
        layer_positions,
        mean_three_norm_lambda.numpy(),
        color="#AA3377",
        linewidth=2,
        label="Three-norm Frobenius",
    )
    ax.fill_between(
        layer_positions,
        (mean_three_norm_lambda - three_norm_lambda_ci95).numpy(),
        (mean_three_norm_lambda + three_norm_lambda_ci95).numpy(),
        color="#AA3377",
        alpha=0.2,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"Effective attention mixing $\lambda_t$")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "lambda_schedule.png", bbox_inches="tight")
    fig.savefig(args.output_dir / "lambda_schedule.pdf", bbox_inches="tight")
    plt.close(fig)

    _save_input_cache(cached_inputs, input_cache_path, input_signature)
    checkpoint_path.unlink(missing_ok=True)

    print(f"samples={sample_count}")
    print(f"lambda_mean={mean_lambda.tolist()}")
    print(f"three_norm_lambda_mean={mean_three_norm_lambda.tolist()}")
    print(
        "token_three_norm_lambda_mean="
        f"{mean_token_three_norm_lambda.tolist()}",
    )
    print(
        "mean_norm_three_norm_lambda_mean="
        f"{mean_mean_norm_three_norm_lambda.tolist()}",
    )
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
