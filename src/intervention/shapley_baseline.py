"""Measure baseline branch geometry for a no-refit norm-Shapley beta intervention.

The script uses only forward activations of the unmodified model.  For every
prompt and layer it records the three scalar sufficient statistics
``||r||^2``, ``||a||^2``, and ``<r,a>``.  They determine the norm-Shapley
coefficient after a hypothetical attention-branch scaling ``a -> beta*a``
without consulting an intervened influence profile.
"""

from pathlib import Path
from typing import Annotated

import structlog
import torch
from torch import nn
import typer

from src.intervention.intervention import (
    _attention_modules,
    _decoder_blocks,
    load_causal_model,
)
from src.util.configs import EvalConfig
from src.util.pydantic import load_config

logger = structlog.get_logger()


def _attention_tensor(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Return the pre-residual attention branch from a supported hook output."""
    if isinstance(output, tuple):
        return output[0]
    return output


def capture_branch_statistics(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return per-layer ``[||r||^2, ||a||^2, <r,a>]`` for one prompt."""
    blocks = _decoder_blocks(model)  # type: ignore[arg-type]
    attention_modules = _attention_modules(model)  # type: ignore[arg-type]
    residuals: dict[int, torch.Tensor] = {}
    statistics: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def residual_hook(layer: int):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            residuals[layer] = inputs[0].detach()

        return hook

    def attention_hook(layer: int):
        def hook(
            _module: nn.Module,
            _inputs: object,
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            residual = residuals.pop(layer).float()
            attention = _attention_tensor(output).detach().float()
            if residual.shape != attention.shape:
                raise ValueError(
                    f"layer {layer}: residual shape {tuple(residual.shape)} does not match "
                    f"attention shape {tuple(attention.shape)}",
                )
            statistics[layer] = torch.stack(
                (
                    torch.sum(residual * residual),
                    torch.sum(attention * attention),
                    torch.sum(residual * attention),
                ),
            ).cpu()

        return hook

    for layer, block in enumerate(blocks):
        handles.append(block.register_forward_pre_hook(residual_hook(layer)))
    for layer, attention_module in enumerate(attention_modules):
        handles.append(attention_module.register_forward_hook(attention_hook(layer)))

    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    if residuals:
        raise RuntimeError(f"attention hooks did not consume residuals for layers {sorted(residuals)}")
    return torch.stack([statistics[layer] for layer in range(len(blocks))]).to(torch.float64)


def shapley_lambdas_from_statistics(statistics: torch.Tensor, beta: float) -> torch.Tensor:
    """Compute per-prompt, per-layer norm-Shapley coefficients at ``beta``.

    ``statistics`` has final dimension ``[r_sq, a_sq, dot]``.  The original
    residual and attention directions are held fixed, making this a baseline
    forward prediction rather than a recalibration on the intervened model.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    if statistics.shape[-1] != 3:
        raise ValueError(f"statistics must have final dimension 3, got {statistics.shape}")

    r_sq, a_sq, dot = statistics.unbind(dim=-1)
    r_norm = torch.sqrt(torch.clamp(r_sq, min=0.0))
    a_norm = torch.sqrt(torch.clamp(a_sq, min=0.0))
    sum_sq = r_sq + 2.0 * beta * dot + beta * beta * a_sq
    sum_norm = torch.sqrt(torch.clamp(sum_sq, min=0.0))
    numerator = sum_norm + beta * a_norm - r_norm
    coefficient = torch.where(sum_norm > 0, numerator / (2.0 * sum_norm), torch.full_like(sum_norm, 0.5))
    return torch.clamp(coefficient, min=0.0, max=1.0)


def mean_shapley_schedule(statistics: torch.Tensor, beta: float) -> list[float]:
    """Average per-input norm-Shapley coefficients within each layer."""
    if statistics.ndim != 3:
        raise ValueError(f"expected [prompts,layers,3] statistics, got {statistics.shape}")
    return shapley_lambdas_from_statistics(statistics, beta).mean(dim=0).tolist()


def main(
    config: Annotated[Path, typer.Option("--config", help="Path to the n=256 EvalConfig YAML.")],
    manifest: Annotated[Path, typer.Option("--manifest", help="Shared fixed-target prompt manifest.")],
    output: Path = Path("results/intervention/baseline_branch_stats.pt"),
) -> None:
    """Measure and save baseline branch statistics for all manifest prompts."""
    eval_config = load_config(config, EvalConfig)
    model, _tokenizer = load_causal_model(eval_config)
    manifest_data = torch.load(manifest, map_location="cpu", weights_only=False)
    input_ids_all = manifest_data["input_ids"]
    attention_mask_all = manifest_data["attention_mask"]

    embed = model.get_input_embeddings()
    assert isinstance(embed, nn.Embedding)
    device = embed.weight.device
    all_statistics = []
    for index, (input_ids, attention_mask) in enumerate(zip(input_ids_all, attention_mask_all, strict=True)):
        statistics = capture_branch_statistics(
            model,
            input_ids.unsqueeze(0).to(device),
            attention_mask.unsqueeze(0).to(device),
        )
        all_statistics.append(statistics)
        logger.info("Measured baseline branch geometry", prompt=index + 1, total=len(input_ids_all))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "statistics": torch.stack(all_statistics),
        "prompt_ids": manifest_data["prompt_ids"],
        "model_id": eval_config.hf_model_config.model_id,
        "sequence_length": eval_config.sequence_length,
        "definition": "frobenius_norm_shapley",
        "columns": ["residual_squared_norm", "attention_squared_norm", "inner_product"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    logger.info("Wrote baseline branch statistics", path=str(output), prompts=len(all_statistics))


if __name__ == "__main__":
    typer.run(main)
