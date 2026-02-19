"""
Evaluate position bias.

Tools to evaluate position-bias metrics for causal language models including
wrapping attention layers, computing attributions, and saving statistics and
plots generated during evaluation.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import datasets
from einops import rearrange
import structlog
import torch
from torch import nn
from torch.autograd.graph import save_on_cpu
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    enable_full_determinism,
)
import typer
import wandb

from src.dataset_wrapper import DatasetWrapper
from src.models.wrap import (
    save_attribution_stats,
    save_wrapper_stats,
    wrap_attention_layers,
)
from src.util.configs import EvalConfig
from src.util.hooks import register_metric_hooks, unregister_hooks
from src.util.pydantic import load_config

logger = structlog.get_logger()


def setup_wandb(eval_config: EvalConfig) -> None:
    """
    Initialize Weights & Biases for logging.

    Args:
        eval_config: Evaluation configuration that contains `wandb_config`.

    """
    wandb_config = eval_config.wandb_config
    wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        dir=wandb_config.dir,
        id=wandb_config.id,
        name=wandb_config.name,
        notes=wandb_config.notes,
        tags=wandb_config.tags,
        group=wandb_config.group,
        job_type=wandb_config.job_type,
        save_code=wandb_config.save_code,
        config=eval_config.model_dump(),
    )

    if wandb.run is not None:
        wandb.run.define_metric("final/*", step_metric="final/layer_idx")


def setup_save_path(eval_config: EvalConfig) -> Path:
    """
    Create and return the save path for evaluation results.

    Args:
        eval_config: Evaluation configuration that provides `save_path` and model details.

    Returns:
        Path: Path where evaluation outputs should be saved.

    """
    save_path = (
        eval_config.save_path
        / f"{eval_config.hf_model_config.model_id.replace('/', '_')}"
        / f"{eval_config.sequence_length}"
    )
    if save_path.exists():
        confirm = typer.confirm(
            f"Save folder {save_path} already exists. Do you want to delete it and continue?",
            default=False,
        )
        if not confirm:
            logger.info("Exiting as per user request.")
            raise SystemExit(0)

        import shutil  # noqa: PLC0415

        shutil.rmtree(save_path)

    save_path.mkdir(parents=True, exist_ok=True)
    return save_path


def load_model(eval_config: EvalConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]:  # noqa: C901
    """
    Load model and tokenizer and prepare the model for evaluation.

    Args:
        eval_config: Evaluation configuration describing the model and runtime options.

    Returns:
        Loaded model and tokenizer.

    """
    model_config = eval_config.hf_model_config

    logger.info("Loading model and tokenizer", model_id=model_config.model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_id,
        padding_side="left",
        cache_dir=model_config.cache_dir,
        device_map=model_config.device_map or "auto",
        max_memory=model_config.max_memory,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_id,
        cache_dir=model_config.cache_dir,
        device_map=model_config.device_map or "auto",
        max_memory=model_config.max_memory,
        dtype=torch.bfloat16,
    ).eval()
    model.requires_grad_(requires_grad=False)

    if eval_config.enable_attribution and hasattr(
        model,
        "gradient_checkpointing_enable",
    ):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    if eval_config.enable_attribution:

        def make_checkpointed_forward(
            layer: nn.Module,
        ) -> Callable[..., torch.Tensor]:
            original_forward = layer.forward

            def checkpointed_forward(
                *args: Any,  # noqa: ANN401
                **kwargs: Any,  # noqa: ANN401
            ) -> torch.Tensor:
                return checkpoint(
                    original_forward,
                    *args,
                    **kwargs,
                    use_reentrant=False,
                )  # pyright: ignore[reportReturnType]

            return checkpointed_forward

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            for layer in model.model.layers:
                layer.forward = make_checkpointed_forward(layer)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            for layer in model.transformer.h:  # pyright: ignore[reportAttributeAccessIssue,reportGeneralTypeIssues]
                layer.forward = make_checkpointed_forward(layer)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
            for layer in model.transformer.blocks:  # pyright: ignore[reportAttributeAccessIssue,reportGeneralTypeIssues]
                layer.forward = make_checkpointed_forward(layer)

    model.config.use_cache = False

    if eval_config.enable_qk_stats:
        wrap_attention_layers(model, eval_config.sequence_length)

    model.eval()

    return model, tokenizer


def setup_dataset(
    eval_config: EvalConfig,
    tokenizer: PreTrainedTokenizer,
) -> DataLoader:
    """
    Create a wrapped dataset and DataLoader for evaluation.

    Args:
        eval_config: Evaluation configuration with dataset parameters.
        tokenizer: Tokenizer to use for tokenization and wrapping.

    Returns:
        DataLoader: DataLoader providing tokenized batches for evaluation.

    """
    dataset_config = eval_config.dataset_config

    dataset = datasets.load_dataset(
        dataset_config.repo_id,
        name=dataset_config.sample,
        revision=dataset_config.revision,
        split=dataset_config.split,
        streaming=dataset_config.streaming,
    )

    logger.info("Creating DataLoader")
    wrapped_dataset = DatasetWrapper(
        dataset,  # pyright: ignore[reportArgumentType]
        tokenizer,
        seq_length=eval_config.sequence_length,
    )
    dataloader = DataLoader(
        wrapped_dataset,  # pyright: ignore[reportArgumentType]
        batch_size=eval_config.batch_size,
        num_workers=eval_config.num_workers,
    )

    return dataloader


def log_attention_stats(
    eval_config: EvalConfig,
    metrics: dict[int, dict[str, float | int]],
) -> None:
    """
    Log attention and residual statistics to console and wandb.

    Args:
        eval_config: Evaluation configuration providing expected totals.
        metrics: Mapping from layer index to collected metric statistics.

    """
    decimal_places = 3

    logger.info(
        "Layer-wise Effective Attention and Residual Strengths (mean ± stddev):",
    )
    for layer_idx, layer_metrics in sorted(metrics.items()):
        expected_total = eval_config.num_batches * eval_config.batch_size
        if layer_metrics["attn_count"] != expected_total:
            logger.warning(
                "Observed sample count (attn_count = %d, res_count = %d) for layer %d does not match the "
                "configured total (%d). This can happen if the dataset has "
                "fewer samples than num_batches * batch_size or if the last "
                "batch is smaller than batch_size.",
                layer_metrics["attn_count"],
                layer_metrics["res_count"],
                layer_idx,
                expected_total,
            )

        attn_mean = layer_metrics["attn_sum"] / layer_metrics["attn_count"]
        attn_var = (layer_metrics["attn_sum_sq"] / layer_metrics["attn_count"]) - (
            attn_mean**2
        )
        attn_std = max(attn_var, 0.0) ** 0.5

        res_mean = layer_metrics["res_sum"] / layer_metrics["res_count"]
        res_var = (layer_metrics["res_sum_sq"] / layer_metrics["res_count"]) - (
            res_mean**2
        )
        res_std = max(res_var, 0.0) ** 0.5

        logger.info(
            f"Layer {layer_idx}:\t{attn_mean:.{decimal_places}f}±{attn_std:.{decimal_places}f},\t{res_mean:.{decimal_places}f}±{res_std:.{decimal_places}f}",
        )

        wandb.log(
            {
                "final/layer_idx": layer_idx,
                "final/attn_mean": attn_mean,
                "final/attn_var": attn_var,
                "final/attn_stddev": attn_std,
                "final/res_mean": res_mean,
                "final/res_var": res_var,
                "final/res_stddev": res_std,
            },
        )


def evaluate_model(
    eval_config: EvalConfig,
    model: PreTrainedModel,
    dataloader: DataLoader,
    save_path: Path,
) -> None:
    """
    Run the evaluation loop and collect metrics.

    Args:
        eval_config: Evaluation configuration and flags for which metrics to collect.
        model: The pretrained model to evaluate.
        dataloader: DataLoader yielding evaluation batches.
        save_path: Directory to save resulting statistics and artifacts.

    """
    logger.info("Evaluating")

    metrics: dict[int, dict[str, float | int]] = {}
    attribution_sum: torch.Tensor | None = None
    attribution_count: torch.Tensor | None = None
    if eval_config.enable_attribution:
        attribution_sum = torch.zeros(eval_config.sequence_length, dtype=torch.float64)
        attribution_count = torch.zeros(eval_config.sequence_length, dtype=torch.int64)

    for batch_idx, inputs in tqdm(enumerate(dataloader), total=eval_config.num_batches):
        if batch_idx >= eval_config.num_batches:
            break

        hook_handles = []
        if eval_config.enable_attention_stats:
            hook_handles = register_metric_hooks(
                model,
                batch_idx,
                metrics,
            )

        model.eval()
        if eval_config.enable_attribution:
            embed_device = model.get_input_embeddings().weight.device
            attention_mask = inputs["attention_mask"].to(embed_device)
            inputs_embeds = model.get_input_embeddings()(
                inputs["input_ids"].to(embed_device),
            ).detach()
            inputs_embeds.requires_grad_(requires_grad=True)

            with save_on_cpu(pin_memory=True):
                outputs = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                logits = outputs.logits
                probs = torch.softmax(logits[:, -1, :].float(), dim=-1)
                target_ids = probs.argmax(dim=-1)
                target_probs = rearrange(
                    probs.gather(1, rearrange(target_ids, "b -> b 1")),
                    "b 1 -> b",
                )

            grads = torch.autograd.grad(
                target_probs.sum(),
                inputs_embeds,
                retain_graph=False,
            )[0]
            grad_norms = grads.norm(dim=-1)
            mask = attention_mask.to(dtype=grad_norms.dtype)
            batch_sum = (grad_norms * mask).sum(dim=0).detach().cpu()
            batch_count = mask.sum(dim=0).detach().cpu()

            attribution_sum += batch_sum.to(dtype=torch.float64)
            attribution_count += batch_count.to(dtype=torch.int64)

            del (
                inputs,
                outputs,
                logits,
                probs,
                target_ids,
                target_probs,
                grads,
                grad_norms,
                mask,
                batch_sum,
                batch_count,
            )
            torch.cuda.empty_cache()
        else:
            with torch.inference_mode():
                outputs = model(
                    input_ids=inputs["input_ids"].to(model.device),
                    attention_mask=inputs["attention_mask"].to(model.device),
                    use_cache=False,
                )

        if hook_handles:
            unregister_hooks(hook_handles)

    if eval_config.enable_qk_stats:
        logger.info("Saving wrapper statistics")
        save_wrapper_stats(model, save_path, eval_config.sequence_length)

    if eval_config.enable_attribution:
        assert attribution_sum is not None
        assert attribution_count is not None

        logger.info("Saving attribution statistics")
        attribution_mean = attribution_sum / attribution_count
        save_attribution_stats(
            attribution_mean,
            save_path,
        )

    if eval_config.enable_attention_stats:
        log_attention_stats(eval_config, metrics)


def main(config_path: Path, seed: int = 42) -> None:
    """
    Main entry point to run the evaluation.

    Args:
        config_path: Path to the YAML configuration file describing evaluation settings.
        seed: Random seed for deterministic behavior (default: 42).

    """
    enable_full_determinism(seed, warn_only=True)

    eval_config = load_config(path=config_path, config_type=EvalConfig)

    logger.info(f"Running configuration: {eval_config}")

    save_path = setup_save_path(eval_config)

    setup_wandb(eval_config)

    model, tokenizer = load_model(eval_config)

    dataloader = setup_dataset(eval_config, tokenizer)

    evaluate_model(eval_config, model, dataloader, save_path)

    wandb.finish()


if __name__ == "__main__":
    typer.run(main)
