"""
Evaluate position bias.

Tools to evaluate position-bias metrics for causal language models including
wrapping attention layers, computing attributions, and saving statistics and
plots generated during evaluation.
"""

from pathlib import Path
from typing import Any

import datasets
import structlog
import torch
from torch import nn
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
from src.util.causal_mask import disable_causal_mask
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
    seq_len_dir = f"{eval_config.sequence_length}"
    if eval_config.disable_causal_mask:
        seq_len_dir = f"{seq_len_dir}_unmasked"
    if eval_config.run_name:
        seq_len_dir = f"{seq_len_dir}_{eval_config.run_name}"
    save_path = (
        eval_config.save_path
        / f"{eval_config.hf_model_config.model_id.replace('/', '_')}"
        / seq_len_dir
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


def load_model(eval_config: EvalConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
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
        attn_implementation="eager" if eval_config.disable_causal_mask else None,
    ).eval()
    model.requires_grad_(requires_grad=False)

    model.config.use_cache = False

    if eval_config.disable_causal_mask:
        disable_causal_mask(model)
        logger.info("Causal masking disabled (fully bidirectional attention).")

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


def close_dataloader_iterator(iterator: object) -> None:
    """Close an interrupted DataLoader iterator and its streaming source."""
    fetcher = getattr(iterator, "_dataset_fetcher", None)
    dataset_iterator = getattr(fetcher, "dataset_iter", None)
    close = getattr(dataset_iterator, "close", None)
    if callable(close):
        close()

    shutdown_workers = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown_workers):
        shutdown_workers()


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


def _get_decoder_layers(model: PreTrainedModel) -> list[nn.Module]:
    """Return the list of decoder/transformer blocks for supported architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType,reportGeneralTypeIssues]
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        return list(model.transformer.blocks)  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType,reportGeneralTypeIssues]
    return []


def evaluate_model(  # noqa: C901, PLR0915
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
    per_head_attribution_sum: torch.Tensor | None = None
    layer_outputs: list[torch.Tensor] = []
    o_proj_inputs: list[torch.Tensor] = []
    hook_handles_attr: list[torch.utils.hooks.RemovableHandle] = []
    num_layers = 0
    num_heads = 0
    head_dim = 0
    if eval_config.enable_attribution:
        decoder_layers = _get_decoder_layers(model)
        num_layers = len(decoder_layers)

        attribution_sum = torch.zeros(
            num_layers,
            eval_config.sequence_length,
            dtype=torch.float64,
            device=model.device,
        )
        attribution_count = torch.zeros(
            eval_config.sequence_length,
            dtype=torch.int64,
            device=model.device,
        )

        if eval_config.enable_attribution_per_head:
            num_heads = model.config.num_attention_heads
            hidden_size = model.config.hidden_size
            head_dim = getattr(
                model.config,
                "head_dim",
                hidden_size // num_heads,
            )
            per_head_attribution_sum = torch.zeros(
                num_layers,
                num_heads,
                eval_config.sequence_length,
                dtype=torch.float64,
                device=model.device,
            )

        capture_enabled = False

        def _capture_layer_output(
            _module: nn.Module,
            _args: tuple[Any, ...],
            output: Any,  # noqa: ANN401
        ) -> None:
            if not capture_enabled:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            layer_outputs.append(hidden)

        def _capture_o_proj_input(
            _module: nn.Module,
            args: tuple[Any, ...],
        ) -> None:
            if not capture_enabled:
                return
            o_proj_in = args[0]
            o_proj_in.retain_grad()
            o_proj_inputs.append(o_proj_in)

        hook_handles_attr = [
            layer.register_forward_hook(_capture_layer_output)
            for layer in decoder_layers
        ]
        if eval_config.enable_attribution_per_head:
            hook_handles_attr.extend(
                layer.self_attn.o_proj.register_forward_pre_hook(_capture_o_proj_input)
                for layer in decoder_layers
            )

    for batch_idx, inputs in tqdm(enumerate(dataloader), total=eval_config.num_batches):
        if batch_idx >= eval_config.num_batches:
            break

        hook_handles = []
        if eval_config.enable_attention_stats:
            hook_handles = register_metric_hooks(
                model,
                batch_idx,
                metrics,
                norm_type=eval_config.attention_stats_norm,
            )

        model.eval()
        if eval_config.enable_attribution:
            assert attribution_sum is not None
            assert attribution_count is not None

            layer_outputs.clear()
            o_proj_inputs.clear()
            embed_device = model.get_input_embeddings().weight.device
            attention_mask = inputs["attention_mask"].to(embed_device)
            inputs_embeds = model.get_input_embeddings()(
                inputs["input_ids"].to(embed_device),
            ).detach()
            inputs_embeds.requires_grad_(requires_grad=True)

            capture_enabled = True
            # with save_on_cpu(pin_memory=True):
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            capture_enabled = False

            assert len(layer_outputs) == num_layers, (
                f"Captured {len(layer_outputs)} layer outputs, expected {num_layers}."
            )
            if eval_config.enable_attribution_per_head:
                assert len(o_proj_inputs) == num_layers, (
                    f"Captured {len(o_proj_inputs)} o_proj inputs, "
                    f"expected {num_layers}."
                )
            per_layer_grads = []
            per_layer_head_grads: list[torch.Tensor] = []
            for layer_idx, hidden in enumerate(layer_outputs):
                scalar = hidden[:, -1, :].norm(dim=-1).sum()
                is_last_layer = layer_idx == num_layers - 1
                retain = not is_last_layer or eval_config.enable_attribution_per_head
                grad = torch.autograd.grad(
                    scalar,
                    inputs_embeds,
                    retain_graph=retain,
                )[0]
                per_layer_grads.append(grad.norm(dim=-1).detach())

                if eval_config.enable_attribution_per_head:
                    # (B, S, H*d_head) -> (B, S, H, d_head)
                    pre_proj = o_proj_inputs[layer_idx]
                    b, s, _ = pre_proj.shape
                    pre_proj_heads = pre_proj.view(b, s, num_heads, head_dim)
                    head_grads_per_head: list[torch.Tensor] = []
                    for head_idx in range(num_heads):
                        # scalar = ||pre_proj[:, -1, h, :]||_2 summed over batch
                        head_scalar = (
                            pre_proj_heads[:, -1, head_idx, :].norm(dim=-1).sum()
                        )
                        last_head = head_idx == num_heads - 1
                        head_retain = not (is_last_layer and last_head)
                        head_grad = torch.autograd.grad(
                            head_scalar,
                            inputs_embeds,
                            retain_graph=head_retain,
                        )[0]
                        # (B, S) grad norm over embedding dim
                        head_grads_per_head.append(head_grad.norm(dim=-1).detach())
                    # (H, B, S) -> (B, H, S)
                    per_layer_head_grads.append(
                        torch.stack(head_grads_per_head).permute(1, 0, 2),
                    )
            grad_norms = torch.stack(per_layer_grads)  # (num_layers, batch, seq_len)

            mask = attention_mask.to(dtype=grad_norms.dtype)
            batch_sum = (grad_norms * mask.unsqueeze(0)).sum(dim=1).detach()
            batch_count = attention_mask.sum(dim=0).detach()

            attribution_sum += batch_sum.to(dtype=attribution_sum.dtype)
            attribution_count += batch_count.to(dtype=attribution_count.dtype)

            if eval_config.enable_attribution_per_head:
                assert per_head_attribution_sum is not None
                # (num_layers, batch, num_heads, seq_len)
                head_grad_norms = torch.stack(per_layer_head_grads)
                # mask is (batch, seq_len); broadcast over layers and heads
                head_mask = mask.unsqueeze(0).unsqueeze(2)
                head_batch_sum = (head_grad_norms * head_mask).sum(dim=1).detach()
                per_head_attribution_sum += head_batch_sum.to(
                    dtype=per_head_attribution_sum.dtype,
                )
                del head_grad_norms, head_mask, head_batch_sum

            del (
                inputs,
                outputs,
                per_layer_grads,
                per_layer_head_grads,
                grad_norms,
                mask,
                batch_sum,
                batch_count,
            )
            layer_outputs.clear()
            o_proj_inputs.clear()
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
        for handle in hook_handles_attr:
            handle.remove()
        hook_handles_attr.clear()

        assert attribution_sum is not None
        assert attribution_count is not None

        logger.info("Saving attribution statistics")
        attribution_mean = (attribution_sum / attribution_count.unsqueeze(0)).cpu()
        per_head_attribution_mean: torch.Tensor | None = None
        if eval_config.enable_attribution_per_head:
            assert per_head_attribution_sum is not None
            per_head_attribution_mean = (
                per_head_attribution_sum / attribution_count.view(1, 1, -1)
            ).cpu()
        save_attribution_stats(
            attribution_mean,
            save_path,
            per_head_attribution=per_head_attribution_mean,
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
