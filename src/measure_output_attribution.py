"""Measure final-output next-token probability attribution over a corpus."""

import gc
import signal
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import enable_full_determinism
import typer

from src.main import close_dataloader_iterator, load_model, setup_dataset
from src.util.configs import EvalConfig
from src.util.pydantic import load_config


def output_probability_attribution(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-token gradient norms and predicted-token probabilities."""
    embed_device = model.get_input_embeddings().weight.device  # type: ignore[attr-defined]
    attention_mask = attention_mask.to(embed_device)
    inputs_embeds = model.get_input_embeddings()(input_ids.to(embed_device)).detach()  # type: ignore[attr-defined]
    inputs_embeds.requires_grad_(requires_grad=True)

    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )
    next_token_probabilities = torch.softmax(outputs.logits[:, -1, :].float(), dim=-1)
    predicted_probabilities = next_token_probabilities.max(dim=-1).values
    gradients = torch.autograd.grad(predicted_probabilities.sum(), inputs_embeds)[0]
    attribution = gradients.float().norm(dim=-1)
    return attribution.detach(), predicted_probabilities.detach()


def main(
    config_path: Path,
    output_dir: Path,
    seed: int = 42,
    checkpoint_every: int = 50,
    no_resume: bool = False,
) -> None:
    """Measure and save the corpus-mean final-output attribution profile."""
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be nonnegative")
    enable_full_determinism(seed, warn_only=True)
    config = load_config(path=config_path, config_type=EvalConfig)
    if config.enable_qk_stats or config.enable_attribution or config.enable_attention_stats:
        raise ValueError(
            "The output-attribution run must disable QK, hidden-state attribution, "
            "and attention-strength hooks.",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "output_attribution_checkpoint.pt"
    signature = {
        "model_id": config.hf_model_config.model_id,
        "dataset": config.dataset_config.model_dump(mode="json"),
        "sequence_length": config.sequence_length,
        "seed": seed,
    }
    attribution_sum = torch.zeros(config.sequence_length, dtype=torch.float64)
    attribution_count = torch.zeros(config.sequence_length, dtype=torch.int64)
    confidence_sum = 0.0
    sample_count = 0
    batches_processed = 0
    if checkpoint_path.exists() and not no_resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if saved.get("signature") != signature:
            raise ValueError(f"Checkpoint does not match this run: {checkpoint_path}")
        attribution_sum = saved["attribution_sum"]
        attribution_count = saved["attribution_count"]
        confidence_sum = float(saved["confidence_sum"])
        sample_count = int(saved["sample_count"])
        batches_processed = int(saved["batches_processed"])
        print(
            f"Resuming from {batches_processed} batches and {sample_count} samples",
            flush=True,
        )

    def save_checkpoint() -> None:
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "signature": signature,
                "attribution_sum": attribution_sum,
                "attribution_count": attribution_count,
                "confidence_sum": confidence_sum,
                "sample_count": sample_count,
                "batches_processed": batches_processed,
            },
            temporary,
        )
        temporary.replace(checkpoint_path)

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; checkpointing after the current batch.", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    model, tokenizer = load_model(config)
    dataloader = setup_dataset(config, tokenizer)
    data_iterator = iter(dataloader)
    try:
        for _ in tqdm(range(batches_processed), desc="Skipping completed samples"):
            try:
                next(data_iterator)
            except StopIteration:
                break

        for batch_index in tqdm(
            range(batches_processed, config.num_batches),
            initial=batches_processed,
            total=config.num_batches,
            desc="Output attribution",
        ):
            try:
                inputs = next(data_iterator)
            except StopIteration:
                break
            attribution, confidence = output_probability_attribution(
                model,
                inputs["input_ids"],
                inputs["attention_mask"],
            )
            mask = inputs["attention_mask"].to(
                attribution.device,
                dtype=attribution.dtype,
            )
            attribution_sum += (attribution * mask).sum(dim=0).cpu().to(torch.float64)
            attribution_count += inputs["attention_mask"].sum(dim=0).cpu().to(
                torch.int64,
            )
            confidence_sum += confidence.sum().item()
            sample_count += confidence.numel()
            batches_processed = batch_index + 1
            if checkpoint_every and batches_processed % checkpoint_every == 0:
                save_checkpoint()
            if stop_requested:
                save_checkpoint()
                print(
                    f"Stopped after {batches_processed} batches and {sample_count} "
                    f"samples; resume state saved to {checkpoint_path}",
                    flush=True,
                )
                return
    finally:
        close_dataloader_iterator(data_iterator)
        del data_iterator
        del dataloader
        gc.collect()

    if batches_processed != config.num_batches:
        raise RuntimeError(
            f"Dataset ended after {batches_processed} of {config.num_batches} batches",
        )
    if sample_count == 0 or torch.any(attribution_count == 0):
        raise RuntimeError("No valid samples were measured for one or more positions")

    attribution_mean = attribution_sum / attribution_count
    torch.save(attribution_mean, output_dir / "output_prob_input_grad_l2_mean.pt")
    np.save(
        output_dir / "output_prob_input_grad_l2_mean.npy",
        attribution_mean.numpy(),
    )
    np.savetxt(
        output_dir / "output_prob_input_grad_l2_mean.csv",
        np.column_stack((np.arange(config.sequence_length), attribution_mean.numpy())),
        delimiter=",",
        header="token_position,attribution",
        comments="",
    )
    print(
        f"saved {sample_count} samples to {output_dir}; "
        f"mean predicted-token probability={confidence_sum / sample_count:.8f}",
    )
    checkpoint_path.unlink(missing_ok=True)


if __name__ == "__main__":
    typer.run(main)
