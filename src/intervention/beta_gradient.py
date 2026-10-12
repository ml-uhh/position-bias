r"""
measured input-gradient attribution under a fixed attention-scaling intervention.

For one beta condition, runs ``src.intervention.intervention.BetaScale(model, beta)`` around the
paper's input-gradient attribution recipe (``src.main._accumulate_attribution``), but against the
shared manifest's *fixed* target ``y(x)`` rather than a fresh argmax on the beta-modified model.
When ``beta == 1.0``, additionally verifies that this pipeline reproduces the unmodified model's
final-position logits (the doc's "beta=1 reproduces the unmodified model" acceptance criterion),
so every real beta=1.0 run is self-checking rather than relying only on a separate smoke test.

Run, e.g.::

    CUDA_VISIBLE_DEVICES=1 uv run --env-file .env \
        -m src.intervention.beta_gradient \
        --config config/empirical/alibi/mpt-7b-256.yaml \
        --manifest results/intervention/prompt_manifest.pt \
        --beta 0.8 --output-dir results/intervention/beta_0.8

"""

import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Annotated

import structlog
import torch
from torch import nn
from transformers import PreTrainedModel
import typer

from src.intervention.intervention import BetaScale, load_causal_model
from src.util.configs import EvalConfig
from src.util.pydantic import load_config

logger = structlog.get_logger()

DEFAULT_OUTPUT_DIR = Path("results/intervention/beta_1.0")
# Two separate forward calls under bf16 can differ slightly due to kernel nondeterminism (e.g.
# flash-attention); generous relative to logit magnitude, still tight enough to catch a real bug
# in BetaScale (e.g. scaling something other than the attention branch).
BETA_ONE_LOGIT_RTOL = 1e-2
BETA_ONE_LOGIT_ATOL = 1e-2


def _git_commit() -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 (fixed argv, resolved via shutil.which, no user input)
            [git, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def _package_versions() -> dict[str, str]:
    packages = ["torch", "transformers", "numpy", "datasets"]
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


def _process_prompt(
    model: PreTrainedModel,
    beta: float,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the raw gradient-norm profile and final-position logits for one prompt under `beta`.

    Args:
        model: The causal LM (see ``src.intervention.intervention`` for supported architectures).
        beta: Attention-branch scale factor.
        input_ids: Token ids, shape ``[1, n]``.
        attention_mask: Attention mask, shape ``[1, n]``.
        target_id: The manifest's fixed target token id for this prompt.

    Returns:
        Tuple ``(score, final_logits)``: ``score`` is the length-``n`` masked gradient-norm profile
        (float64, CPU); ``final_logits`` is the float32 CPU final-position logit vector.

    """
    embed = model.get_input_embeddings()
    assert isinstance(embed, nn.Embedding)
    device = embed.weight.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    inputs_embeds = embed(input_ids).detach()
    inputs_embeds.requires_grad_(requires_grad=True)

    with BetaScale(model, beta):
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)

    final_logits = outputs.logits[0, -1, :].float()
    probs = torch.softmax(final_logits, dim=-1)
    target_prob = probs[target_id]

    grad = torch.autograd.grad(target_prob, inputs_embeds)[0]
    grad_norm = grad.norm(dim=-1)  # [1, n]
    mask = attention_mask.to(dtype=grad_norm.dtype)
    score = (grad_norm * mask).squeeze(0).detach().to(torch.float64).cpu()

    return score, final_logits.detach().cpu()


def _verify_beta_one_matches_unmodified(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    beta_one_logits: torch.Tensor,
    prompt_idx: int,
) -> None:
    """Assert an unmodified (no BetaScale) forward pass reproduces beta=1's final logits."""
    embed = model.get_input_embeddings()
    assert isinstance(embed, nn.Embedding)
    device = embed.weight.device
    with torch.no_grad():
        unmodified_logits = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            use_cache=False,
        ).logits[0, -1, :].float().cpu()

    max_abs_diff = float((unmodified_logits - beta_one_logits).abs().max())
    assert torch.allclose(unmodified_logits, beta_one_logits, rtol=BETA_ONE_LOGIT_RTOL, atol=BETA_ONE_LOGIT_ATOL), (
        f"prompt {prompt_idx}: beta=1.0 does not reproduce the unmodified model "
        f"(max abs logit diff={max_abs_diff:.6f})"
    )
    logger.info(f"prompt {prompt_idx}: beta=1.0 self-check passed (max abs logit diff={max_abs_diff:.6f})")


def main(
    config: Annotated[Path, typer.Option("--config", help="Path to an EvalConfig YAML (n=256).")],
    manifest: Annotated[Path, typer.Option("--manifest", help="Path to prompt_manifest.pt.")],
    beta: float = 1.0,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
) -> None:
    """
    Compute measured input-gradient attribution under a fixed beta intervention, against fixed targets.

    Args:
        config: Path to an EvalConfig YAML (only ``hf_model_config``/``dataset_config`` are used).
        manifest: Path to the shared ``prompt_manifest.pt`` (from ``src.intervention.make_manifest``).
        beta: Attention-branch scale factor. If exactly ``1.0``, every prompt's beta-modified
            output is additionally checked against an unmodified forward pass.
        output_dir: Directory to save raw/aggregated gradient profiles, logits, and run metadata into.
        seed: Recorded in ``run_meta.json`` (no additional randomness is introduced here).

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    eval_config = load_config(config, EvalConfig)
    model, _tokenizer = load_causal_model(eval_config)

    # weights_only=False: the manifest is a self-produced artifact holding non-tensor fields
    # (prompt_ids: list[str], seed: int), not untrusted input.
    manifest_data = torch.load(manifest, map_location="cpu", weights_only=False)
    input_ids_all = manifest_data["input_ids"]
    attention_mask_all = manifest_data["attention_mask"]
    target_ids_all = manifest_data["target_ids"]
    prompt_ids = manifest_data["prompt_ids"]
    num_prompts = input_ids_all.shape[0]
    logger.info(f"Loaded manifest with {num_prompts} prompts", beta=beta)

    scores = []
    logits = []
    for idx in range(num_prompts):
        input_ids = input_ids_all[idx].unsqueeze(0)
        attention_mask = attention_mask_all[idx].unsqueeze(0)
        target_id = int(target_ids_all[idx].item())

        score, final_logits = _process_prompt(model, beta, input_ids, attention_mask, target_id)
        if beta == 1.0:
            _verify_beta_one_matches_unmodified(model, input_ids, attention_mask, final_logits, idx)

        scores.append(score)
        logits.append(final_logits)
        torch.cuda.empty_cache()
        logger.info(f"Processed prompt {idx + 1}/{num_prompts}")

    raw_gradient_profiles = torch.stack(scores)  # [N, n], float64
    gradient_profile = raw_gradient_profiles.mean(dim=0)
    gradient_profile = gradient_profile / gradient_profile.sum()
    final_logits_all = torch.stack(logits)  # [N, vocab], float32

    runtime_seconds = time.time() - start_time
    peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

    torch.save(raw_gradient_profiles, output_dir / "raw_gradient_profiles.pt")
    torch.save(gradient_profile, output_dir / "gradient_profile.pt")
    torch.save(final_logits_all, output_dir / "final_logits.pt")
    (output_dir / "prompt_ids.json").write_text(json.dumps(prompt_ids, indent=2))

    run_meta = {
        "beta": beta,
        "seed": seed,
        "num_prompts": num_prompts,
        "prompt_ids": prompt_ids,
        "model_id": eval_config.hf_model_config.model_id,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "git_commit": _git_commit(),
        "runtime_seconds": runtime_seconds,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "package_versions": _package_versions(),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    logger.info("Wrote beta_gradient outputs", output_dir=str(output_dir), beta=beta, runtime_seconds=runtime_seconds)


if __name__ == "__main__":
    typer.run(main)
