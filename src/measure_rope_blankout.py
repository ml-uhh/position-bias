"""Matched-prompt blank-out/probability-gradient validation for RoPE runs.

Aggregate raw scores across prompts before normalizing. No rollout fitting.
"""

import argparse
import csv
import hashlib
import json
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from src.blankout import blankout_profile


def probability_gradient(model, ids, mask, target):
    device = model.get_input_embeddings().weight.device
    embeddings = model.get_input_embeddings()(ids.to(device)).detach().requires_grad_(True)
    logits = model(inputs_embeds=embeddings, attention_mask=mask.to(device), use_cache=False).logits[:, -1].float()
    probabilities = logits.softmax(-1)
    if int(probabilities.argmax(-1).item()) != target:
        raise RuntimeError("Unperturbed target changed between blank-out and gradient forwards")
    gradient = torch.autograd.grad(probabilities[:, target].sum(), embeddings)[0]
    return gradient.float().norm(dim=-1)[0].detach().double().cpu()


def normalize(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or p.min() < 0 or p.sum() <= 0:
        raise ValueError("Invalid influence profile")
    return p/p.sum()


def w1(p, q):
    p, q = normalize(p), normalize(q)
    if p.shape != q.shape:
        raise ValueError("Profile lengths must match")
    return float(np.abs(np.cumsum(p-q)[:-1]).sum()/(len(p)-1))


def paper_rollout_profiles(saved):
    """Load the three manuscript rollouts from a paper rollout bundle."""
    if all(
        key in saved
        for key in (
            "profile_attention_only",
            "profile_fixed_half",
            "profile_three_norm_frobenius",
        )
    ):
        return {
            "attention_only": normalize(saved["profile_attention_only"]),
            "fixed_half": normalize(saved["profile_fixed_half"]),
            "adaptive": normalize(saved["profile_three_norm_frobenius"]),
        }
    if all(key in saved for key in ("profile_0_0", "profile_0_1", "profile_0_2")):
        variant_set = str(saved["variant_set"]) if "variant_set" in saved else ""
        if variant_set == "paper":
            return {
                "attention_only": normalize(saved["profile_0_2"]),
                "fixed_half": normalize(saved["profile_0_1"]),
                "adaptive": normalize(saved["profile_0_0"]),
            }
    if "no_residual_layer_varying" in saved and "layer_varying" in saved:
        profiles = {
            "attention_only": normalize(saved["no_residual_layer_varying"]),
            "adaptive": normalize(saved["layer_varying"]),
        }
        if "fixed_half" in saved:
            profiles["fixed_half"] = normalize(saved["fixed_half"])
        return profiles
    raise ValueError("Rollout bundle does not contain the manuscript variants")


def save_results(output, blank, gradients, ids, targets, meta, rollout_path):
    braw, graw = np.stack(blank), np.stack(gradients)
    b, g = normalize(braw.mean(0)), normalize(graw.mean(0))
    profiles = {"blankout": b, "matched_probability_gradient": g}
    if rollout_path is not None:
        with np.load(rollout_path) as saved:
            profiles.update(paper_rollout_profiles(saved))
            if "measured" in saved:
                profiles["archival_gradient_reference"] = normalize(saved["measured"])
    rows = [dict(profile=name, w1_to_blankout=w1(b,p), w1_to_matched_gradient=w1(g,p))
            for name,p in profiles.items()]
    np.savez_compressed(output/"blankout_profiles.npz", blankout_raw=braw,
                        probability_gradient_raw=graw, input_ids=np.stack(ids),
                        target_ids=np.asarray(targets), **profiles)
    with (output/"metrics.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    meta["completed_samples"] = len(blank)
    (output/"manifest.json").write_text(json.dumps(meta,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout-profiles", type=Path)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--position-batch-size", type=int, default=8)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Save resumable results after this many prompts; 0 disables checkpoints.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Require an empty output directory instead of resuming partial results.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        args.num_samples < 1
        or args.position_batch_size < 1
        or args.checkpoint_every < 0
    ):
        parser.error("Sample and batch counts must be positive")
    result_path = args.output_dir / "blankout_profiles.npz"
    manifest_path = args.output_dir / "manifest.json"
    output_has_files = args.output_dir.exists() and any(args.output_dir.iterdir())
    if output_has_files and args.no_resume:
        parser.error("--no-resume requires an empty output directory")
    if output_has_files and (not result_path.exists() or not manifest_path.exists()):
        parser.error(
            "Output directory is non-empty but has no complete resumable checkpoint",
        )
    from transformers import enable_full_determinism
    from src.main import load_model, setup_dataset
    from src.util.configs import EvalConfig
    from src.util.pydantic import load_config

    enable_full_determinism(args.seed, warn_only=True)
    config = load_config(args.eval_config, EvalConfig)
    if args.rollout_profiles is not None:
        with np.load(args.rollout_profiles) as saved:
            for key, profile in paper_rollout_profiles(saved).items():
                if profile.shape != (config.sequence_length,):
                    parser.error(f"Wrong sequence length in saved rollout: {key}")
    if config.enable_attention_stats or config.enable_qk_stats or config.enable_attribution:
        parser.error("Disable statistics and attribution hooks; gradients are measured explicitly")
    if config.disable_causal_mask:
        parser.error("This validation requires the unmodified causal model")
    config = config.model_copy(update={"batch_size":1,"num_workers":0,"num_batches":args.num_samples})
    model, tokenizer = load_model(config)
    loader = setup_dataset(config,tokenizer)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    meta = dict(protocol="zero embedding; fixed original argmax next-token logit; absolute logit change",
                gradient_target="original argmax next-token probability; input embedding L2 gradient norm",
                aggregation="normalize corpus mean of raw per-prompt scores",
                estimator_source_commit="05d172e60fec8cddfac1b44a037c8b454059d40a",
                config=config.model_dump(mode="json"), seed=args.seed,
                requested_samples=args.num_samples,position_batch_size=args.position_batch_size,
                rollout_profiles=str(args.rollout_profiles), completed=False,
                torch_version=torch.__version__, prompt_sha256=[])
    blanks, gradients, ids, targets = [], [], [], []
    if output_has_files:
        previous_meta = json.loads(manifest_path.read_text())
        previous_config = previous_meta.get("config", {})
        current_config = config.model_dump(mode="json")
        signature_fields = ("sequence_length", "hf_model_config", "dataset_config")
        if any(previous_config.get(k) != current_config.get(k) for k in signature_fields):
            parser.error("Existing checkpoint uses a different model, corpus, or length")
        if previous_meta.get("seed") != args.seed:
            parser.error("Existing checkpoint uses a different seed")
        with np.load(result_path) as saved:
            blanks = list(saved["blankout_raw"])
            gradients = list(saved["probability_gradient_raw"])
            ids = list(saved["input_ids"])
            targets = [int(value) for value in saved["target_ids"]]
        if len(blanks) > args.num_samples:
            parser.error("Existing checkpoint has more samples than requested")
        meta["prompt_sha256"] = previous_meta.get("prompt_sha256", [])
        meta["resumed_from_samples"] = len(blanks)
        print(f"Resuming from {len(blanks)} completed prompts", flush=True)
        for _ in islice(loader, len(blanks)):
            pass
    try:
        for sample in islice(loader,args.num_samples):
            token_ids, mask = sample["input_ids"], sample["attention_mask"]
            if not bool(mask.all()):
                raise ValueError("Use full-length unpadded prompts to match the paper protocol")
            b, target, _ = blankout_profile(model,token_ids,mask,args.position_batch_size)
            g = probability_gradient(model,token_ids,mask,target)
            blanks.append(b.numpy())
            gradients.append(g.numpy())
            ids.append(token_ids[0].cpu().numpy())
            targets.append(target)
            meta["prompt_sha256"].append(hashlib.sha256(ids[-1].astype("<i8").tobytes()).hexdigest())
            meta["elapsed_seconds"] = time.monotonic()-started
            print(f"{len(blanks)}/{args.num_samples}: {meta['elapsed_seconds']:.1f}s",flush=True)
            if args.checkpoint_every and len(blanks) % args.checkpoint_every == 0:
                save_results(args.output_dir,blanks,gradients,ids,targets,meta,args.rollout_profiles)
        if len(blanks) != args.num_samples:
            raise RuntimeError("Dataset ended before the requested sample count")
        meta["completed"] = True
    finally:
        if blanks:
            meta["elapsed_seconds"] = time.monotonic()-started
            save_results(args.output_dir,blanks,gradients,ids,targets,meta,args.rollout_profiles)


if __name__ == "__main__":
    main()
