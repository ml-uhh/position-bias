"""Run the complete paper reproduction pipeline from one manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "config" / "paper.json"
RESULTS = ROOT / "results"
RUN_CONFIGS = RESULTS / "run_configs"


def load_environment(path: Path) -> None:
    """Load simple KEY=VALUE entries without replacing exported variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_environment(ROOT / ".env")


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def select(values: str, available: list[str], label: str) -> list[str]:
    if values == "all":
        return available
    selected = [value.strip() for value in values.split(",") if value.strip()]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise SystemExit(f"Unknown {label}: {', '.join(unknown)}")
    return selected


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())


def evaluation_config(
    manifest: dict[str, Any],
    model: str,
    dataset: str,
    sequence_length: int,
    prompts: int,
) -> Path:
    model_spec = manifest["models"][model]
    dataset_spec = manifest["datasets"][dataset]
    cache_dir = str(Path(os.environ.get("HF_HOME", ROOT / ".cache" / "huggingface")))
    payload = {
        "num_batches": prompts,
        "batch_size": 1,
        "num_workers": 0,
        "save_path": str(RESULTS),
        "sequence_length": sequence_length,
        "hf_model_config": {
            "model_id": model_spec["model_id"],
            "cache_dir": cache_dir,
            "device_map": "auto",
        },
        "dataset_config": {
            "dataset_name": dataset,
            "repo_id": dataset_spec["repo_id"],
            "sample": dataset_spec["sample"],
            "revision": dataset_spec["revision"],
            "split": dataset_spec["split"],
            "streaming": True,
        },
        "wandb_config": {"project": "position-bias-reproduction"},
        "enable_attention_stats": False,
        "enable_attribution": False,
        "enable_qk_stats": False,
    }
    output = RUN_CONFIGS / dataset / str(sequence_length) / f"{model}.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def experiment_dir(dataset: str, sequence_length: int, model: str) -> Path:
    return RESULTS / dataset / str(sequence_length) / model


def attribution(
    manifest: dict[str, Any], model: str, dataset: str, length: int, prompts: int
) -> Path:
    config = evaluation_config(manifest, model, dataset, length, prompts)
    output = experiment_dir(dataset, length, model) / "attribution"
    profile = output / "output_prob_input_grad_l2_mean.npy"
    if not profile.exists():
        run(
            [
                sys.executable,
                "-m",
                "src.measure_output_attribution",
                str(config),
                str(output),
                "--seed",
                str(manifest["seed"]),
                "--checkpoint-every",
                "25",
            ]
        )
    return profile


def rollout(
    manifest: dict[str, Any], model: str, dataset: str, length: int, prompts: int
) -> Path:
    spec = manifest["models"][model]
    config = evaluation_config(manifest, model, dataset, length, prompts)
    measured = attribution(manifest, model, dataset, length, prompts)
    output = experiment_dir(dataset, length, model) / "rollout"
    output.mkdir(parents=True, exist_ok=True)
    if spec["family"] == "alibi":
        result = output / "alibi_three_norm_results.npz"
        rollout_config = Path(spec["rollout_config"])
        if length == 2048:
            rollout_config = rollout_config.parent / "2048" / rollout_config.name
        if not result.exists():
            run(
                [
                    sys.executable,
                    "-m",
                    "scripts.measure_alibi_three_norm_lambda",
                    "--eval-config",
                    str(config),
                    "--rollout-config",
                    str(rollout_config),
                    "--measured-profile",
                    str(measured),
                    "--output-dir",
                    str(output),
                    "--num-batches",
                    str(prompts),
                ]
            )
        return result

    result = output / "forward_source_profiles.npz"
    if not result.exists():
        run(
            [
                sys.executable,
                "-m",
                "scripts.measure_forward_source_rollout",
                "--eval-config",
                str(config),
                "--measured-profile",
                str(measured),
                "--output-dir",
                str(output),
                "--num-batches",
                str(prompts),
                "--variant-set",
                "paper",
                "--checkpoint-every",
                "25",
            ]
        )
    return result


def run_matrix(
    stage: str,
    manifest: dict[str, Any],
    models: list[str],
    datasets: list[str],
    prompts_override: int | None,
) -> None:
    section = manifest[stage]
    length = int(section["sequence_length"])
    prompts = prompts_override or int(section["prompts"])
    for dataset in datasets:
        for model in models:
            rollout(manifest, model, dataset, length, prompts)


def run_blankout(
    manifest: dict[str, Any], models: list[str], prompts_override: int | None
) -> None:
    section = manifest["blankout"]
    length = int(section["sequence_length"])
    prompts = prompts_override or int(section["prompts"])
    for dataset in section["datasets"]:
        for model in models:
            config = evaluation_config(manifest, model, dataset, length, prompts)
            profiles = rollout(manifest, model, dataset, length, manifest["main"]["prompts"])
            output = experiment_dir(dataset, length, model) / "blankout"
            if not (output / "metrics.csv").exists():
                run(
                    [
                        sys.executable,
                        "-m",
                        "src.measure_rope_blankout",
                        "--eval-config",
                        str(config),
                        "--output-dir",
                        str(output),
                        "--rollout-profiles",
                        str(profiles),
                        "--num-samples",
                        str(prompts),
                        "--checkpoint-every",
                        "25",
                    ]
                )


def run_approximations(
    manifest: dict[str, Any], models: list[str], datasets: list[str], prompts: int | None
) -> None:
    count = prompts or int(manifest["main"]["prompts"])
    for dataset in datasets:
        for model in models:
            bundle = rollout(manifest, model, dataset, 256, count)
            if manifest["models"][model]["family"] == "rope":
                output = bundle.parent / "approximation_ladder_metrics.csv"
                if not output.exists():
                    run([sys.executable, "-m", "scripts.compute_approximation_ladder", str(bundle.parent)])


def run_mixing_ablation(
    manifest: dict[str, Any], models: list[str], prompts_override: int | None
) -> None:
    dataset = "fineweb-edu"
    length = 256
    count = prompts_override or int(manifest["main"]["prompts"])
    for model in models:
        spec = manifest["models"][model]
        if spec["family"] == "alibi":
            rollout(manifest, model, dataset, length, count)
            continue
        config = evaluation_config(manifest, model, dataset, length, count)
        measured = attribution(manifest, model, dataset, length, count)
        output = experiment_dir(dataset, length, model) / "rollout_diagnostic"
        result = output / "forward_source_profiles.npz"
        if not result.exists():
            run(
                [
                    sys.executable,
                    "-m",
                    "scripts.measure_forward_source_rollout",
                    "--eval-config",
                    str(config),
                    "--measured-profile",
                    str(measured),
                    "--output-dir",
                    str(output),
                    "--num-batches",
                    str(count),
                    "--variant-set",
                    "diagnostic",
                    "--checkpoint-every",
                    "25",
                ]
            )


def run_monotonicity(
    manifest: dict[str, Any], models: list[str], datasets: list[str], prompts: int | None
) -> None:
    count = prompts or int(manifest["main"]["prompts"])
    for dataset in datasets:
        for model in models:
            spec = manifest["models"][model]
            command = [sys.executable, "-m", "scripts.analyze_kernel_monotonicity"]
            if spec["family"] == "rope":
                config = evaluation_config(manifest, model, dataset, 256, count)
                output_dir = (
                    experiment_dir(dataset, 256, model)
                    / "monotonicity"
                    / f"samples-{count}"
                )
                bundle = output_dir / "transport_kernels.npz"
                if not bundle.exists():
                    run(
                        [
                            sys.executable,
                            "-m",
                            "scripts.measure_forward_source_rollout",
                            "--eval-config",
                            str(config),
                            "--output-dir",
                            str(output_dir),
                            "--num-batches",
                            str(count),
                            "--variant-set",
                            "paper",
                            "--kernel-only",
                            "--checkpoint-every",
                            "25",
                        ]
                    )
                output = output_dir / "monotonicity.csv"
                command.extend(
                    [
                        "--profiles",
                        str(bundle),
                        "--kernel-key",
                        "mean_forward_alti_attention",
                    ]
                )
            else:
                output = experiment_dir(dataset, 256, model) / "monotonicity.csv"
                command.extend(["--rollout-config", spec["rollout_config"]])
            command.extend(["--output", str(output)])
            run(command)


def run_intervention(
    manifest: dict[str, Any], models: list[str], prompts_override: int | None
) -> None:
    section = manifest["intervention"]
    dataset = section["dataset"]
    length = int(section["sequence_length"])
    prompts = prompts_override or int(section["prompts"])
    betas = [float(beta) for beta in section["betas"]]
    for model in models:
        spec = manifest["models"][model]
        config = evaluation_config(manifest, model, dataset, length, prompts)
        root = RESULTS / "intervention" / model
        manifest_file = root / "prompt_manifest.pt"
        branch_stats = root / "baseline_branch_stats.pt"
        if not manifest_file.exists():
            run(
                [
                    sys.executable,
                    "-m",
                    "src.intervention.make_manifest",
                    "--config",
                    str(config),
                    "--num-prompts",
                    str(prompts),
                    "--seed",
                    str(manifest["seed"]),
                    "--output",
                    str(manifest_file),
                ]
            )
        if not branch_stats.exists():
            run(
                [
                    sys.executable,
                    "-m",
                    "src.intervention.shapley_baseline",
                    "--config",
                    str(config),
                    "--manifest",
                    str(manifest_file),
                    "--output",
                    str(branch_stats),
                ]
            )
        for beta in betas:
            beta_dir = root / f"beta_{beta}"
            if not (beta_dir / "gradient_profile.pt").exists():
                run(
                    [
                        sys.executable,
                        "-m",
                        "src.intervention.beta_gradient",
                        "--config",
                        str(config),
                        "--manifest",
                        str(manifest_file),
                        "--beta",
                        str(beta),
                        "--output-dir",
                        str(beta_dir),
                        "--seed",
                        str(manifest["seed"]),
                    ]
                )
        if spec["family"] == "alibi":
            run(
                [
                    sys.executable,
                    "-m",
                    "src.intervention.beta_report",
                    "--root",
                    str(root),
                    "--baseline-rollout-config",
                    spec["rollout_config"],
                    "--branch-stats",
                    str(branch_stats),
                ]
            )
        else:
            kernels = rollout(manifest, model, dataset, length, manifest["main"]["prompts"])
            run(
                [
                    sys.executable,
                    "-m",
                    "src.intervention.rope_beta_report",
                    "--root",
                    str(root),
                    "--kernel-npz",
                    str(kernels),
                    "--branch-stats",
                    str(branch_stats),
                ]
            )


def download(manifest: dict[str, Any], models: list[str], datasets: list[str]) -> None:
    run(
        [
            sys.executable,
            "-m",
            "scripts.download_assets",
            "--models",
            ",".join(models),
            "--datasets",
            ",".join(datasets),
        ]
    )


def check(manifest: dict[str, Any]) -> None:
    for model in manifest["models"]:
        for dataset in manifest["datasets"]:
            evaluation_config(manifest, model, dataset, 256, 1)
    run([sys.executable, "-m", "pytest", "-q"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--models", default="all")
    download_parser.add_argument("--datasets", default="all")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "stage",
        choices=(
            "main",
            "context",
            "blankout",
            "approximations",
            "mixing",
            "monotonicity",
            "intervention",
        ),
    )
    run_parser.add_argument("--models", default="all")
    run_parser.add_argument("--datasets", default="all")
    run_parser.add_argument("--prompts", type=int)
    subparsers.add_parser("render")
    args = parser.parse_args()
    manifest = load_manifest()
    all_models = list(manifest["models"])
    all_datasets = list(manifest["datasets"])
    if args.command == "check":
        check(manifest)
        return
    if args.command == "render":
        run([sys.executable, "-m", "scripts.render_paper_outputs"])
        return
    models = select(args.models, all_models, "models")
    datasets = select(args.datasets, all_datasets, "datasets")
    if args.command == "download":
        download(manifest, models, datasets)
    elif args.stage in ("main", "context"):
        allowed = manifest[args.stage]["datasets"]
        chosen = [dataset for dataset in datasets if dataset in allowed]
        run_matrix(args.stage, manifest, models, chosen, args.prompts)
    elif args.stage == "blankout":
        run_blankout(manifest, models, args.prompts)
    elif args.stage == "approximations":
        run_approximations(manifest, models, datasets, args.prompts)
    elif args.stage == "mixing":
        run_mixing_ablation(manifest, models, args.prompts)
    elif args.stage == "monotonicity":
        run_monotonicity(manifest, models, datasets, args.prompts)
    elif args.stage == "intervention":
        permitted = set(manifest["intervention"]["models"])
        run_intervention(manifest, [model for model in models if model in permitted], args.prompts)


if __name__ == "__main__":
    main()
