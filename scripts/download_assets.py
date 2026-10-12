"""Download public model snapshots and initialize dataset caches."""

from __future__ import annotations

import argparse
from itertools import islice
import json
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

from reproduce import load_environment


ROOT = Path(__file__).resolve().parents[1]
load_environment(ROOT / ".env")


def parse_selection(value: str, available: list[str], label: str) -> list[str]:
    selected = available if value == "all" else value.split(",")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise SystemExit(f"Unknown {label}: {', '.join(unknown)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--dataset-examples", type=int, default=1)
    args = parser.parse_args()

    manifest = json.loads((ROOT / "config" / "paper.json").read_text(encoding="utf-8"))
    models = parse_selection(args.models, list(manifest["models"]), "models")
    datasets = parse_selection(args.datasets, list(manifest["datasets"]), "datasets")
    cache_dir = Path(os.environ.get("HF_HOME", ROOT / ".cache" / "huggingface"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    for name in models:
        repo_id = manifest["models"][name]["model_id"]
        print(f"Downloading model snapshot: {name} ({repo_id})", flush=True)
        snapshot_download(repo_id=repo_id, cache_dir=cache_dir)

    for name in datasets:
        spec = manifest["datasets"][name]
        print(f"Initializing dataset cache: {name} ({spec['repo_id']})", flush=True)
        stream = load_dataset(
            spec["repo_id"],
            name=spec["sample"],
            revision=spec["revision"],
            split=spec["split"],
            streaming=True,
        )
        list(islice(stream, args.dataset_examples))


if __name__ == "__main__":
    main()
