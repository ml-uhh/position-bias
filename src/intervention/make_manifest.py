r"""
build the shared prompt manifest with fixed targets y(x).

For each prompt, records the unmodified (beta=1, no intervention) model's argmax next-token
target ``y(x) = argmax_v z_v(x)`` at the final position. This target is fixed once here and reused
unchanged by every beta condition in ``src.intervention.beta_gradient`` -- Q3's no-refit design
requires the *same* prompts and the *same* target IDs across beta=0.8/1.0/1.2.

Run, e.g.::

    CUDA_VISIBLE_DEVICES=0 uv run --env-file .env \
        -m src.intervention.make_manifest \
        --config config/empirical/alibi/mpt-7b-256.yaml \
        --num-prompts 128 --seed 42 \
        --output results/intervention/prompt_manifest.pt

"""

from pathlib import Path
from typing import Annotated
import itertools

import datasets
import structlog
import torch
from torch import nn
import typer

from src.dataset_wrapper import DatasetWrapper
from src.intervention.intervention import load_causal_model
from src.util.configs import EvalConfig
from src.util.pydantic import load_config

logger = structlog.get_logger()

DEFAULT_OUTPUT = Path("results/intervention/prompt_manifest.pt")


def load_prompts(eval_config: EvalConfig, tokenizer, num_prompts: int) -> list[dict[str, torch.Tensor]]:
    """Load the first length-filtered prompts from the configured streaming dataset."""
    dataset_config = eval_config.dataset_config
    dataset = datasets.load_dataset(
        dataset_config.repo_id,
        name=dataset_config.sample,
        revision=dataset_config.revision,
        split=dataset_config.split,
        streaming=dataset_config.streaming,
    )
    wrapped = DatasetWrapper(dataset, tokenizer, seq_length=eval_config.sequence_length)
    return list(itertools.islice(iter(wrapped), num_prompts))


def main(
    config: Annotated[Path, typer.Option("--config", help="Path to an EvalConfig YAML (n=256).")],
    num_prompts: int = 128,
    seed: int = 42,
    output: Path = DEFAULT_OUTPUT,
) -> None:
    """
    Build the shared prompt manifest: tokenized prompts plus each one's fixed baseline target y(x).

    Args:
        config: Path to an EvalConfig YAML (only ``hf_model_config``/``dataset_config``/
            ``sequence_length`` are used).
        num_prompts: Number of prompts to include.
        seed: Random seed (recorded for reproducibility; prompt order itself comes from
            ``load_prompts``'s deterministic dataset iteration).
        output: Path to save the manifest ``.pt`` file to.

    """
    torch.manual_seed(seed)
    output.parent.mkdir(parents=True, exist_ok=True)

    eval_config = load_config(config, EvalConfig)
    model, tokenizer = load_causal_model(eval_config)
    prompts = load_prompts(eval_config, tokenizer, num_prompts)
    logger.info(f"Loaded {len(prompts)} prompts")

    embed = model.get_input_embeddings()
    assert isinstance(embed, nn.Embedding)
    device = embed.weight.device
    input_ids_list = []
    attention_mask_list = []
    target_ids_list = []
    prompt_ids = []

    with torch.no_grad():
        for idx, prompt in enumerate(prompts):
            input_ids = prompt["input_ids"].unsqueeze(0)  # type: ignore[union-attr]
            attention_mask = prompt["attention_mask"].unsqueeze(0)  # type: ignore[union-attr]
            prompt_ids.append(str(prompt.get("sample_id", f"prompt:{idx}")))

            logits = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                use_cache=False,
            ).logits[0, -1, :].float()
            target_ids_list.append(int(logits.argmax().item()))
            input_ids_list.append(input_ids[0].cpu())
            attention_mask_list.append(attention_mask[0].cpu())
            logger.info(f"Processed prompt {idx + 1}")

    manifest = {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "target_ids": torch.tensor(target_ids_list, dtype=torch.long),
        "prompt_ids": prompt_ids,
        "seed": seed,
    }
    torch.save(manifest, output)
    logger.info("Wrote prompt manifest", path=str(output), num_prompts=len(prompts))


if __name__ == "__main__":
    typer.run(main)
