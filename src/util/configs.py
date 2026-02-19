"""Configuration classes for model and evaluation."""

from pathlib import Path

import pydantic

from src.dataset import DatasetConfig, FineWebEduConfig


class HFModelConfig(pydantic.BaseModel):
    """Configuration for the model."""

    model_id: str = pydantic.Field(
        default="tiiuae/falcon-rw-1b",
        description="The model ID to use from the Hugging Face Hub.",
    )
    cache_dir: pydantic.DirectoryPath | None = pydantic.Field(
        default=None,
        description="Directory to cache the model files.",
    )
    max_memory: dict[int | str, str] | None = pydantic.Field(
        default=None,
        description="Maximum memory to use for each device.",
    )
    device_map: dict[int | str, int | str] | str | None = pydantic.Field(
        default=None,
        description="Device map for model placement (e.g., 'auto' or a dict).",
    )

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")


class WandBConfig(pydantic.BaseModel):
    """Configuration for Weights & Biases logging."""

    project: str = pydantic.Field(
        default="position-bias",
        description="WandB project name",
    )
    entity: str | None = pydantic.Field(
        default=None,
        description="WandB entity (team) name",
    )
    dir: Path | None = pydantic.Field(
        default=None,
        description="Save directory for WandB files",
    )
    id: str | None = pydantic.Field(
        default=None,
        description="WandB run ID (for resuming)",
    )
    name: str | None = pydantic.Field(default=None, description="WandB run name")
    notes: str | None = pydantic.Field(
        default=None,
        description="Detailed notes on a run",
    )
    tags: list[str] | None = pydantic.Field(
        default=None,
        description="List of tags for organization",
    )
    group: str | None = pydantic.Field(
        default=None,
        description="WandB group name (for grouping runs)",
    )
    job_type: str | None = pydantic.Field(
        default=None,
        description="WandB job type (for organizing within groups)",
    )
    save_code: bool | None = pydantic.Field(
        default=None,
        description="Whether to save code files to WandB",
    )

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")


class EvalConfig(pydantic.BaseModel):
    """Configuration for evaluation."""

    batch_size: pydantic.PositiveInt = pydantic.Field(
        description="Batch size for evaluation.",
    )
    num_batches: pydantic.PositiveInt = pydantic.Field(
        description="Number of batches to evaluate.",
    )
    num_workers: pydantic.NonNegativeInt = pydantic.Field(
        default=0,
        description="Number of DataLoader workers. Set to 0 for streaming datasets, >0 for local datasets.",
    )
    sequence_length: pydantic.PositiveInt = pydantic.Field(
        default=1024,
        description="Sequence length (in tokens) for model inputs.",
    )
    save_path: Path = pydantic.Field(
        default=Path("./results"),
        description="Path to save evaluation results.",
    )
    enable_attention_stats: bool = pydantic.Field(
        default=False,
        description="Whether to register attention hooks and log attention/residual statistics.",
    )
    enable_qk_stats: bool = pydantic.Field(
        default=False,
        description="Whether to wrap attention layers and save QK statistics.",
    )
    enable_attribution: bool = pydantic.Field(
        default=False,
        description="Whether to compute input-gradient attribution.",
    )
    hf_model_config: HFModelConfig = pydantic.Field(
        default_factory=HFModelConfig,
        description="The model configuration.",
    )
    dataset_config: DatasetConfig = pydantic.Field(
        default_factory=FineWebEduConfig,  # type: ignore[arg-type]
        description="The dataset configuration.",
    )
    wandb_config: WandBConfig = pydantic.Field(
        default_factory=WandBConfig,
        description="Configuration for Weights & Biases logging",
    )

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
