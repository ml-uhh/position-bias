"""Configuration models for paper experiments."""

from pathlib import Path

import pydantic

from src.dataset import DatasetConfig, FineWebEduConfig
from src.metrics.metrics import NormType


class HFModelConfig(pydantic.BaseModel):
    """Hugging Face model loading configuration."""

    model_id: str
    cache_dir: Path | None = None
    max_memory: dict[int | str, str] | None = None
    device_map: dict[int | str, int | str] | str | None = None

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")


class WandBConfig(pydantic.BaseModel):
    """Optional experiment logging configuration."""

    project: str = "position-bias-reproduction"
    entity: str | None = None
    dir: Path | None = None
    id: str | None = None
    name: str | None = None
    notes: str | None = None
    tags: list[str] | None = None
    group: str | None = None
    job_type: str | None = None
    save_code: bool | None = None

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")


class EvalConfig(pydantic.BaseModel):
    """Configuration for model measurements."""

    batch_size: pydantic.PositiveInt = 1
    num_batches: pydantic.PositiveInt
    num_workers: pydantic.NonNegativeInt = 0
    sequence_length: pydantic.PositiveInt = 256
    save_path: Path = Path("./results")
    enable_attention_stats: bool = False
    attention_stats_norm: NormType = "frobenius"
    enable_qk_stats: bool = False
    enable_attribution: bool = False
    enable_attribution_per_head: bool = False
    disable_causal_mask: bool = False
    hf_model_config: HFModelConfig
    dataset_config: DatasetConfig = pydantic.Field(
        default_factory=lambda: FineWebEduConfig(dataset_name="fineweb-edu")
    )
    wandb_config: WandBConfig = pydantic.Field(default_factory=WandBConfig)
    run_name: str | None = None

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
