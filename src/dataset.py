"""Dataset Configs."""

from abc import ABC
from typing import Annotated, Literal

import pydantic
import structlog

log = structlog.get_logger()


class BaseDatasetConfig(pydantic.BaseModel, ABC):
    """Abstract base class for dataset configurations."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")


class FineWebEduConfig(BaseDatasetConfig):
    """Configuration for downloading the FineWeb-Edu dataset."""

    dataset_name: Literal["fineweb-edu"] = pydantic.Field(
        description="Name identifier for the FineWeb-Edu dataset",
    )

    sample: Literal["sample-10BT", "sample-100BT", "sample-350BT"] = pydantic.Field(
        default="sample-10BT",
        description="Dataset sample to use (10BT, 100BT, or 350BT tokens)",
    )

    split: Literal["train"] = pydantic.Field(
        default="train",
        description="Dataset split to load",
    )

    repo_id: str = pydantic.Field(
        default="HuggingFaceFW/fineweb-edu",
        description="HuggingFace repository identifier for FineWeb-Edu",
    )
    revision: str = pydantic.Field(
        default="v1.4.0",
        description="Dataset version or git revision",
    )
    streaming: bool = pydantic.Field(
        default=True,
        description="Whether to load the dataset in streaming mode",
    )

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")


class DCLMBaselineConfig(BaseDatasetConfig):
    """Configuration for downloading the DCLM Baseline dataset."""

    dataset_name: Literal["dclm-baseline"] = pydantic.Field(
        description="Name identifier for the DCLM Baseline dataset",
    )

    sample: None = pydantic.Field(
        default=None,
        description="No sampling options for DCLM Baseline dataset",
    )
    split: Literal["train"] = pydantic.Field(
        default="train",
        description="Dataset split to load",
    )
    repo_id: str = pydantic.Field(
        default="mlfoundations/dclm-baseline-1.0",
        description="HuggingFace repository identifier for DCLM Baseline",
    )
    revision: None = pydantic.Field(
        default=None,
        description="No specific revision for DCLM Baseline dataset",
    )
    streaming: bool = pydantic.Field(
        default=True,
        description="Whether to load the dataset in streaming mode",
    )


class WikipediaConfig(BaseDatasetConfig):
    """Configuration for downloading the Wikipedia dataset."""

    dataset_name: Literal["wikipedia"] = pydantic.Field(
        default="wikipedia",
        description="Name identifier for the Wikipedia dataset",
    )

    sample: str = pydantic.Field(
        default="20231101.en",
        description="Dataset sample to use",
    )
    split: Literal["train"] = pydantic.Field(
        default="train",
        description="Dataset split to load",
    )
    repo_id: str = pydantic.Field(
        default="wikimedia/wikipedia",
        description="HuggingFace repository identifier for Wikipedia",
    )
    revision: None = pydantic.Field(
        default=None,
        description="Dataset version or git revision",
    )
    streaming: bool = pydantic.Field(
        default=True,
        description="Whether to load the dataset in streaming mode",
    )


DatasetConfig = Annotated[
    FineWebEduConfig | DCLMBaselineConfig | WikipediaConfig,
    pydantic.Field(discriminator="dataset_name"),
]
