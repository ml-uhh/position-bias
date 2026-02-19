"""Lambda schedule utilities for rollout analysis."""

from typing import Annotated, Literal

from jaxtyping import Float
import numpy as np
from numpy.typing import NDArray
import pydantic

LambdaScheduleType = Literal["constant", "power_decay", "custom"]


class BaseLambdaScheduleConfig(pydantic.BaseModel):
    """Base configuration for lambda schedule."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")


class ConstantLambdaScheduleConfig(BaseLambdaScheduleConfig):
    """Constant lambda schedule configuration."""

    schedule_type: Literal["constant"] = "constant"
    lambda_const: float = pydantic.Field(
        ge=0.0,
        le=1.0,
        description="Constant lambda value",
    )


class PowerDecayLambdaScheduleConfig(BaseLambdaScheduleConfig):
    """Power decay lambda schedule configuration."""

    schedule_type: Literal["power_decay"] = "power_decay"
    lambda_0: pydantic.PositiveFloat = pydantic.Field(
        description="Initial lambda value at layer t=0",
    )
    beta: pydantic.PositiveFloat = pydantic.Field(
        description="Decay exponent",
    )


class CustomLambdaScheduleConfig(BaseLambdaScheduleConfig):
    """Custom lambda schedule configuration."""

    schedule_type: Literal["custom"] = "custom"
    lambdas_custom: list[float] = pydantic.Field(
        description="Custom lambda schedule of length num_layers",
    )


LambdaScheduleConfig = Annotated[
    ConstantLambdaScheduleConfig
    | PowerDecayLambdaScheduleConfig
    | CustomLambdaScheduleConfig,
    pydantic.Field(discriminator="schedule_type"),
]


def make_lambda_schedule(
    config: LambdaScheduleConfig,
    num_layers: int,
) -> Float[NDArray, "num_layers"]:
    """
    Construct lambda_t for t=0..T.

    Supports three schedule types:
    - constant:     lambda_t = lambda_const
    - power_decay:  lambda_t = lambda_0 / (t+1)^beta
    - custom:       user-supplied list/array length num_layers

    Args:
        config: Lambda schedule configuration specifying the schedule type and params.
        num_layers: Number of layers.

    Returns:
        Array of lambda values with values clipped to [0, 1].

    """
    match config:
        case ConstantLambdaScheduleConfig():
            lambdas = np.full(num_layers, config.lambda_const)
        case PowerDecayLambdaScheduleConfig():
            t = np.arange(num_layers, dtype=float)
            lambdas = config.lambda_0 / (t + 1.0) ** config.beta
        case CustomLambdaScheduleConfig():
            if len(config.lambdas_custom) != num_layers:
                raise ValueError(
                    f"lambdas_custom must have length num_layers ({num_layers}), got {len(config.lambdas_custom)}",
                )

            lambdas = np.array(config.lambdas_custom)
        case _:
            raise TypeError("Unsupported lambda schedule config")

    lambdas = np.clip(lambdas, 0.0, 1.0)

    return lambdas
