"""Utility functions for Pydantic configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pydantic
from pydantic_yaml import parse_yaml_file_as, to_yaml_file

if TYPE_CHECKING:
    from pathlib import Path


def check_is_yaml(path: Path) -> None:
    """
    Check if the file path has a YAML extension.

    Args:
        path: The file path to check.

    Raises:
        ValueError: If the path does not end with '.yaml' or '.yml'.

    """
    if path.suffix not in (".yaml", ".yml"):
        raise ValueError(f"Config path must end with '.yaml' or '.yml': {path}")


def load_config[T: pydantic.BaseModel](path: Path, config_type: type[T]) -> T:
    """
    Load a Pydantic configuration from a YAML file.

    Args:
        path: The path to the YAML file.
        config_type: The Pydantic model class.

    Returns:
        The loaded configuration object.

    Raises:
        RuntimeError: If loading fails.

    """
    check_is_yaml(path)
    try:
        with path.open(encoding="utf-8") as f:
            config = parse_yaml_file_as(config_type, f)
            assert type(config) is config_type
            return config
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {path}: {e}") from e


def save_config(path: Path, config: pydantic.BaseModel) -> None:
    """
    Save a Pydantic configuration to a YAML file.

    Args:
        path: The path to the YAML file.
        config: The configuration object to save.

    Raises:
        RuntimeError: If saving fails.

    """
    check_is_yaml(path)
    try:
        with path.open("w", encoding="utf-8") as f:
            to_yaml_file(f, config)
    except Exception as e:
        raise RuntimeError(f"Failed to save config to {path}: {e}") from e
