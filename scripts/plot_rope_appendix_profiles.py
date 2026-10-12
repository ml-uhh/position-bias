# ruff: noqa: INP001
"""Export RoPE profiles using the original ALiBi paper plotting routine."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import scienceplots  # noqa: F401


MODELS = (
    ("smollm3-3b", "HuggingFaceTB_SmolLM3-3B-Base"),
    ("qwen2.5-7b", "Qwen_Qwen2.5-7B"),
    ("qwen3-8b", "Qwen_Qwen3-8B-Base"),
    ("qwen2.5-32b", "Qwen_Qwen2.5-32B"),
    ("qwen2.5-72b", "Qwen_Qwen2.5-72B"),
)

VARIANTS = (
    ("noresidual", "attention", False),
    ("fixed-half", "half", False),
    ("adaptive", "adaptive", False),
    ("measured", "measured", True),
)


def _save_paper_profile(
    profile: np.ndarray,
    output_dir: Path,
    *,
    measured: bool = False,
) -> None:
    """Use the original individual-panel style from the ALiBi figures."""
    plt.style.use(["science", "bright", "grid", "no-latex"])
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
            "mathtext.fontset": "dejavusans",
        },
    )
    positions = np.arange(profile.size)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=400)
    (line,) = ax.plot(positions, profile, marker="o", linewidth=1.5)
    if measured:
        line.set_color(plt.rcParams["axes.prop_cycle"].by_key()["color"][4])
    ax.set_xlabel("Token position", fontsize=14)
    ax.set_ylabel(r"$\hat{p}^{(T)}(j)$" if measured else r"$p^{(T)}(j)$", fontsize=14)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.margins(x=0.02)
    ax.grid(visible=True, which="major", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "last_row_distribution.pdf", bbox_inches="tight", dpi=400)
    fig.savefig(output_dir / "last_row_distribution.png", bbox_inches="tight", dpi=400)
    np.savetxt(
        output_dir / "last_row_distribution.csv",
        np.column_stack((positions, profile)),
        delimiter=",",
        header="token_position,probability",
        comments="",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.profiles) as profiles:
        for model, key_prefix in MODELS:
            for variant, key_suffix, measured in VARIANTS:
                output_dir = args.output_dir / f"{model}-{variant}"
                output_dir.mkdir(parents=True, exist_ok=True)
                _save_paper_profile(
                    profiles[f"{key_prefix}_{key_suffix}"],
                    output_dir,
                    measured=measured,
                )


if __name__ == "__main__":
    main()
