"""Utility functions for rollout analysis and visualization."""

from pathlib import Path

from jaxtyping import Float
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
from numpy.typing import NDArray


def style_plot(ax, title=None):  # type: ignore[no-untyped-def] # noqa: D103, ANN201, ANN001
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import scienceplots  # noqa: F401, PLC0415

    plt.style.use(["science", "bright", "grid", "no-latex"])
    # Use sans-serif fonts (incl. mathtext) for cleaner small-scale rendering.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
            "mathtext.fontset": "dejavusans",
        },
    )
    ax.set_xlabel("Token position", fontsize=14)
    ax.set_ylabel("$p^{{(T)}}(j)$", fontsize=14)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
    ax.tick_params(axis="both", which="major", labelsize=11)
    if title:
        ax.set_title(title)
    ax.margins(x=0.02)
    ax.grid(True, which="major", alpha=0.2)  # noqa: FBT003


def plot_lambda_schedule(
    lambdas: Float[NDArray, "num_layers"],
    *,
    save_dir: Path | None = None,
) -> None:
    """
    Plot the lambda schedule over layers.

    Produces a matplotlib figure of the lambda values across layers and either displays it or saves it to the provided directory.

    Args:
        lambdas: 1D array of lambda values indexed by layer.
        save_dir: Optional directory to save the plot and CSV data.

    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import scienceplots  # noqa: F401, PLC0415

    plt.style.use(["science", "bright", "grid", "no-latex"])
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
            "mathtext.fontset": "dejavusans",
        },
    )
    fig, ax = plt.subplots(figsize=(4, 3), dpi=400)
    ax.plot(lambdas, marker="o", linewidth=1.5)
    style_plot(ax)  # type: ignore[no-untyped-call]
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"$\lambda_t$")
    fig.tight_layout()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "lambda_schedule.pdf", bbox_inches="tight", dpi=400)

        np.savetxt(
            save_dir / "lambda_schedule.csv",
            lambdas,
            delimiter=",",
            header="lambda",
            comments="",
        )

        plt.close(fig)
    else:
        plt.show()


def plot_last_row_distribution(
    scores_product: Float[NDArray, "sequence_length sequence_length"],
    *,
    save_dir: Path | None = None,
) -> None:
    """
    Plot the distribution over token positions in the last row of the scores_product matrix.

    Normalizes the final row of the matrix to a probability distribution and plots the resulting values versus token position.

    Args:
        scores_product: 2D scores product matrix.
        save_dir: Optional directory to save the plot and CSV data.

    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sequence_length, _ = scores_product.shape

    row = scores_product[-1].copy()
    s = row.sum()
    if s > 0:
        row /= s

    x = np.arange(sequence_length)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=400)
    ax.plot(x, row, marker="o", linewidth=1.5)
    style_plot(ax)  # type: ignore[no-untyped-call]
    fig.tight_layout()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_dir / "last_row_distribution.pdf",
            bbox_inches="tight",
            dpi=400,
        )

        data_to_save = np.column_stack((x, row))
        np.savetxt(
            save_dir / "last_row_distribution.csv",
            data_to_save,
            delimiter=",",
            header="token_position,probability",
            comments="",
        )

        plt.close(fig)
    else:
        plt.show()


def max_offdiag_mass(
    attention_scores: Float[NDArray, "sequence_length sequence_length"],
) -> float:
    """
    Compute the maximum off-diagonal mass 1 - min_i A[i,i].

    Args:
        attention_scores: Square attention score matrix.

    Returns:
        float: The maximum off-diagonal mass.

    """
    return float(np.max(1.0 - np.diag(attention_scores)))


def row_entropy(
    scores_product: Float[NDArray, "sequence_length sequence_length"],
    eps: float = 1e-12,
) -> Float[NDArray, "sequence_length"]:
    """
    Compute the entropy of each row in the scores_product matrix.

    Clips values to avoid numerical issues and returns the per-row Shannon entropy.

    Args:
        scores_product: 2D matrix of row distributions (may need normalization outside).
        eps: Small epsilon for numerical stability when taking logs.

    Returns:
        Entropy per row.

    """
    scores_product = np.clip(scores_product, eps, 1.0)
    return -np.sum(scores_product * np.log(scores_product), axis=1)
