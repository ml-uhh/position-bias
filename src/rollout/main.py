"""
Model for residual-aware attention rollout.

Residual-aware (lazy) attention rollout with:
  - positional-only ALiBi (single- or multi-head),
  - optional "rough average content attention" prior C,
  - lambda schedules: constant / power-law decay / custom,
  - rollout P^(T) = R^(T) ... R^(0),
  - plotting: last-row distribution of P (where last token draws from after rollout).

Conventions:
  - tokens indexed i,j in {0,...,n-1}
  - A[i,j] is weight used to update token i from token j (edge j -> i).
  - causal masking (default): j <= i.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from jaxtyping import Float
import numpy as np
from numpy.typing import NDArray
import pydantic
import structlog
from tqdm import tqdm
import typer

from src.mitigation.correction import compute_correction_vector
from src.rollout.attention import (
    AttentionConfig,
    aggregate_measured_attention,
    make_multihead_attention,
)
from src.rollout.content_prior import (
    ContentPriorConfig,
    make_content_prior,
    make_measured_attention,
)
from src.rollout.lambda_schedule import (
    LambdaScheduleConfig,
    make_lambda_schedule,
)
from src.rollout.mask import MaskConfig, make_mask
from src.rollout.util import (
    max_offdiag_mass,
    plot_lambda_schedule,
    plot_last_row_distribution,
    row_entropy,
)
from src.util.pydantic import load_config

logger = structlog.get_logger()


class ModelConfig(pydantic.BaseModel):
    """Configuration for the residual-aware attention rollout model."""

    sequence_length: pydantic.PositiveInt = pydantic.Field(
        description="Sequence length (number of tokens)",
    )
    num_layers: pydantic.PositiveInt = pydantic.Field(
        description="Number of layers",
    )

    mask_config: MaskConfig = pydantic.Field(
        default_factory=MaskConfig,
        description="Configuration for the attention mask",
    )
    attention_config: AttentionConfig = pydantic.Field(
        description="Configuration for the attention mechanism",
    )
    content_config: ContentPriorConfig = pydantic.Field(
        description="Configuration for the content prior",
    )
    lambda_schedule_config: LambdaScheduleConfig = pydantic.Field(
        description="Configuration for the lambda schedule",
    )

    save_dir: Path | None = pydantic.Field(
        default=None,
        description="Directory to save plots and outputs",
    )

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.model_validator(mode="after")
    def validate_attention_layers(self) -> "ModelConfig":
        """Validates that attention config matches number of layers."""
        if len(self.attention_config.alibi_slopes) not in (1, self.num_layers):
            raise ValueError(
                "alibi_slopes outer list length must be 1 or match num_layers",
            )

        if (
            self.content_config.attention_heatmap_path is None
            and self.content_config.attention_probs_path is None
        ):
            if self.content_config.diagonal_scores is None:
                raise ValueError("content prior diagonal_scores must be provided")

            if self.content_config.base_scores is None:
                raise ValueError("content prior base_scores must be provided")

            if len(self.content_config.diagonal_scores) not in (1, self.num_layers):
                raise ValueError(
                    "content prior diagonal_scores outer list length must be 1 or match num_layers",
                )

            if len(self.content_config.base_scores) not in (1, self.num_layers):
                raise ValueError(
                    "content prior base_scores outer list length must be 1 or match num_layers",
                )

        return self


def add_residual(
    attention_scores: Float[NDArray, "sequence_length sequence_length"],
    attention_weight: float,
) -> Float[NDArray, "sequence_length sequence_length"]:
    """
    Residual operator: R = (1-λ)I + λA.

    Args:
        attention_scores: Attention scores matrix A.
        attention_weight: Residual weight lambda in [0, 1].

    Returns:
        Residual operator R.

    """
    assert 0.0 <= attention_weight <= 1.0

    sequence_length, _ = attention_scores.shape

    return (1.0 - attention_weight) * np.eye(
        sequence_length,
    ) + attention_weight * attention_scores


def rollout(
    layer_scores_list: Sequence[Float[NDArray, "sequence_length sequence_length"]],
) -> Float[NDArray, "sequence_length sequence_length"]:
    """
    Compute P = R_T @ ... @ R_0 (assuming Rs is ordered [R_0, R_1, ..., R_T]).

    Args:
        layer_scores_list: Sequence of layer score matrices R^(t).

    Returns:
        The rollout matrix P^(T).

    """
    sequence_length, _ = layer_scores_list[0].shape
    scores_product = np.eye(sequence_length, dtype=float)

    for layer_scores in layer_scores_list:
        scores_product = layer_scores @ scores_product

    return scores_product


@dataclass
class ModelResults:
    """Results of the model computation."""

    attention_scores_list: list[
        Float[NDArray, "sequence_length sequence_length"]
    ]  # List of effective attention matrices A^(t)
    layer_scores_list: list[
        Float[NDArray, "sequence_length sequence_length"]
    ]  # List of residual operators R^(t)
    scores_product: Float[
        NDArray,
        "sequence_length sequence_length",
    ]  # Rollout matrix P^(T)
    lambdas: Float[NDArray, "num_layers"]  # Array lambda_t


def compute_model_results(
    config: ModelConfig,
    correction: Float[NDArray, " sequence_length"]
    | Float[NDArray, "sequence_length sequence_length"]
    | Sequence[Float[NDArray, " sequence_length"]]
    | Sequence[Float[NDArray, "sequence_length sequence_length"]]
    | None = None,
    num_correction_layers: int = 1,
    correction_strength: float = 1.0,
) -> ModelResults:
    """
    Computes the model results including attention scores, layer scores, and rollout matrix.

    Args:
        config: ModelConfig containing all necessary configurations.
        correction: Optional correction added to attention logits before
            softmax. May be:

            * a single 1D vector ``(sequence_length,)`` — broadcast across
              queries and across all corrected layers (used by 1D modes);
            * a single 2D matrix ``(sequence_length, sequence_length)`` —
              per-(query, key) bias broadcast across all corrected layers
              (used by ``attention_delta``);
            * a sequence of 1D or 2D arrays — one per corrected layer
              (deepest-corrected first).
        num_correction_layers: Number of layers from the end where the
            correction is applied (default 1).
        correction_strength: Multiplicative gain applied to the normalized
            correction inside attention computation.

    Returns:
        ModelResults containing attention scores, layer scores, rollout matrix, and lambda schedule.

    """
    sequence_length, num_layers = config.sequence_length, config.num_layers
    mask = make_mask(config.mask_config, sequence_length)

    lambdas = make_lambda_schedule(config.lambda_schedule_config, num_layers)

    # Determine which layers get the correction
    correction_start = num_layers - num_correction_layers

    # Normalize correction to a per-layer list (or None). A single ndarray
    # (1D or 2D) broadcasts to every corrected layer; anything else is
    # treated as an explicit per-layer sequence. The 2D case must NOT fall
    # through to ``list(correction)``, which would split a matrix row-wise.
    per_layer_corrections: (
        list[
            Float[NDArray, " sequence_length"]
            | Float[NDArray, "sequence_length sequence_length"]
        ]
        | None
    ) = None
    if correction is not None:
        if isinstance(correction, np.ndarray):
            per_layer_corrections = [correction] * num_correction_layers
        else:
            per_layer_corrections = list(correction)

    attention_scores_list: list[Float[NDArray, "sequence_length sequence_length"]] = []
    layer_scores_list: list[Float[NDArray, "sequence_length sequence_length"]] = []

    for t in tqdm(range(num_layers), desc="Rollout layers"):
        measured_attention = make_measured_attention(
            config.content_config,
            sequence_length=sequence_length,
            layer_idx=t,
            num_heads=config.attention_config.num_heads,
        )

        layer_correction: (
            Float[NDArray, " sequence_length"]
            | Float[NDArray, "sequence_length sequence_length"]
            | None
        ) = None
        if per_layer_corrections is not None and t >= correction_start:
            layer_correction = per_layer_corrections[t - correction_start]

        if measured_attention is not None:
            if layer_correction is not None:
                raise ValueError(
                    "Logit corrections cannot be applied to measured "
                    "post-softmax attention kernels.",
                )
            attention_scores = aggregate_measured_attention(
                config.attention_config,
                head_attention=measured_attention,
                mask=mask,
                layer_index=t,
            )
        else:
            content_scores = make_content_prior(
                config.content_config,
                sequence_length=sequence_length,
                layer_idx=t,
                num_heads=config.attention_config.num_heads,
            )
            attention_scores = make_multihead_attention(
                config.attention_config,
                mask=mask,
                content_scores=content_scores,
                layer_index=t,
                logit_correction=layer_correction,
                correction_strength=correction_strength,
            )

        layer_scores = add_residual(attention_scores, attention_weight=lambdas[t])

        attention_scores_list.append(attention_scores)
        layer_scores_list.append(layer_scores)

    scores_product = rollout(layer_scores_list)

    return ModelResults(
        attention_scores_list,
        layer_scores_list,
        scores_product,
        lambdas,
    )


def compute_per_layer_corrections(
    config: ModelConfig,
    num_correction_layers: int,
    correction_strength: float = 1.0,
    n_start: int = 0,
    n_end: int = 0,
) -> list[Float[NDArray, " sequence_length"]]:
    """
    Compute per-layer correction vectors via closed-loop rollout refinement.

    Runs the rollout simulation layer by layer. At each corrected layer,
    computes the partial rollout (with previous corrections already applied),
    derives a correction from the residual bias, re-runs that layer with the
    correction, and continues. Each layer's correction only fixes the bias
    left over from previous layers, preventing overcorrection.

    Args:
        config: Rollout model configuration.
        num_correction_layers: Number of last layers to correct.
        correction_strength: Beta scaling applied when simulating each
            correction's effect.
        n_start: Number of leading positions to exclude from correction.
        n_end: Number of trailing positions to exclude from correction.

    Returns:
        List of correction vectors, one per corrected layer (index 0 is
        the earliest corrected layer).

    """
    sequence_length, num_layers = config.sequence_length, config.num_layers
    mask = make_mask(config.mask_config, sequence_length)
    lambdas = make_lambda_schedule(config.lambda_schedule_config, num_layers)
    correction_start = num_layers - num_correction_layers

    # Build partial rollout up to the first corrected layer (uncorrected).
    partial_rollout = np.eye(sequence_length, dtype=float)

    # Pre-compute content scores for all layers (they don't change).
    all_content_scores = [
        make_content_prior(
            config.content_config,
            sequence_length=sequence_length,
            layer_idx=t,
            num_heads=config.attention_config.num_heads,
        )
        for t in range(num_layers)
    ]

    # Run uncorrected layers before the correction window.
    for t in range(correction_start):
        attention = make_multihead_attention(
            config.attention_config,
            mask=mask,
            content_scores=all_content_scores[t],
            layer_index=t,
        )
        layer_scores = add_residual(attention, attention_weight=lambdas[t])
        partial_rollout = layer_scores @ partial_rollout

    # Sequentially compute corrections for each corrected layer.
    corrections: list[Float[NDArray, " sequence_length"]] = []

    for t in range(correction_start, num_layers):
        # Compute uncorrected attention at this layer (to build baseline P^(t)).
        attention_uncorrected = make_multihead_attention(
            config.attention_config,
            mask=mask,
            content_scores=all_content_scores[t],
            layer_index=t,
        )
        layer_scores_uncorrected = add_residual(
            attention_uncorrected,
            attention_weight=lambdas[t],
        )
        p_at_t = layer_scores_uncorrected @ partial_rollout

        # Derive correction from the partial rollout's last-row profile.
        profile = p_at_t[-1, :]
        profile = profile / profile.sum()
        c_t = compute_correction_vector(
            profile,
            beta=1.0,
            n_start=n_start,
            n_end=n_end,
        )
        corrections.append(c_t)

        # Re-run this layer WITH the correction applied.
        attention_corrected = make_multihead_attention(
            config.attention_config,
            mask=mask,
            content_scores=all_content_scores[t],
            layer_index=t,
            logit_correction=c_t,
            correction_strength=correction_strength,
        )
        layer_scores_corrected = add_residual(
            attention_corrected,
            attention_weight=lambdas[t],
        )

        # Update partial rollout with the corrected layer.
        partial_rollout = layer_scores_corrected @ partial_rollout

    return corrections


def compute_causal_correction(  # noqa: C901, PLR0912, PLR0915
    config: ModelConfig,
    beta: float,
    skip_first_n_layers: int = 1,
    correction_source: Literal[
        "rollout_profile",
        "layer_column_mass",
        "no_primacy_profile",
        "flat_rollout",
        "attention_delta",
    ] = "rollout_profile",
    n_primacy: int = 1,
    n_recency: int = 0,
    flat_target_mode: Literal["uniform", "middle_uniform"] = "uniform",
) -> (
    list[Float[NDArray, " sequence_length"]]
    | list[Float[NDArray, "sequence_length sequence_length"]]
):
    """
    Compute per-layer corrections from the causal-vs-full mask delta.

    Three signals are supported via ``correction_source``:

    ``rollout_profile`` (default): compares the accumulated partial rollout
    profile under causal masking with the profile under full (bidirectional)
    masking at each layer. The log-ratio of these accumulated profiles gives
    a logit-space correction that compensates for the distortion introduced
    by causal masking (primarily primacy bias). Corrections are iterative:
    each layer's correction accounts for corrections already applied at
    earlier layers, preventing overcorrection. This uses rollout profiles
    rather than single-layer attention because the last row of any single
    layer's attention matrix is identical under both masks — the causal/full
    delta only appears in accumulated state, where earlier tokens'
    representations (built under restricted visibility) have had time to
    propagate.

    ``layer_column_mass``: compares per-layer attention column sums
    directly. At each corrected layer t, computes ``A_causal_t.sum(axis=0)``
    and ``A_full_t.sum(axis=0)`` — how much total attention each key
    position receives across all queries — normalizes each to a probability
    distribution, and takes ``beta * log(col_full / col_causal)``. This is
    a stateless per-layer signal: no accumulated rollout, no parallel
    full-mask pass, no re-run step. Cheaper and more local; captures
    reception-fairness distortion at a single layer without modeling
    compounding.

    ``flat_rollout``: direct realization of the theoretical Approach 1.
    Targets a flat last-row rollout profile directly instead of comparing
    against full attention (whose rollout is itself recency-skewed under
    ALiBi). At each corrected layer, takes the log-ratio of a flat target
    vs the current causal partial-rollout last row and applies it as a
    per-key logit bias; iterative/closed-loop so later layers see the
    diminished residual. ``flat_target_mode`` selects the target shape:

    * ``uniform`` — target is exactly ``1/n`` at every position. Fights
      the unfixable primacy floor from Thm 5.4(i), but simplest to reason
      about. Recommended starting point.
    * ``middle_uniform`` — target copies the current causal profile
      verbatim on the first ``n_primacy`` and last ``n_recency`` positions
      (no correction there) and spreads the remaining budget uniformly
      across the middle window. Accepts the primacy floor and only
      redistributes the correctable budget.

    Note that strict "doubly-stochastic causal kernel" (the closed-form
    uniform fixed point of the rollout) is infeasible on the causal
    triangle — row 0 forces the kernel to the identity. This iterative
    log-ratio loop is the practical realization: it converges to the
    KL-closest feasible profile to the target.

    ``attention_delta``: stateless per-layer 2D ``(query, key)`` correction.
    Computes the dataset-averaged single-layer causal attention
    ``A^(t)_causal`` and forms ``C^(t)[i, j] = beta * log(A_target[i, j] /
    A^(t)_causal[i, j])`` for ``j <= i`` and 0 above the diagonal, with the
    flat target ``A_target[i, j] = 1 / (i + 1)`` (row-uniform on each
    query's causal support). Resolves the type-mismatch of
    ``rollout_profile`` (accumulated signal vs per-layer intervention) and
    the information loss of ``layer_column_mass`` (which collapses an nxn
    delta to an n-vector via column sums) at once. Row 0 is automatically a
    no-op so the Thm 5.4(i) primacy floor is respected by construction.
    Unlike all other modes here, this returns a list of 2D matrices.

    ``no_primacy_profile``: edits the real causal partial-rollout profile
    at each layer in place of comparing against full attention. Partitions
    the profile into a primacy head (first ``n_primacy`` positions), a
    recency tail (last ``n_recency`` positions), and a middle window.
    Caps each primacy position to the median of the non-primacy profile,
    redistributes the removed mass uniformly into the middle window, and
    leaves the recency tail verbatim. The correction is the log-ratio of
    this edited target vs the causal profile. Iterative/closed-loop like
    ``rollout_profile`` so later layers see the diminished residual. This
    mode intentionally preserves the recency gradient — use it when
    ``rollout_profile`` / ``layer_column_mass`` overcorrect and amplify
    recency into a harmful spike.

    Args:
        config: Rollout model configuration.
        beta: Correction strength. 0 = no correction, 1 = fully undo
            causal masking effect.
        skip_first_n_layers: Number of early layers to leave uncorrected.
            Layer 0 operates on raw embeddings where content dominates,
            and errors there compound maximally.
        correction_source: Signal used to derive corrections. See above.
        n_primacy: Size of the primacy head to cap. Only used when
            ``correction_source='no_primacy_profile'``. Default 1.
        n_recency: Size of the recency tail to leave untouched. Only used
            when ``correction_source='no_primacy_profile'``. Default 0
            (middle window extends to the final position).
        flat_target_mode: Target shape for ``correction_source='flat_rollout'``.
            ``"uniform"`` uses a literal 1/n target at every position.
            ``"middle_uniform"`` copies current primacy/recency from the
            causal profile (using ``n_primacy``/``n_recency``) and spreads
            the remaining budget uniformly across the middle window.

    Returns:
        List of correction vectors, one per corrected layer (index 0 is
        the earliest corrected layer). Length is
        ``num_layers - skip_first_n_layers`` and each vector has shape
        ``(sequence_length,)`` regardless of ``correction_source``.

    """
    sequence_length, num_layers = config.sequence_length, config.num_layers
    causal_mask = make_mask(MaskConfig(mask_type="causal"), sequence_length)
    full_mask = make_mask(MaskConfig(mask_type="full"), sequence_length)
    lambdas = make_lambda_schedule(config.lambda_schedule_config, num_layers)

    all_content_scores = [
        make_content_prior(
            config.content_config,
            sequence_length=sequence_length,
            layer_idx=t,
            num_heads=config.attention_config.num_heads,
        )
        for t in range(num_layers)
    ]

    if correction_source == "layer_column_mass":
        col_corrections: list[Float[NDArray, " sequence_length"]] = []
        for t in range(skip_first_n_layers, num_layers):
            content = all_content_scores[t]
            a_causal = make_multihead_attention(
                config.attention_config,
                mask=causal_mask,
                content_scores=content,
                layer_index=t,
            )
            a_full = make_multihead_attention(
                config.attention_config,
                mask=full_mask,
                content_scores=content,
                layer_index=t,
            )
            col_causal = a_causal.sum(axis=0)
            col_full = a_full.sum(axis=0)
            col_causal = col_causal / col_causal.sum()
            col_full = col_full / col_full.sum()
            col_causal = np.clip(col_causal, 1e-30, None)
            col_full = np.clip(col_full, 1e-30, None)
            c_t = beta * np.log(col_full / col_causal)
            col_corrections.append(c_t)
        return col_corrections

    if correction_source == "attention_delta":
        # Stateless 2D per-layer correction. Target is the row-uniform-on-
        # causal-support kernel A_target[i, j] = 1/(i+1) for j <= i. The
        # log-ratio against A^(t)_causal lives in the right coordinate
        # system (pre-softmax additive logit) so the per-layer signal and
        # per-layer intervention are type-matched — no closed-loop
        # iteration needed.
        idx = np.arange(sequence_length)
        support_size = (idx + 1).astype(float)
        a_target = np.tril(
            np.ones((sequence_length, sequence_length), dtype=float),
        )
        a_target /= support_size[:, np.newaxis]
        log_target = np.full(
            (sequence_length, sequence_length),
            -np.inf,
            dtype=float,
        )
        tri = np.tril_indices(sequence_length)
        log_target[tri] = np.log(a_target[tri])

        corrections_2d: list[Float[NDArray, "sequence_length sequence_length"]] = []
        for t in range(skip_first_n_layers, num_layers):
            content = all_content_scores[t]
            a_causal = make_multihead_attention(
                config.attention_config,
                mask=causal_mask,
                content_scores=content,
                layer_index=t,
            )
            a_causal_safe = np.clip(a_causal, 1e-30, None)
            c_t_mat = np.zeros(
                (sequence_length, sequence_length),
                dtype=float,
            )
            c_t_mat[tri] = beta * (log_target[tri] - np.log(a_causal_safe[tri]))
            corrections_2d.append(c_t_mat)
        return corrections_2d

    # Pre-compute full-mask rollout to get target profiles at each layer.
    # Only needed for rollout_profile mode — no_primacy_profile derives its
    # target from the causal profile alone, so we skip this loop.
    full_partial_rollouts: list[Float[NDArray, "sequence_length sequence_length"]] = []
    if correction_source == "rollout_profile":
        full_rollout = np.eye(sequence_length, dtype=float)
        for t in tqdm(range(num_layers), desc="Full-mask rollout"):
            a_full = make_multihead_attention(
                config.attention_config,
                mask=full_mask,
                content_scores=all_content_scores[t],
                layer_index=t,
            )
            r_full = add_residual(a_full, attention_weight=lambdas[t])
            full_rollout = r_full @ full_rollout
            full_partial_rollouts.append(full_rollout.copy())

    # Middle window bounds are needed by both no_primacy_profile and the
    # flat_rollout/middle_uniform branches. Compute once up front so pyright
    # can see they're always bound when referenced below.
    middle_start = n_primacy
    middle_end = sequence_length - n_recency
    needs_middle_window = correction_source == "no_primacy_profile" or (
        correction_source == "flat_rollout" and flat_target_mode == "middle_uniform"
    )
    if needs_middle_window and middle_end <= middle_start:
        raise ValueError(
            f"{correction_source} middle window is empty: "
            f"n_primacy={n_primacy}, n_recency={n_recency}, "
            f"sequence_length={sequence_length}. Need "
            f"middle_end ({middle_end}) > middle_start ({middle_start}).",
        )

    flat_target_uniform: NDArray | None = None
    if correction_source == "flat_rollout" and flat_target_mode == "uniform":
        flat_target_uniform = np.full(
            sequence_length,
            1.0 / sequence_length,
            dtype=float,
        )

    # Iteratively compute causal corrections.
    causal_rollout = np.eye(sequence_length, dtype=float)
    corrections: list[Float[NDArray, " sequence_length"]] = []

    for t in tqdm(range(num_layers), desc="Causal correction"):
        content = all_content_scores[t]

        if t < skip_first_n_layers:
            attention = make_multihead_attention(
                config.attention_config,
                mask=causal_mask,
                content_scores=content,
                layer_index=t,
            )
            layer_scores = add_residual(attention, attention_weight=lambdas[t])
            causal_rollout = layer_scores @ causal_rollout
            continue

        # Uncorrected causal attention and rollout at this layer.
        a_causal = make_multihead_attention(
            config.attention_config,
            mask=causal_mask,
            content_scores=content,
            layer_index=t,
        )
        r_causal = add_residual(a_causal, attention_weight=lambdas[t])
        p_causal_at_t = r_causal @ causal_rollout

        # Derive a target profile from the causal partial rollout. The two
        # modes differ only in how the target is built.
        if correction_source == "no_primacy_profile":
            profile_causal_raw = p_causal_at_t[-1, :].astype(float)
            profile_causal_raw = profile_causal_raw / profile_causal_raw.sum()

            # One-sided cap: cap the primacy head to the median of the
            # non-primacy profile, but ONLY for positions currently above
            # the median. Positions already at-or-below the baseline are
            # left alone — we reduce attention to the front, never raise
            # it. Excess is redistributed uniformly into the middle window;
            # the recency tail is copied verbatim.
            baseline = float(np.median(profile_causal_raw[middle_start:]))
            primacy_head = profile_causal_raw[:middle_start]
            capped_head = np.minimum(primacy_head, baseline)
            excess = float((primacy_head - capped_head).sum())

            target_raw = profile_causal_raw.copy()
            target_raw[:middle_start] = capped_head
            middle_len = middle_end - middle_start
            target_raw[middle_start:middle_end] += excess / middle_len
            target_raw = target_raw / target_raw.sum()

            profile_causal = np.clip(profile_causal_raw, 1e-30, None)
            profile_target = np.clip(target_raw, 1e-30, None)
        elif correction_source == "flat_rollout":
            profile_causal_raw = p_causal_at_t[-1, :].astype(float)
            profile_causal_raw = profile_causal_raw / profile_causal_raw.sum()

            if flat_target_mode == "uniform":
                assert flat_target_uniform is not None
                target_raw = flat_target_uniform
            else:  # middle_uniform
                target_raw = profile_causal_raw.copy()
                copied_primacy_mass = float(target_raw[:middle_start].sum())
                copied_recency_mass = (
                    float(target_raw[middle_end:].sum()) if n_recency > 0 else 0.0
                )
                middle_budget = 1.0 - copied_primacy_mass - copied_recency_mass
                middle_len = middle_end - middle_start
                target_raw[middle_start:middle_end] = middle_budget / middle_len

            profile_causal = np.clip(profile_causal_raw, 1e-30, None)
            profile_target = np.clip(target_raw, 1e-30, None)
        else:
            # rollout_profile: compare against the full-mask partial rollout.
            profile_causal = np.clip(p_causal_at_t[-1, :], 1e-30, None)
            profile_target = np.clip(
                full_partial_rollouts[t][-1, :],
                1e-30,
                None,
            )

        c_t = beta * np.log(profile_target / profile_causal)

        corrections.append(c_t)

        # Re-run layer with correction under causal mask.
        a_corrected = make_multihead_attention(
            config.attention_config,
            mask=causal_mask,
            content_scores=content,
            layer_index=t,
            logit_correction=c_t,
            correction_strength=1.0,
        )
        layer_scores = add_residual(a_corrected, attention_weight=lambdas[t])
        causal_rollout = layer_scores @ causal_rollout

    return corrections


def main(
    config_file: Path,
    beta: float = 0.0,
    num_correction_layers: int = 1,
    n_start: int = 0,
    n_end: int = 0,
    per_layer: bool = True,
) -> None:
    """
    Main function to compute and log rollout results.

    Args:
        config_file: Path to the model configuration file.
        beta: Correction strength for position bias mitigation (0 = no correction).
        num_correction_layers: Number of last layers to apply correction to.
        n_start: Number of leading positions to exclude from correction.
        n_end: Number of trailing positions to exclude from correction.
        per_layer: Use closed-loop per-layer corrections (True) or a single
            correction vector applied to all corrected layers (False).

    """
    config = load_config(config_file, ModelConfig)

    # Compute corrections
    correction = None
    if beta > 0:
        uncorrected = compute_model_results(config)
        if per_layer:
            correction = compute_per_layer_corrections(
                config,
                num_correction_layers=num_correction_layers,
                correction_strength=beta,
                n_start=n_start,
                n_end=n_end,
            )
        else:
            profile = uncorrected.scores_product[-1, :]
            correction = compute_correction_vector(
                profile,
                beta=1.0,
                n_start=n_start,
                n_end=n_end,
            )
        logger.info(
            "Applying correction",
            beta=beta,
            num_correction_layers=num_correction_layers,
            per_layer=per_layer,
            n_start=n_start,
            n_end=n_end,
        )
        # Save uncorrected plot
        plot_last_row_distribution(
            uncorrected.scores_product,
            save_dir=config.save_dir,
            filename_suffix="_uncorrected",
        )

    model_results = compute_model_results(
        config,
        correction=correction,
        num_correction_layers=num_correction_layers,
        correction_strength=beta,
    )

    logger.info("=== Summary ===")
    logger.info(config)
    logger.info(
        f"row sums max |sum-1| in P: {float(np.max(np.abs(model_results.scores_product.sum(axis=1) - 1.0))):.3e}",
    )
    logger.info(
        f"min_i P[i,i]: {float(np.min(np.diag(model_results.scores_product))):.6f}",
    )
    logger.info(
        f"max_i P[i,0]: {float(np.max(model_results.scores_product[:, 0])):.6f}",
    )
    logger.info(
        f"mean row entropy: {float(np.mean(row_entropy(model_results.scores_product))):.6f} nats",
    )

    # Approximate theoretical control quantity sum_t lambda_t * delta_t
    deltas = np.array(
        [max_offdiag_mass(A) for A in model_results.attention_scores_list],
        dtype=float,
    )
    logger.info(
        f"sum_t lambda_t * delta_t: {float(np.sum(model_results.lambdas * deltas)):.6f}",
    )
    # Plots
    plot_lambda_schedule(
        model_results.lambdas,
        save_dir=config.save_dir,
    )
    corrected_suffix = (
        f"_corrected_beta_{beta}_k_{num_correction_layers}" if beta > 0 else ""
    )

    # Accumulate last-row distribution for each layer's partial rollout.
    partial = np.eye(config.sequence_length, dtype=float)
    per_layer_distributions = []
    for layer_scores in model_results.layer_scores_list:
        partial = layer_scores @ partial
        per_layer_distributions.append(partial[-1].copy())
    per_layer_distributions_array = np.stack(per_layer_distributions, axis=0)

    if config.save_dir is not None:
        config.save_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            config.save_dir / f"per_layer_distributions{corrected_suffix}.npy",
            per_layer_distributions_array,
        )

    plot_last_row_distribution(
        model_results.scores_product,
        save_dir=config.save_dir,
        filename_suffix=corrected_suffix,
    )

    # Save full rollout matrix
    if config.save_dir is not None:
        config.save_dir.mkdir(parents=True, exist_ok=True)
        np.save(
            config.save_dir / "rollout_matrix.npy",
            model_results.scores_product,
        )
        logger.info(f"Rollout matrix saved to {config.save_dir / 'rollout_matrix.npy'}")


if __name__ == "__main__":
    typer.run(main)
