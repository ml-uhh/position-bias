"""Rollout-derived correction for position bias mitigation."""

from collections.abc import Callable, Sequence
from typing import Any, Literal
import warnings

from jaxtyping import Float
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.falcon.modeling_falcon import FalconForCausalLM
from transformers.models.mpt.modeling_mpt import MptForCausalLM

StdMatchMode = Literal["per_head", "global", "none"]

# A 2-D correction carries a per-query-position axis in addition to the
# per-key axis; 1-D corrections are broadcast across all query positions.
_CORRECTION_2D_NDIM = 2

# Model-specific config: which positional arg index holds the ALiBi bias
_ALIBI_ARG_INDEX = {
    "bloom": 2,  # forward(hidden_states, residual, alibi, ...)
    "falcon": 1,  # forward(hidden_states, alibi, ...)
    "mpt": 1,  # forward(hidden_states, position_bias, ...)
}


def compute_correction_vector(
    rollout_profile: Float[NDArray, " sequence_length"],
    beta: float,
    n_start: int = 0,
    n_end: int = 0,
) -> Float[NDArray, " sequence_length"]:
    """
    Compute the additive logit correction from a rollout profile.

    For each key position j, the correction is:
        c(j) = beta * log((1/n) / r(j, n))

    where r(j, n) is the rollout-predicted influence of token j on the
    last token, and 1/n is the uniform target. Positions receiving excess
    attention are suppressed; starved middle positions are boosted.

    Positions in the exclusion zones (first *n_start*, last *n_end*) are
    zeroed out. The recency tail spike is a hard constraint of causal
    masking — the last token's causal window **is** those nearby tokens —
    so correcting it leads to overcorrection.

    Args:
        rollout_profile: Last-row rollout distribution r(j, n), shape (n,).
                         Must be strictly positive and sum to 1.
        beta: Correction strength in [0, 1]. 0 means no correction.
        n_start: Number of leading positions to exclude from correction.
        n_end: Number of trailing positions to exclude from correction.

    Returns:
        Correction vector c of shape (n,).

    """
    n = len(rollout_profile)
    if n_start + n_end >= n:
        raise ValueError(
            f"Exclusion zones cover the entire sequence: "
            f"n_start ({n_start}) + n_end ({n_end}) >= n ({n}).",
        )
    uniform = 1.0 / n
    profile = np.clip(rollout_profile, 1e-30, None)
    correction = beta * np.log(uniform / profile)
    if n_start > 0:
        correction[:n_start] = 0.0
    if n_end > 0:
        correction[-n_end:] = 0.0
    return correction


def get_attention_modules(
    model: nn.Module,
    num_correction_layers: int,
) -> list[nn.Module]:
    """Return attention modules for the last *num_correction_layers* layers of an ALiBi model."""
    modules: list[nn.Module]
    match model:
        case BloomForCausalLM():
            layers = list(model.transformer.h)
            modules = [layer.self_attention for layer in layers]  # pyright: ignore[reportAssignmentType]
        case FalconForCausalLM():
            layers = list(model.transformer.h)
            modules = [layer.self_attention for layer in layers]  # pyright: ignore[reportAssignmentType]
        case MptForCausalLM():
            layers = list(model.transformer.blocks)
            modules = [layer.attn for layer in layers]  # pyright: ignore[reportAssignmentType]
        case _:
            raise TypeError(
                f"Model type {type(model)} not supported for correction hooks. "
                f"Supported: BloomForCausalLM, FalconForCausalLM, MptForCausalLM.",
            )

    total_layers = len(modules)
    if num_correction_layers > total_layers:
        raise ValueError(
            f"num_correction_layers ({num_correction_layers}) exceeds "
            f"model depth ({total_layers}).",
        )

    return modules[-num_correction_layers:]


def _detect_model_family(module: nn.Module) -> str:
    """Detect which ALiBi model family an attention module belongs to."""
    module_type = type(module).__name__
    if "Bloom" in module_type:
        return "bloom"
    if "Falcon" in module_type:
        return "falcon"
    if "Mpt" in module_type:
        return "mpt"
    raise TypeError(f"Unknown attention module type: {module_type}")


def _expand_2d_to_bias_leading(
    c2d: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Expand a shared ``(S, S)`` correction to ``(leading, S, S)`` by broadcast."""
    leading = bias.shape[0]
    return c2d.unsqueeze(0).expand(leading, -1, -1)


def _expand_2d_per_head_to_bias(
    c2d_per_head: torch.Tensor,  # (H, S, S)
    bias: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Expand a per-head ``(H, S, S)`` correction to match the bias's leading layout."""
    leading = bias.shape[0]
    if leading == num_heads:
        # MPT-style: (H, 1, S) bias, leading dim is already heads.
        return c2d_per_head
    batch = leading // num_heads
    # Bloom/Falcon-style: (B*H, 1, S) bias. Expand to (B, H, S, S), then flatten.
    return (
        c2d_per_head.unsqueeze(0)
        .expand(batch, num_heads, *c2d_per_head.shape[1:])
        .reshape(batch * num_heads, *c2d_per_head.shape[1:])
    )


def _fit_correction_2d(
    correction: torch.Tensor,
    bias: torch.Tensor,
    correction_strength: float,
    num_heads: int,
    std_match: StdMatchMode,
    q_length: int,
) -> torch.Tensor:
    """
    Fit a 2D ``(S, S)`` correction (attention_delta mode) to the bias.

    Bloom/MPT pass the ALiBi bias as ``(leading, 1, kv_length)`` in *both*
    prefill and decode — the query dim is broadcast-1 regardless of
    ``q_length``. But ``baddbmm`` internally expands the bias to match the
    actual output shape ``(leading, q_length, kv_length)``; a 2D bias with
    middle dim != q_length would fail there (``expand`` can't shrink a
    non-singleton dim). So the caller tells us the real ``q_length`` (from
    ``hidden_states.shape[1]``) and we slice the 2D correction to exactly
    the current query rows: ``c[kv_len - q_length : kv_len, :kv_len]``.
    This works for both prefill (``q_length == kv_length`` → all rows) and
    decode with KV cache (``q_length == 1`` → the one row corresponding to
    the current query position, which is ``kv_length - 1``).

    Returns a delta of shape ``(leading, q_length, kv_length)`` which adds
    cleanly to the ``(leading, 1, kv_length)`` bias and matches the
    downstream ``baddbmm`` / MPT-slice output shape.
    """
    c = correction.to(device=bias.device, dtype=bias.dtype)
    kv_length = bias.shape[-1]

    if c.shape[-1] != c.shape[-2]:
        raise ValueError(
            f"2D correction must be square; got shape {tuple(c.shape)}.",
        )
    if kv_length > c.shape[-1]:
        raise ValueError(
            f"2D correction seq_len {c.shape[-1]} is shorter than bias "
            f"kv_length {kv_length}; attention_delta needs a rollout "
            f"computed at >= the runtime seq_len.",
        )
    if q_length > kv_length:
        raise ValueError(
            f"q_length ({q_length}) cannot exceed kv_length ({kv_length}).",
        )

    # Compute sigma_c from the FULL 2D correction's causal support BEFORE
    # slicing to the current query rows. This keeps the scale stable across
    # prefill (many rows) and decode (one row) — otherwise q_length=1 would
    # give us one row's worth of samples and a noisy, q-dependent scale.
    c_full = c[:kv_length, :kv_length]
    tri_mask = torch.tril(
        torch.ones(kv_length, kv_length, dtype=torch.bool, device=c.device),
    )
    c_support = c_full[tri_mask]
    sigma_c = c_support.float().std() if c_support.numel() > 1 else torch.tensor(0.0)

    # Now slice to the exact query rows the current forward pass will compute.
    # Prefill: start=0, rows=[0..kv_length). Decode: start=kv_length-1, rows=[kv_length-1].
    start_q = kv_length - q_length
    c = c[start_q : start_q + q_length, :kv_length]  # (q_length, kv_length)

    if std_match == "none":
        return _expand_2d_to_bias_leading(correction_strength * c, bias)

    if std_match == "global":
        sigma_bias = bias.float().std()
        scale = sigma_bias / sigma_c if sigma_c > 0 else torch.tensor(1.0)
        return _expand_2d_to_bias_leading(
            correction_strength * scale.to(c.dtype) * c,
            bias,
        )

    # std_match == "per_head"
    leading = bias.shape[0]
    if leading == num_heads:
        bias_per_head = bias.reshape(num_heads, -1)
    elif leading % num_heads == 0:
        batch = leading // num_heads
        bias_per_head = (
            bias.view(batch, num_heads, *bias.shape[1:])
            .movedim(1, 0)
            .reshape(num_heads, -1)
        )
    else:
        warnings.warn(
            f"_fit_correction_2d: bias shape {tuple(bias.shape)} is not "
            f"compatible with num_heads={num_heads} (leading dim {leading} "
            f"is not a multiple of num_heads). Falling back to a single "
            f"global sigma; per-head std-matching is disabled.",
            RuntimeWarning,
            stacklevel=2,
        )
        sigma_bias = bias.float().std()
        scale = sigma_bias / sigma_c if sigma_c > 0 else torch.tensor(1.0)
        return _expand_2d_to_bias_leading(
            correction_strength * scale.to(c.dtype) * c,
            bias,
        )

    sigma_h = bias_per_head.float().std(dim=-1)  # (H,)
    scale_h = sigma_h / sigma_c if sigma_c > 0 else torch.ones_like(sigma_h)
    c_per_head = scale_h.to(c.dtype).view(num_heads, 1, 1) * c.unsqueeze(0)
    c_per_head = correction_strength * c_per_head  # (H, S, S)
    return _expand_2d_per_head_to_bias(c_per_head, bias, num_heads)


def _fit_correction(  # noqa: C901, PLR0912
    correction: torch.Tensor,
    bias: torch.Tensor,
    correction_strength: float,
    num_heads: int,
    std_match: StdMatchMode = "per_head",
    q_length: int | None = None,
) -> torch.Tensor:
    """
    Scale and resize the correction to match the bias tensor.

    Accepts both 1D ``(S,)`` corrections (the additive-logit vector produced
    by every causal-correction source except ``attention_delta``) and 2D
    ``(S, S)`` corrections (the per-query-per-key log-ratio matrix produced
    by ``attention_delta``). The 2D path returns a delta of shape
    ``(leading, S, S)`` that *upgrades* the bias's query dim from 1 to S,
    exploiting PyTorch's broadcast semantics at the ``bias + delta`` site.
    This shape is accepted by Bloom's ``baddbmm`` and MPT's explicit 3D
    ``position_bias`` slicing, but NOT by Falcon (see the Falcon guard in
    ``register_correction_hooks``).

    The *std_match* argument selects how the correction is scaled before
    being added to the bias:

    * ``"per_head"`` (default): reshape the bias to ``(H, ..., S)`` and
      compute one sigma per head over every non-head axis, then scale
      ``c`` independently for each head. Avoids the ~100x over-amplification
      shallow-slope heads suffer under a single global sigma.
    * ``"global"``: compute one sigma over the full bias tensor and broadcast
      one scaled ``c`` to every head.
    * ``"none"``: no std-matching at all. ``c`` is added raw (only
      ``correction_strength`` is applied), which is useful when the
      correction is already in logit units.

    The returned tensor is shape-compatible with *bias* under addition.
    """
    if correction.ndim == _CORRECTION_2D_NDIM:
        if q_length is None:
            raise ValueError(
                "2D correction requires q_length to be passed explicitly "
                "(the current forward pass's hidden_states.shape[1]). The "
                "caller should extract it from the pre-hook's args[0].",
            )
        return _fit_correction_2d(
            correction,
            bias,
            correction_strength,
            num_heads,
            std_match,
            q_length,
        )

    c = correction.to(device=bias.device, dtype=bias.dtype)
    seq_len = bias.shape[-1]

    if seq_len <= c.shape[0]:
        c = c[:seq_len]
    else:
        c = torch.nn.functional.pad(c, (0, seq_len - c.shape[0]))

    if std_match == "none":
        return (correction_strength * c).expand_as(bias)

    if std_match == "global":
        sigma_bias = bias.std()
        sigma_c_global = c.std()
        if sigma_c_global > 0:
            c = (sigma_bias / sigma_c_global) * c
        return (correction_strength * c).expand_as(bias)

    leading = bias.shape[0]
    if leading == num_heads:
        # MPT-style: (H, 1, S). Heads are already the leading dim.
        bias_per_head = bias.reshape(num_heads, -1)
        delta_shape: tuple[int, ...] = bias.shape
        broadcast_view: tuple[int, ...] = (num_heads, *([1] * (bias.ndim - 2)), seq_len)
    elif leading % num_heads == 0:
        # Bloom/Falcon-style: (B*H, 1, S). Split off batch and move H first.
        batch = leading // num_heads
        bias_h = bias.view(batch, num_heads, *bias.shape[1:]).movedim(1, 0)
        bias_per_head = bias_h.reshape(num_heads, -1)
        delta_shape = bias.shape
        broadcast_view = (1, num_heads, *([1] * (bias.ndim - 2)), seq_len)
    else:
        # Unknown layout: we can't locate the head axis, so per-head matching
        # is impossible. Warn loudly and fall back to a single global sigma —
        # this reproduces the pre-fix behavior and is known to over-amplify
        # shallow-slope heads, so a layout branch should be added above.
        warnings.warn(
            f"_fit_correction: bias shape {tuple(bias.shape)} is not compatible "
            f"with num_heads={num_heads} (leading dim {leading} is not a multiple "
            f"of num_heads). Falling back to a single global sigma; per-head "
            f"std-matching is disabled and shallow heads may be over-amplified.",
            RuntimeWarning,
            stacklevel=2,
        )
        sigma_bias = bias.std()
        sigma_c_global = c.std()
        if sigma_c_global > 0:
            c = (sigma_bias / sigma_c_global) * c
        return (correction_strength * c).expand_as(bias)

    sigma_h = bias_per_head.std(dim=-1)  # (H,)
    sigma_c = c.std()
    scale = sigma_h / sigma_c if sigma_c > 0 else torch.ones_like(sigma_h)

    c_per_head = scale.unsqueeze(-1) * c.unsqueeze(0)  # (H, S)
    c_per_head = correction_strength * c_per_head

    if leading == num_heads:
        delta = c_per_head.view(broadcast_view).expand(delta_shape)
    else:
        batch = leading // num_heads
        delta = (
            c_per_head.view(broadcast_view)
            .expand(batch, num_heads, *bias.shape[1:])
            .reshape(delta_shape)
        )
    return delta


def _apply_correction(
    correction: torch.Tensor,
    bias: torch.Tensor,
    correction_strength: float,
    num_heads: int,
    std_match: StdMatchMode = "per_head",
    q_length: int | None = None,
) -> torch.Tensor:
    """Add the fitted correction to the bias. Shape matches baddbmm output."""
    delta = _fit_correction(
        correction,
        bias,
        correction_strength,
        num_heads,
        std_match=std_match,
        q_length=q_length,
    )
    return bias + delta


def _get_num_heads(module: nn.Module) -> int:
    """Return the head count of an attention module (Bloom/Falcon: ``num_heads``; MPT: ``n_heads``)."""
    for attr in ("num_heads", "n_heads"):
        value = getattr(module, attr, None)
        if isinstance(value, int):
            return value
    raise AttributeError(
        f"Cannot determine num_heads for module of type {type(module).__name__}.",
    )


def _make_correction_pre_hook(
    correction: torch.Tensor,
    alibi_arg_index: int,
    correction_strength: float,
    num_heads: int,
    std_match: StdMatchMode = "per_head",
) -> Callable[..., Any]:
    """
    Create a forward pre-hook that adds the correction to the ALiBi bias tensor.

    The hook first checks the positional arg at *alibi_arg_index*; if not
    found there, it falls back to the ``alibi`` keyword argument.
    """

    def hook(
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        # Extract q_length from hidden_states (first positional arg on every
        # supported attention module). Only needed for 2D corrections, but
        # cheap to read unconditionally.
        q_length: int | None = None
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if (
            isinstance(hidden_states, torch.Tensor)
            and hidden_states.ndim >= _CORRECTION_2D_NDIM
        ):
            q_length = hidden_states.shape[1]

        # Check positional arg first
        if alibi_arg_index < len(args) and args[alibi_arg_index] is not None:
            bias = args[alibi_arg_index]
            args_list = list(args)
            args_list[alibi_arg_index] = _apply_correction(
                correction,
                bias,
                correction_strength,
                num_heads,
                std_match=std_match,
                q_length=q_length,
            )
            return tuple(args_list), kwargs

        # Fall back to the "alibi" keyword argument (transformers >= 4.40+)
        if "alibi" in kwargs and kwargs["alibi"] is not None:
            bias = kwargs["alibi"]
            return args, {
                **kwargs,
                "alibi": _apply_correction(
                    correction,
                    bias,
                    correction_strength,
                    num_heads,
                    std_match=std_match,
                    q_length=q_length,
                ),
            }

        return None

    return hook


def register_correction_hooks(
    attention_modules: Sequence[nn.Module],
    correction_vectors: NDArray[np.floating[Any]] | Sequence[NDArray[np.floating[Any]]],
    alibi_arg_index: int | None = None,
    correction_strength: float = 1.0,
    std_match: StdMatchMode = "per_head",
) -> list[torch.utils.hooks.RemovableHandle]:
    """
    Register correction pre-hooks on the given attention modules.

    Uses a forward pre-hook that modifies the ALiBi bias argument
    to add the correction vector before each forward pass.

    Args:
        attention_modules: Attention sub-modules to hook (from *get_attention_modules*).
        correction_vectors: Additive logit correction — either a single 1D array
            of shape ``(seq_length,)`` applied to all modules, a sequence of
            1D arrays (one per module) for per-layer correction, or a sequence
            of 2D ``(seq_length, seq_length)`` arrays for ``attention_delta``
            mode. 2D corrections are only supported on Bloom and MPT; passing
            them to a Falcon module raises ``TypeError`` at registration time.
        alibi_arg_index: Override the positional-arg index that carries the ALiBi
            bias tensor. When *None* (the default), the index is auto-detected
            from the module class name.
        correction_strength: Multiplicative gain applied after correction
            normalization. This is where beta should be applied.
        std_match: How to scale the correction relative to the bias. See
            ``_fit_correction`` — ``"per_head"`` (default) matches each
            head's own sigma, ``"global"`` matches the flattened bias
            sigma, ``"none"`` disables std-matching entirely.

    Returns:
        A list of removable hook handles (one per module).

    """
    # Normalize to a list of per-module correction tensors.
    if isinstance(correction_vectors, np.ndarray) and correction_vectors.ndim == 1:
        tensors = [
            torch.tensor(correction_vectors, dtype=torch.float32),
        ] * len(attention_modules)
    else:
        if len(correction_vectors) != len(attention_modules):
            raise ValueError(
                f"Number of correction vectors ({len(correction_vectors)}) must match "
                f"number of attention modules ({len(attention_modules)}).",
            )
        tensors = [torch.tensor(cv, dtype=torch.float32) for cv in correction_vectors]

    has_2d_correction = any(t.ndim == _CORRECTION_2D_NDIM for t in tensors)

    handles: list[torch.utils.hooks.RemovableHandle] = []
    for module, correction_tensor in zip(attention_modules, tensors, strict=True):
        family: str | None
        if alibi_arg_index is not None:
            arg_index = alibi_arg_index
            family = None
        else:
            family = _detect_model_family(module)
            arg_index = _ALIBI_ARG_INDEX[family]
        if has_2d_correction and family == "falcon":
            raise TypeError(
                "attention_delta (2D correction) is not supported on Falcon: "
                "transformers' FalconAttention.forward at "
                "modeling_falcon.py:397 hardcodes the alibi query dim to 1 "
                "via .view(batch, heads, 1, -1), which would fail with a 2D "
                "correction. Use a 1D causal source (rollout_profile / "
                "layer_column_mass / no_primacy_profile / flat_rollout) on "
                "Falcon models.",
            )
        handle = module.register_forward_pre_hook(
            _make_correction_pre_hook(
                correction_tensor,
                arg_index,
                correction_strength,
                _get_num_heads(module),
                std_match=std_match,
            ),
            with_kwargs=True,
        )
        handles.append(handle)
    return handles


def unregister_correction_hooks(
    handles: list[torch.utils.hooks.RemovableHandle],
) -> None:
    """
    Remove all registered correction hooks.

    Args:
        handles: Hook handles from register_correction_hooks.

    """
    for handle in handles:
        handle.remove()
