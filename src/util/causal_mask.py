"""Disable causal masking on HuggingFace decoder models."""

import sys

import torch


def disable_causal_mask(model: torch.nn.Module) -> None:
    """
    Make attention fully bidirectional by disabling causal masking.

    Patches `create_causal_mask` (and `create_sliding_window_causal_mask`,
    if present) in the inner model's modeling module to return None, and
    sets `is_causal=False` on all attention submodules so the SDPA path
    does not re-enable causality when no mask is present.
    Wraps each attention `forward` with an assertion that `is_causal` is
    False and `attention_mask` is None on every call.

    This mutates `transformers` module globals — it affects any model whose
    class lives in the same modeling module for the remainder of the process.
    """
    inner = getattr(model, "model", model)
    module = sys.modules.get(inner.__class__.__module__)
    if module is not None:
        for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
            if hasattr(module, name):
                setattr(module, name, lambda **_: None)

    for submodule in model.modules():
        if getattr(submodule, "is_causal", None) is True:
            submodule.is_causal = False  # type: ignore[reportArgumentTyp, unused-ignore]
            _wrap_forward_with_assert(submodule)


def _wrap_forward_with_assert(attn_module: torch.nn.Module) -> None:
    original_forward = attn_module.forward

    def forward_with_assert(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        assert attn_module.is_causal is False, (
            f"{type(attn_module).__name__}.is_causal must be False when "
            f"causal mask is disabled, got {attn_module.is_causal!r}"
        )
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None and len(args) >= 3:
            attention_mask = args[2]
        assert attention_mask is None, (
            f"{type(attn_module).__name__}.forward received non-None "
            f"attention_mask when causal mask is disabled"
        )
        assert kwargs.get("is_causal", False) is False, (
            f"{type(attn_module).__name__}.forward received is_causal=True "
            f"kwarg when causal mask is disabled"
        )
        return original_forward(*args, **kwargs)

    attn_module.forward = forward_with_assert  # type: ignore[method-assign]
