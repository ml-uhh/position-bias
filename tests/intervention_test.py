"""Sanity checks for attention-branch scaling."""
# ruff: noqa: INP001 (test module, intentionally not a package)

from collections.abc import Callable

import pytest
import torch
from torch import nn
from transformers import BloomConfig, BloomForCausalLM, FalconConfig, FalconForCausalLM, Qwen2Config, Qwen2ForCausalLM

from src.intervention.intervention import BetaScale, _attention_modules

# Tiny, offline, randomly-initialized -- no network access, no pretrained weights. Mirrors
# falcon-rw-7b's architecture flags (alibi, non-parallel, non-new-decoder) at toy dimensions, so
# these tests exercise the real FalconAttention/FalconDecoderLayer forward logic this module's
# docstring relies on, not just a duck-typed stand-in.
_TINY_FALCON_RW_CONFIG = FalconConfig(
    vocab_size=32,
    hidden_size=16,
    num_hidden_layers=3,
    num_attention_heads=2,
    alibi=True,
    new_decoder_architecture=False,
    parallel_attn=False,
    multi_query=False,
)

# Tiny, offline, randomly-initialized -- default (non-tensor-parallel) Bloom architecture flags, so
# these tests exercise the real BloomAttention/BloomBlock forward logic (in particular
# self_attention.dense's pre-residual output), not just a duck-typed stand-in.
_TINY_BLOOM_CONFIG = BloomConfig(
    vocab_size=32,
    hidden_size=16,
    n_layer=3,
    n_head=2,
    pretraining_tp=1,
    slow_but_exact=False,
)

_TINY_QWEN2_CONFIG = Qwen2Config(
    vocab_size=32,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=3,
    num_attention_heads=2,
    num_key_value_heads=2,
    head_dim=8,
)


class _ToyAttn(nn.Module):
    """Toy attention submodule: linear context branch plus a fixed second ("weights") tensor."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (context_output, attn_weights), mirroring MptAttention's tuple output."""
        context = x @ self.weight
        weights = torch.ones(x.shape[0], x.shape[1])
        return context, weights


class _ToyBlock(nn.Module):
    """Toy transformer block: residual + attention output, matching MptBlock's ordering."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.attn = _ToyAttn(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return x + Attn(x)[0], mirroring a real block's post-attention residual update."""
        attn_output, _attn_weights = self.attn(x)
        return x + attn_output


class _ToyTransformer(nn.Module):
    def __init__(self, num_blocks: int, weight: torch.Tensor) -> None:
        super().__init__()
        # A plain list (not nn.ModuleList) keeps indexing/iteration statically typed as
        # _ToyBlock -- nn.ModuleList.__getitem__ is typed to return the generic nn.Module.
        self.blocks: list[_ToyBlock] = [_ToyBlock(weight) for _ in range(num_blocks)]


class _ToyModel(nn.Module):
    """Duck-types MptForCausalLM's ``model.transformer.blocks[i].attn`` access pattern."""

    def __init__(self, num_blocks: int = 2, hidden: int = 3) -> None:
        super().__init__()
        weight = torch.eye(hidden) * 0.5
        self.transformer = _ToyTransformer(num_blocks, weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.transformer.blocks:
            x = block(x)
        return x


def test_beta_one_reproduces_unmodified_output() -> None:
    """BetaScale(model, 1.0) leaves the forward output bit-exact (x * 1.0 is exact)."""
    model = _ToyModel()
    x = torch.randn(2, 4, 3)

    baseline = model(x)
    with BetaScale(model, 1.0):  # type: ignore[arg-type]
        scaled = model(x)

    assert torch.equal(baseline, scaled)


def test_only_attention_branch_scaled() -> None:
    """Beta multiplies only the attention-output tensor; attn weights and the residual are untouched."""
    model = _ToyModel(num_blocks=1)
    x = torch.randn(2, 4, 3)
    beta = 2.0

    unscaled_attn_output, unscaled_weights = model.transformer.blocks[0].attn(x)

    captured: dict[str, torch.Tensor] = {}

    def capture_hook(_module: nn.Module, _input: object, output: tuple[torch.Tensor, torch.Tensor]) -> None:
        captured["attn_output"], captured["attn_weights"] = output

    with BetaScale(model, beta):  # type: ignore[arg-type]
        # Registered after BetaScale's own hook, so this one observes the already-scaled output --
        # the same value the block actually uses for its residual add.
        handle = model.transformer.blocks[0].attn.register_forward_hook(capture_hook)
        try:
            block_output = model(x)
        finally:
            handle.remove()

    assert torch.allclose(captured["attn_output"], beta * unscaled_attn_output)
    assert torch.equal(captured["attn_weights"], unscaled_weights)
    assert torch.allclose(block_output, x + beta * unscaled_attn_output)


def test_gradients_flow_through_scaled_branch() -> None:
    """No detach/no_grad in the hook: gradients reach the input under a non-unit beta."""
    model = _ToyModel(num_blocks=1)
    x = torch.randn(2, 4, 3, requires_grad=True)
    beta = 1.2

    with BetaScale(model, beta):  # type: ignore[arg-type]
        output = model(x)
    output.sum().backward()

    assert x.grad is not None
    weight = model.transformer.blocks[0].attn.weight
    expected_grad = torch.ones_like(x) + beta * torch.ones(2, 4, 3) @ weight.T
    assert torch.allclose(x.grad, expected_grad, atol=1e-6)


def test_beta_one_reproduces_unmodified_gradients() -> None:
    """
    BetaScale(model, 1.0) leaves backward-pass gradients bit-exact, not just the forward output.

    The identity-intervention criterion is "beta=1 reproduces unmodified logits
    *and gradients*" -- test_beta_one_reproduces_unmodified_output only covers the forward output.
    """
    model = _ToyModel(num_blocks=1)
    x_baseline = torch.randn(2, 4, 3, requires_grad=True)
    x_scaled = x_baseline.detach().clone()
    x_scaled.requires_grad_(True)  # noqa: FBT003 (Tensor.requires_grad_ is a positional-bool API)

    model(x_baseline).sum().backward()
    with BetaScale(model, 1.0):  # type: ignore[arg-type]
        model(x_scaled).sum().backward()

    assert x_baseline.grad is not None
    assert x_scaled.grad is not None
    assert torch.equal(x_baseline.grad, x_scaled.grad)


def test_hooks_removed_after_exit() -> None:
    """After the context exits, a fresh forward matches the unmodified (beta=1) baseline."""
    model = _ToyModel()
    x = torch.randn(2, 4, 3)
    baseline = model(x)

    with BetaScale(model, 2.0):  # type: ignore[arg-type]
        scaled = model(x)
    assert not torch.equal(baseline, scaled)

    after_exit = model(x)
    assert torch.equal(baseline, after_exit)
    for block in model.transformer.blocks:
        assert len(block.attn._forward_hooks) == 0  # noqa: SLF001 (verifying no leaked hooks)


def _tiny_falcon_model() -> FalconForCausalLM:
    model = FalconForCausalLM(_TINY_FALCON_RW_CONFIG).eval()
    model.requires_grad_(requires_grad=False)
    return model


def test_falcon_rw_style_attention_modules_dispatch() -> None:
    """_attention_modules resolves one real FalconAttention submodule per layer, in layer order."""
    model = _tiny_falcon_model()

    modules = _attention_modules(model)

    assert len(modules) == _TINY_FALCON_RW_CONFIG.num_hidden_layers
    assert modules == [layer.self_attention for layer in model.transformer.h]


def test_falcon_rw_style_beta_one_reproduces_unmodified_output() -> None:
    """
    On a real (tiny) Falcon-RW-style model, beta=1.0 reproduces the unmodified forward bit-exactly.

    This is the empirical check that FalconDecoderLayer really does add the residual *outside*
    self_attention (as this module's docstring claims) -- if that read were wrong, scaling
    self_attention's output by exactly 1.0 would not be a no-op and this would fail.
    """
    model = _tiny_falcon_model()
    input_ids = torch.randint(0, _TINY_FALCON_RW_CONFIG.vocab_size, (1, 6))
    attention_mask = torch.ones(1, 6, dtype=torch.long)

    with torch.no_grad():
        baseline = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        with BetaScale(model, 1.0):
            scaled = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    assert torch.equal(baseline, scaled)


def test_falcon_parallel_attn_variant_rejected() -> None:
    """Falcon's parallel-attention variant is explicitly unsupported (attn output isn't residual-external)."""
    config = FalconConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        alibi=True,
        new_decoder_architecture=False,
        parallel_attn=True,
        multi_query=False,
    )
    model = FalconForCausalLM(config).eval()

    with pytest.raises(NotImplementedError):
        _attention_modules(model)


def test_unsupported_model_type_rejected() -> None:
    """An architecture with no dispatch case (and no MPT-shaped duck-typed attributes) raises TypeError."""
    with pytest.raises(TypeError):
        _attention_modules(nn.Linear(2, 2))  # type: ignore[arg-type]


def _tiny_bloom_model() -> BloomForCausalLM:
    model = BloomForCausalLM(_TINY_BLOOM_CONFIG).eval()
    model.requires_grad_(requires_grad=False)
    return model


def test_bloom_attention_modules_dispatch_targets_dense_not_self_attention() -> None:
    """_attention_modules resolves self_attention.dense (not self_attention itself) per layer for Bloom."""
    model = _tiny_bloom_model()

    modules = _attention_modules(model)

    assert len(modules) == _TINY_BLOOM_CONFIG.n_layer
    assert modules == [layer.self_attention.dense for layer in model.transformer.h]  # pyright: ignore[reportAttributeAccessIssue]


def test_bloom_beta_one_reproduces_unmodified_output() -> None:
    """
    On a real (tiny) Bloom model, beta=1.0 reproduces the unmodified forward bit-exactly.

    This is the empirical check that hooking self_attention.dense (rather than self_attention
    itself, which internally adds the residual before returning) really does isolate the
    pre-residual attention branch -- if that read were wrong, scaling dense's output by exactly 1.0
    would not be a no-op and this would fail.
    """
    model = _tiny_bloom_model()
    input_ids = torch.randint(0, _TINY_BLOOM_CONFIG.vocab_size, (1, 6))
    attention_mask = torch.ones(1, 6, dtype=torch.long)

    with torch.no_grad():
        baseline = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        with BetaScale(model, 1.0):
            scaled = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    assert torch.equal(baseline, scaled)


def test_bloom_beta_scales_dense_output_before_residual_add() -> None:
    """
    A non-unit beta on Bloom scales layer 0's self_attention.dense output by exactly beta.

    Single-layer model, so layer 0's *input* is identical between the scaled and unscaled runs (raw
    embeddings, unaffected by any beta scaling) -- with more layers, an outer layer's hidden-state
    input would itself already differ due to upstream layers' scaling, confounding a direct
    per-layer multiplicative comparison. Complements the beta=1.0 bit-exact check: confirms a real
    intervention actually happens (not a hook silently doing nothing), and that BetaScale's generic
    bare-tensor hook branch -- not the tuple-output branch used for MPT/Falcon -- produces the
    expected ``beta * dense_output`` value before Bloom's internal residual add ever sees it.
    """
    config = BloomConfig(vocab_size=32, hidden_size=16, n_layer=1, n_head=2, pretraining_tp=1, slow_but_exact=False)
    model = BloomForCausalLM(config).eval()
    model.requires_grad_(requires_grad=False)
    input_ids = torch.randint(0, config.vocab_size, (1, 6))
    attention_mask = torch.ones(1, 6, dtype=torch.long)
    beta = 2.0
    dense: nn.Module = model.transformer.h[0].self_attention.dense  # pyright: ignore[reportAttributeAccessIssue, reportAssignmentType]

    def capture_hook(storage: list[torch.Tensor]) -> Callable[[nn.Module, object, torch.Tensor], None]:
        def hook(_module: nn.Module, _input_data: object, output: torch.Tensor) -> None:
            storage.append(output)

        return hook

    unscaled: list[torch.Tensor] = []
    handle = dense.register_forward_hook(capture_hook(unscaled))
    with torch.no_grad():
        baseline = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    handle.remove()

    scaled_captured: list[torch.Tensor] = []
    with BetaScale(model, beta):
        handle = dense.register_forward_hook(capture_hook(scaled_captured))
        try:
            with torch.no_grad():
                scaled = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        finally:
            handle.remove()

    assert not torch.equal(baseline, scaled)
    assert len(unscaled) == len(scaled_captured) == 1
    assert torch.allclose(scaled_captured[0], beta * unscaled[0], atol=1e-5)


def test_bloom_slow_but_exact_tensor_parallel_path_rejected() -> None:
    """The pretraining_tp>1 + slow_but_exact path bypasses self_attention.dense and is rejected."""
    config = BloomConfig(
        vocab_size=32,
        hidden_size=16,
        n_layer=2,
        n_head=2,
        pretraining_tp=2,
        slow_but_exact=True,
    )
    model = BloomForCausalLM(config).eval()

    with pytest.raises(NotImplementedError):
        _attention_modules(model)


def test_qwen2_attention_modules_and_beta_one() -> None:
    """Qwen2 is hooked at self_attn, before the decoder layer's external residual addition."""
    model = Qwen2ForCausalLM(_TINY_QWEN2_CONFIG).eval()
    model.requires_grad_(requires_grad=False)
    modules = _attention_modules(model)
    assert modules == [layer.self_attn for layer in model.model.layers]

    input_ids = torch.randint(0, _TINY_QWEN2_CONFIG.vocab_size, (1, 6))
    attention_mask = torch.ones(1, 6, dtype=torch.long)
    with torch.no_grad():
        baseline = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        with BetaScale(model, 1.0):
            scaled = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    assert torch.equal(baseline, scaled)
