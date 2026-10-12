"""
Token blank-out (occlusion) positional-influence estimator.

Non-gradient robustness check following Abnar & Zuidema, "Quantifying Attention
Flow in Transformers" (ACL 2020, 2020.acl-main.385). For an input sequence we fix
the model's own next-token prediction ``y = argmax_v z_v(x)`` and its pre-softmax
target logit ``z_y(x)``. For each position ``j`` we zero the token embedding at
``j`` (leaving sequence length, positional processing and attention mask
untouched) and measure ``b_j(x) = |z_y(x) - z_y(x^{(-j)})|``. Forward passes only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel


def _blank_positions(
    base_embeds: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """
    Build a perturbation batch that zeros exactly one embedding row per sample.

    Args:
        base_embeds: Unperturbed embeddings of shape ``[1, n, hidden]``.
        positions: 1-D tensor of positions to blank, one per output row (``[p]``).

    Returns:
        Tensor of shape ``[p, n, hidden]`` where row ``k`` equals ``base_embeds``
        with position ``positions[k]`` zeroed.

    """
    p = positions.shape[0]
    n = base_embeds.shape[1]
    emb = base_embeds.expand(p, n, -1).clone()
    emb[torch.arange(p, device=positions.device), positions, :] = 0.0
    return emb


@torch.no_grad()
def blankout_profile(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_batch_size: int,
) -> tuple[torch.Tensor, int, float]:
    """
    Compute the per-position blank-out influence for a single prompt.

    Args:
        model: Causal LM that accepts ``inputs_embeds``.
        input_ids: Token ids of shape ``[n]`` or ``[1, n]``.
        attention_mask: Attention mask of shape ``[n]`` or ``[1, n]``.
        position_batch_size: Number of perturbed positions to forward at once.

    Returns:
        Tuple ``(b, target_id, z0)`` where ``b`` is a length-``n`` float64 CPU
        tensor of ``|z_y(x) - z_y(x^{(-j)})|``, ``target_id`` is the fixed target
        token ``y``, and ``z0`` is the original pre-softmax target logit ``z_y(x)``.

    """
    embed = model.get_input_embeddings()
    device = embed.weight.device
    input_ids = input_ids.reshape(-1).to(device)
    attention_mask = attention_mask.reshape(-1).to(device)
    seq_len = input_ids.shape[0]

    base_embeds = embed(input_ids).unsqueeze(0)  # [1, n, hidden]
    base_logits = model(
        inputs_embeds=base_embeds,
        attention_mask=attention_mask.unsqueeze(0),
        use_cache=False,
    ).logits[0, -1].float()
    target_id = int(base_logits.argmax().item())
    z0 = float(base_logits[target_id].item())

    scores = torch.zeros(seq_len, dtype=torch.float64)
    for start in range(0, seq_len, position_batch_size):
        positions = torch.arange(
            start,
            min(start + position_batch_size, seq_len),
            device=device,
        )
        perturbed = _blank_positions(base_embeds, positions)
        mask = attention_mask.unsqueeze(0).expand(positions.shape[0], seq_len)
        # Target token y is held fixed across every perturbation.
        logits = model(
            inputs_embeds=perturbed,
            attention_mask=mask,
            use_cache=False,
        ).logits[:, -1, target_id].float()
        scores[start : start + positions.shape[0]] = (z0 - logits).abs().double().cpu()

    return scores, target_id, z0
