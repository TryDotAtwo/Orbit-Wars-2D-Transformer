"""GPU-resident policy-gradient loss for OWV8 decoded action slots."""
from __future__ import annotations

import torch
from torch import nn

from tools.v8_resident_device_batch import ResidentBatchTensorView


def compute_resident_policy_loss(
    outputs: dict[str, torch.Tensor],
    batch: ResidentBatchTensorView,
    rewards: torch.Tensor,
    *,
    source_weight: float = 1.0,
    target_weight: float = 1.0,
    amount_weight: float = 0.75,
    fire_weight: float = 0.35,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute policy loss fully on CUDA tensors.

    ``rewards`` is shaped ``[requests]`` or ``[requests, slots]`` and should use
    +1 for wins and -1 for losses/draws.  The function maximizes selected-action
    log probability for positive rewards and minimizes it for negative rewards.
    """
    device = outputs["fire_logits"].device
    labels_fire = batch.labels_fire.to(device=device, dtype=torch.float32)
    labels_source = batch.labels_source.to(device=device, dtype=torch.long)
    labels_target = batch.labels_target.to(device=device, dtype=torch.long)
    labels_amount = batch.labels_amount.to(device=device, dtype=torch.long)
    rewards = rewards.to(device=device, dtype=torch.float32)
    if rewards.ndim == 1:
        rewards = rewards[:, None].expand_as(labels_fire)
    if rewards.shape != labels_fire.shape:
        raise ValueError(f"bad rewards shape {tuple(rewards.shape)} expected {tuple(labels_fire.shape)}")

    active = labels_fire > 0.5
    fire_logprob = -nn.functional.binary_cross_entropy_with_logits(
        outputs["fire_logits"],
        labels_fire,
        reduction="none",
    )
    selected_logprob = fire_weight * fire_logprob
    if active.any():
        source_logprob = outputs["source_logits"].log_softmax(dim=-1)
        target_logprob = outputs["target_logits"].log_softmax(dim=-1)
        amount_logprob = outputs["amount_logits"].log_softmax(dim=-1)
        safe_source = labels_source.clamp_min(0)
        safe_target = labels_target.clamp_min(0)
        safe_amount = labels_amount.clamp_min(0)
        selected_logprob = selected_logprob + active * (
            source_weight * source_logprob.gather(-1, safe_source.unsqueeze(-1)).squeeze(-1)
            + target_weight * target_logprob.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
            + amount_weight * amount_logprob.gather(-1, safe_amount.unsqueeze(-1)).squeeze(-1)
        )

    # Policy gradient objective: minimize -reward * logprob.
    loss = -(rewards * selected_logprob).mean()
    active_count = active.sum().detach()
    metrics = {
        "loss": float(loss.detach().cpu()),
        "active_slots": float(active_count.cpu()),
        "mean_reward": float(rewards.detach().mean().cpu()),
        "mean_selected_logprob": float(selected_logprob.detach().mean().cpu()),
    }
    return loss, metrics

