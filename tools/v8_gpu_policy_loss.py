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
    value_weight: float = 0.5,
    invalid_action_weight: float = 3.0,
    ppo_clip: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the reward-weighted policy loss fully on CUDA tensors."""
    selected_logprob, labels_fire = compute_resident_selected_logprob(
        outputs,
        batch,
        source_weight=source_weight,
        target_weight=target_weight,
        amount_weight=amount_weight,
        fire_weight=fire_weight,
    )
    device = selected_logprob.device
    rewards = rewards.to(device=device, dtype=torch.float32)
    if rewards.ndim == 1:
        rewards = rewards[:, None].expand_as(labels_fire)
    if rewards.shape != labels_fire.shape:
        raise ValueError(f"bad rewards shape {tuple(rewards.shape)} expected {tuple(labels_fire.shape)}")

    policy_loss = -(selected_logprob * rewards).mean()
    loss = policy_loss
    active_count = (labels_fire > 0.5).sum().detach()
    zero = selected_logprob.new_zeros(())
    metrics = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": 0.0,
        "invalid_loss": 0.0,
        "active_slots": float(active_count.cpu()),
        "mean_reward": float(rewards.detach().mean().cpu()),
        "mean_value": 0.0,
        "invalid_slots": 0.0,
        "mean_selected_logprob": float(selected_logprob.detach().mean().cpu()),
        "approx_kl": float(zero.cpu()),
        "clip_fraction": float(zero.cpu()),
    }
    return loss, metrics


def compute_resident_selected_logprob(
    outputs: dict[str, torch.Tensor],
    batch: ResidentBatchTensorView,
    *,
    source_weight: float = 1.0,
    target_weight: float = 1.0,
    amount_weight: float = 0.75,
    fire_weight: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-request/slot selected log-probability and fire labels on GPU."""
    device = outputs["fire_logits"].device
    labels_fire = batch.labels_fire.to(device=device, dtype=torch.float32)
    labels_source = batch.labels_source.to(device=device, dtype=torch.long)
    labels_target = batch.labels_target.to(device=device, dtype=torch.long)
    labels_amount = batch.labels_amount.to(device=device, dtype=torch.long)
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
        active_action_logprob = (
            source_weight * source_logprob.gather(-1, safe_source.unsqueeze(-1)).squeeze(-1)
            + target_weight * target_logprob.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
            + amount_weight * amount_logprob.gather(-1, safe_amount.unsqueeze(-1)).squeeze(-1)
        )
        selected_logprob = selected_logprob + active_action_logprob.masked_fill(~active, 0.0)
    return selected_logprob, labels_fire
