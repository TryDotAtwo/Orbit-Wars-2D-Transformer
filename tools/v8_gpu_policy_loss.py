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
    """Compute policy loss fully on CUDA tensors.

    ``rewards`` is shaped ``[requests]`` or ``[requests, slots]`` and should use
    +1 for wins and -1 for losses/draws.  The function maximizes selected-action
    log probability for positive rewards and minimizes it for negative rewards.
    """
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

    values = outputs.get("value")
    if values is None:
        advantages = rewards
        value_loss = selected_logprob.new_zeros(())
        mean_value = selected_logprob.new_zeros(())
    else:
        values = values.to(device=device, dtype=torch.float32)
        if values.ndim != 1 or values.shape[0] != labels_fire.shape[0]:
            raise ValueError(f"bad value shape {tuple(values.shape)} expected {(labels_fire.shape[0],)}")
        request_rewards = rewards.mean(dim=1)
        advantages = rewards - values.detach()[:, None]
        value_loss = nn.functional.mse_loss(values, request_rewards)
        mean_value = values.detach().mean()

    invalid_mask = batch.labels_confidence.to(device=device, dtype=torch.float32) < 0.0
    invalid_loss = selected_logprob.new_zeros(())
    if invalid_mask.any():
        invalid_fire = nn.functional.binary_cross_entropy_with_logits(
            outputs["fire_logits"].float(),
            torch.zeros_like(outputs["fire_logits"], dtype=torch.float32),
            reduction="none",
        )
        invalid_loss = invalid_fire.masked_select(invalid_mask).mean()

    old_logprob = batch.old_logprob
    if old_logprob is None:
        old_logprob = selected_logprob.detach()
    else:
        old_logprob = old_logprob.to(device=device, dtype=torch.float32)
        if old_logprob.shape != selected_logprob.shape:
            raise ValueError(f"bad old_logprob shape {tuple(old_logprob.shape)} expected {tuple(selected_logprob.shape)}")
    log_ratio = selected_logprob - old_logprob.detach()
    ratio = log_ratio.clamp(min=-20.0, max=20.0).exp()
    clipped_ratio = ratio.clamp(1.0 - ppo_clip, 1.0 + ppo_clip)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)

    # PPO clipped policy objective.
    policy_loss = -surrogate.mean()
    loss = policy_loss + value_weight * value_loss + invalid_action_weight * invalid_loss
    active_count = (labels_fire > 0.5).sum().detach()
    approx_kl = (old_logprob.detach() - selected_logprob.detach()).mean()
    clip_fraction = ((ratio.detach() - 1.0).abs() > ppo_clip).to(torch.float32).mean()
    metrics = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "invalid_loss": float(invalid_loss.detach().cpu()),
        "active_slots": float(active_count.cpu()),
        "mean_reward": float(rewards.detach().mean().cpu()),
        "mean_value": float(mean_value.cpu()),
        "invalid_slots": float(invalid_mask.sum().detach().cpu()),
        "mean_selected_logprob": float(selected_logprob.detach().mean().cpu()),
        "approx_kl": float(approx_kl.cpu()),
        "clip_fraction": float(clip_fraction.cpu()),
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
