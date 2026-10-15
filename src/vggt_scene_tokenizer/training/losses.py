"""Geometry objective used by the released tokenizer training stages."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vggt_scene_tokenizer.core.types import DepthPrediction


@dataclass
class LossOutput:
    total: torch.Tensor
    log_depth: torch.Tensor
    gradient: torch.Tensor


def _weighted_mean_per_scene(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    value = value.reshape(value.shape[0], -1)
    weight = weight.reshape(weight.shape[0], -1)
    return (value * weight).sum(1) / weight.sum(1).clamp_min(1e-8)


def _masked_log_depth_l1_per_scene(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    confidence_weight: torch.Tensor,
) -> torch.Tensor:
    if prediction.ndim == 4:
        prediction = prediction[:, 0]
    error = (prediction.clamp_min(1e-6).log() - target.clamp_min(1e-6).log()).abs()
    return _weighted_mean_per_scene(error, confidence_weight * valid)


def _masked_gradient_loss_per_scene(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    confidence_weight: torch.Tensor,
) -> torch.Tensor:
    if prediction.ndim == 4:
        prediction = prediction[:, 0]
    prediction = prediction.clamp_min(1e-6).log()
    target = target.clamp_min(1e-6).log()
    dx_valid = valid[..., :, 1:] & valid[..., :, :-1]
    dy_valid = valid[..., 1:, :] & valid[..., :-1, :]
    dx_weight = torch.minimum(confidence_weight[..., :, 1:], confidence_weight[..., :, :-1]) * dx_valid
    dy_weight = torch.minimum(confidence_weight[..., 1:, :], confidence_weight[..., :-1, :]) * dy_valid
    dx = ((prediction[..., :, 1:] - prediction[..., :, :-1]) - (target[..., :, 1:] - target[..., :, :-1])).abs()
    dy = ((prediction[..., 1:, :] - prediction[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])).abs()
    return _weighted_mean_per_scene(dx, dx_weight) + _weighted_mean_per_scene(dy, dy_weight)


def geometry_losses_per_scene(
    target_prediction: DepthPrediction,
    target_depth: torch.Tensor,
    valid: torch.Tensor,
    confidence_weight: torch.Tensor,
    *,
    log_depth_weight: float = 1.0,
    gradient_weight: float = 0.1,
) -> LossOutput:
    log_depth = _masked_log_depth_l1_per_scene(
        target_prediction.z_depth, target_depth, valid, confidence_weight
    )
    gradient = (
        _masked_gradient_loss_per_scene(
            target_prediction.z_depth, target_depth, valid, confidence_weight
        )
        if gradient_weight
        else torch.zeros_like(log_depth)
    )
    return LossOutput(log_depth_weight * log_depth + gradient_weight * gradient, log_depth, gradient)
