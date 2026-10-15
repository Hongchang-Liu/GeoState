"""Anchored continuous-horizon camera trajectory prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


def invert_rigid(transform: torch.Tensor) -> torch.Tensor:
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    result = torch.zeros_like(transform)
    result[..., :3, :3] = rotation.transpose(-1, -2)
    result[..., :3, 3] = -(rotation.transpose(-1, -2) @ translation[..., None]).squeeze(-1)
    result[..., 3, 3] = 1
    return result


def skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero), dim=-1
    ).reshape(*vector.shape[:-1], 3, 3)


def so3_exp(vector: torch.Tensor) -> torch.Tensor:
    angle2 = vector.square().sum(-1)
    angle = angle2.clamp_min(1e-12).sqrt()
    a = torch.where(
        angle2 < 1e-8,
        1 - angle2 / 6 + angle2.square() / 120,
        torch.sin(angle) / angle,
    )
    b = torch.where(
        angle2 < 1e-8,
        0.5 - angle2 / 24 + angle2.square() / 720,
        (1 - torch.cos(angle)) / angle2.clamp_min(1e-12),
    )
    omega = skew(vector)
    identity = torch.eye(3, device=vector.device, dtype=vector.dtype).expand_as(omega)
    return identity + a[..., None, None] * omega + b[..., None, None] * (omega @ omega)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(
        -1 + 1e-7, 1 - 1e-7
    )
    angle = torch.acos(cosine)
    factor = angle / (2 * torch.sin(angle).clamp_min(1e-7))
    factor = torch.where(angle < 1e-3, 0.5 + angle.square() / 12, factor)
    return factor[..., None] * vee


def so3_left_jacobian(vector: torch.Tensor) -> torch.Tensor:
    angle2 = vector.square().sum(-1)
    angle = angle2.clamp_min(1e-12).sqrt()
    b = torch.where(
        angle2 < 1e-8,
        0.5 - angle2 / 24 + angle2.square() / 720,
        (1 - torch.cos(angle)) / angle2.clamp_min(1e-12),
    )
    c = torch.where(
        angle2 < 1e-8,
        1 / 6 - angle2 / 120 + angle2.square() / 5040,
        (angle - torch.sin(angle)) / (angle2 * angle).clamp_min(1e-12),
    )
    omega = skew(vector)
    identity = torch.eye(3, device=vector.device, dtype=vector.dtype).expand_as(omega)
    return identity + b[..., None, None] * omega + c[..., None, None] * (omega @ omega)


def so3_left_jacobian_inverse(vector: torch.Tensor) -> torch.Tensor:
    angle2 = vector.square().sum(-1)
    angle = angle2.clamp_min(1e-12).sqrt()
    coefficient = torch.where(
        angle2 < 1e-8,
        1 / 12 + angle2 / 720 + angle2.square() / 30240,
        1 / angle2.clamp_min(1e-12)
        - (1 + torch.cos(angle))
        / (2 * angle * torch.sin(angle)).clamp_min(1e-12),
    )
    omega = skew(vector)
    identity = torch.eye(3, device=vector.device, dtype=vector.dtype).expand_as(omega)
    return identity - 0.5 * omega + coefficient[..., None, None] * (omega @ omega)


def se3_exp(vector: torch.Tensor) -> torch.Tensor:
    translation_vector, rotation_vector = vector[..., :3], vector[..., 3:]
    result = torch.zeros(*vector.shape[:-1], 4, 4, device=vector.device, dtype=vector.dtype)
    result[..., :3, :3] = so3_exp(rotation_vector)
    result[..., :3, 3] = (
        so3_left_jacobian(rotation_vector) @ translation_vector[..., None]
    ).squeeze(-1)
    result[..., 3, 3] = 1
    return result


def se3_log(transform: torch.Tensor) -> torch.Tensor:
    rotation_vector = so3_log(transform[..., :3, :3])
    translation_vector = (
        so3_left_jacobian_inverse(rotation_vector) @ transform[..., :3, 3, None]
    ).squeeze(-1)
    return torch.cat((translation_vector, rotation_vector), dim=-1)


def rotation_angle(rotation: torch.Tensor) -> torch.Tensor:
    cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
    return torch.acos(cosine)


@dataclass(frozen=True)
class TrajectoryStatistics:
    motion_mean: torch.Tensor
    motion_std: torch.Tensor
    acceleration_mean: torch.Tensor
    acceleration_std: torch.Tensor
    residual_std: torch.Tensor

    @classmethod
    def from_dict(cls, values: dict[str, torch.Tensor]) -> "TrajectoryStatistics":
        keys = (
            "motion_mean", "motion_std", "acceleration_mean",
            "acceleration_std", "residual_std",
        )
        return cls(**{key: values[key].float() for key in keys})


@dataclass(frozen=True)
class AnchoredPredictorConfig:
    history: int = 6
    dim: int = 256
    encoder_layers: int = 4
    decoder_layers: int = 3
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    time_bands: int = 8
    step_seconds: float = 0.2
    correction_limit: float = 6.0


class HorizonCrossAttentionBlock(nn.Module):
    def __init__(self, config: AnchoredPredictorConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.dim)
        self.memory_norm = nn.LayerNorm(config.dim)
        self.attention = nn.MultiheadAttention(
            config.dim, config.heads, dropout=config.dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(config.dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, int(config.dim * config.mlp_ratio)),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(int(config.dim * config.mlp_ratio), config.dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.attention(
            self.query_norm(query), normalized_memory, normalized_memory, need_weights=False
        )
        query = query + attended
        return query + self.ffn(self.ffn_norm(query))


class AnchoredCameraPredictor(nn.Module):
    def __init__(
        self,
        config: AnchoredPredictorConfig,
        statistics: TrajectoryStatistics,
    ) -> None:
        super().__init__()
        if not 2 <= config.history <= 10:
            raise ValueError("history must be between 2 and 10 poses")
        self.config = config
        self.register_buffer("motion_mean", statistics.motion_mean.reshape(1, 1, 6))
        self.register_buffer("motion_std", statistics.motion_std.reshape(1, 1, 6))
        self.register_buffer(
            "acceleration_mean", statistics.acceleration_mean.reshape(1, 1, 6)
        )
        self.register_buffer(
            "acceleration_std", statistics.acceleration_std.reshape(1, 1, 6)
        )
        self.register_buffer("residual_std", statistics.residual_std)
        self.input_projection = nn.Linear(12, config.dim)
        self.history_position = nn.Parameter(torch.zeros(1, 9, config.dim))
        encoder_layer = nn.TransformerEncoderLayer(
            config.dim,
            config.heads,
            int(config.dim * config.mlp_ratio),
            config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, config.encoder_layers, nn.LayerNorm(config.dim)
        )
        time_features = 2 + 2 * config.time_bands
        self.time_projection = nn.Sequential(
            nn.Linear(time_features, config.dim), nn.GELU(), nn.Linear(config.dim, config.dim)
        )
        self.decoder = nn.ModuleList(
            HorizonCrossAttentionBlock(config) for _ in range(config.decoder_layers)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, 6),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def time_features(self, future_seconds: torch.Tensor) -> torch.Tensor:
        maximum_training_time = self.residual_std.shape[0] * self.config.step_seconds
        normalized = future_seconds / maximum_training_time
        frequencies = 2.0 ** torch.arange(
            self.config.time_bands, device=future_seconds.device, dtype=future_seconds.dtype
        )
        angles = math.pi * normalized[..., None] * frequencies
        return torch.cat(
            (normalized[..., None], normalized.square()[..., None], angles.sin(), angles.cos()),
            dim=-1,
        )

    def residual_scale(self, future_seconds: torch.Tensor) -> torch.Tensor:
        position = future_seconds / self.config.step_seconds - 1
        maximum = self.residual_std.shape[0] - 1
        lower = position.floor().long().clamp(0, maximum)
        upper = (lower + 1).clamp(0, maximum)
        fraction = (position - lower.to(position.dtype)).clamp(0, 1)
        scale = self.residual_std[lower] * (1 - fraction[:, None]) + self.residual_std[
            upper
        ] * fraction[:, None]
        beyond = (future_seconds / (self.config.step_seconds * (maximum + 1))).clamp_min(1)
        return scale * beyond[:, None]

    def forward(
        self,
        history_c2w: torch.Tensor,
        future_seconds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if history_c2w.shape[-3:] != (self.config.history, 4, 4):
            raise ValueError(
                f"history_c2w must have shape [B,{self.config.history},4,4]"
            )
        if future_seconds.ndim != 1 or bool((future_seconds <= 0).any()):
            raise ValueError("future_seconds must be a positive one-dimensional tensor")
        motions = invert_rigid(history_c2w[:, :-1]) @ history_c2w[:, 1:]
        motion = se3_log(motions)
        acceleration = torch.zeros_like(motion)
        acceleration[:, 1:] = motion[:, 1:] - motion[:, :-1]
        features = torch.cat(
            (
                (motion - self.motion_mean) / self.motion_std,
                (acceleration - self.acceleration_mean) / self.acceleration_std,
            ),
            dim=-1,
        )
        raw_memory = self.input_projection(features)
        raw_memory = raw_memory + self.history_position[:, -(self.config.history - 1) :]
        memory = self.encoder(raw_memory)

        query = self.time_projection(self.time_features(future_seconds))
        query = query[None].expand(history_c2w.shape[0], -1, -1)
        for block in self.decoder:
            query = block(query, memory)
        last_twist = motion[:, -1]
        horizon_steps = future_seconds / self.config.step_seconds
        constant_baseline = se3_exp(last_twist[:, None] * horizon_steps[None, :, None])
        normalized_output = torch.tanh(self.output(query)) * self.config.correction_limit
        correction = normalized_output * self.residual_scale(future_seconds)[None]
        predicted_relative = constant_baseline @ se3_exp(correction)
        return {
            "predicted_relative": predicted_relative,
            "baseline_relative": constant_baseline,
            "correction": correction,
            "normalized_output": normalized_output,
        }

