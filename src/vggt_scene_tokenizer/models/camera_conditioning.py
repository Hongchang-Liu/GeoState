"""Paper camera predictor conditioned on current frozen VGGT patch tokens."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn

from vggt_scene_tokenizer.models.camera_layers import CrossAttentionBlock
from vggt_scene_tokenizer.models.camera_prediction import (
    AnchoredCameraPredictor,
    AnchoredPredictorConfig,
    HorizonCrossAttentionBlock,
    TrajectoryStatistics,
    invert_rigid,
    se3_exp,
    se3_log,
)


@dataclass(frozen=True)
class VGGTSpatialCameraConfig:
    history: int = 4
    feature_dim: int = 2048
    tokens_per_frame: int = 48
    dim: int = 256
    scene_queries: int = 8
    scene_layers: int = 2
    decoder_layers: int = 2
    heads: int = 8
    decoder_mlp_ratio: float = 2.0
    dropout: float = 0.1
    time_bands: int = 8
    step_seconds: float = 0.2
    correction_limit: float = 2.0


class VGGTSpatialCameraPredictor(nn.Module):
    """Frozen motion predictor plus a current-image visual residual."""

    def __init__(
        self,
        pose_model: AnchoredCameraPredictor,
        config: VGGTSpatialCameraConfig,
    ) -> None:
        super().__init__()
        if pose_model.config.history != config.history:
            raise ValueError("pose model history mismatch")
        if pose_model.config.dim != config.dim:
            raise ValueError("pose and adapter dimensions must match")
        self.pose_model = pose_model
        self.config = config
        self.pose_model.requires_grad_(False)
        self.pose_model.eval()
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, config.dim),
            nn.GELU(),
        )
        self.spatial_position = nn.Parameter(
            torch.zeros(1, config.tokens_per_frame, config.dim)
        )
        self.visual_queries = nn.Parameter(
            torch.randn(1, config.scene_queries, config.dim) * 0.02
        )
        self.visual_blocks = nn.ModuleList(
            CrossAttentionBlock(config.dim, config.heads, config.dropout)
            for _ in range(config.scene_layers)
        )
        time_features = 2 + 2 * config.time_bands
        self.time_projection = nn.Sequential(
            nn.Linear(time_features, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, config.dim),
        )
        decoder_config = AnchoredPredictorConfig(
            history=config.history,
            dim=config.dim,
            decoder_layers=config.decoder_layers,
            heads=config.heads,
            mlp_ratio=config.decoder_mlp_ratio,
            dropout=config.dropout,
        )
        self.decoder = nn.ModuleList(
            HorizonCrossAttentionBlock(decoder_config)
            for _ in range(config.decoder_layers)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.dim),
            nn.Linear(config.dim, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, 6),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.pose_model.eval()
        return self

    def time_features(self, future_seconds: torch.Tensor) -> torch.Tensor:
        normalized = future_seconds / (10 * self.config.step_seconds)
        frequencies = 2.0 ** torch.arange(
            self.config.time_bands,
            device=future_seconds.device,
            dtype=future_seconds.dtype,
        )
        angles = math.pi * normalized[:, None] * frequencies
        return torch.cat(
            (
                normalized[:, None],
                normalized.square()[:, None],
                angles.sin(),
                angles.cos(),
            ),
            dim=-1,
        )

    def pose_memory(self, history: torch.Tensor) -> torch.Tensor:
        motion = se3_log(invert_rigid(history[:, :-1]) @ history[:, 1:])
        acceleration = torch.zeros_like(motion)
        acceleration[:, 1:] = motion[:, 1:] - motion[:, :-1]
        features = torch.cat(
            (
                (motion - self.pose_model.motion_mean) / self.pose_model.motion_std,
                (acceleration - self.pose_model.acceleration_mean)
                / self.pose_model.acceleration_std,
            ),
            dim=-1,
        )
        memory = self.pose_model.input_projection(features)
        memory = memory + self.pose_model.history_position[
            :, -(self.config.history - 1) :
        ]
        return self.pose_model.encoder(memory)

    def encode_scene(self, feature: torch.Tensor) -> torch.Tensor:
        expected = (self.config.tokens_per_frame, self.config.feature_dim)
        if feature.shape[-2:] != expected:
            raise ValueError(f"current_features must end in {expected}")
        patches = self.feature_projection(feature) + self.spatial_position
        queries = self.visual_queries.expand(feature.shape[0], -1, -1)
        for block in self.visual_blocks:
            queries = block(queries, patches)
        return queries

    def forward(
        self,
        history: torch.Tensor,
        future_seconds: torch.Tensor,
        *,
        current_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            pose_output = self.pose_model(history, future_seconds)
            motion_memory = self.pose_memory(history)
        scene_memory = self.encode_scene(current_features)
        memory = torch.cat((motion_memory, scene_memory), dim=1)
        query = self.time_projection(self.time_features(future_seconds))[None]
        query = query.expand(history.shape[0], -1, -1)
        for block in self.decoder:
            query = block(query, memory)
        normalized = torch.tanh(self.output(query)) * self.config.correction_limit
        correction = normalized * self.pose_model.residual_scale(future_seconds)[None]
        predicted = pose_output["predicted_relative"] @ se3_exp(correction)
        return {
            **pose_output,
            "pose_only_relative": pose_output["predicted_relative"],
            "predicted_relative": predicted,
            "scene_correction": correction,
            "scene_memory": scene_memory,
        }


def _build_pose_model(
    model_config: dict,
    statistics: dict,
    device: torch.device,
) -> AnchoredCameraPredictor:
    raw_config = dict(model_config)
    if raw_config.pop("prediction_target", "residual") != "residual":
        raise ValueError("The paper camera predictor requires residual pose prediction")
    if raw_config.pop("architecture", "transformer") != "transformer":
        raise ValueError("The paper camera predictor requires the Transformer architecture")
    raw_config.pop("recent_history", None)
    config = AnchoredPredictorConfig(**raw_config)
    return AnchoredCameraPredictor(
        config, TrajectoryStatistics.from_dict(statistics)
    ).to(device)


def _paper_camera_config(values: dict) -> VGGTSpatialCameraConfig:
    unsupported_defaults = {
        "train_pose": False,
        "scene_token_dim": 0,
        "horizon_calibration": False,
        "use_vggt_features": True,
        "dino_feature_dim": 0,
    }
    for key, expected in unsupported_defaults.items():
        if values.get(key, expected) != expected:
            raise ValueError(f"The paper camera checkpoint requires {key}={expected!r}")
    names = {field.name for field in fields(VGGTSpatialCameraConfig)}
    return VGGTSpatialCameraConfig(
        **{key: value for key, value in values.items() if key in names}
    )


def load_vggt_spatial_camera_checkpoint(
    path: str,
    device: torch.device,
) -> tuple[VGGTSpatialCameraPredictor, dict, VGGTSpatialCameraConfig]:
    """Load the self-contained paper camera predictor."""
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    pose_model = _build_pose_model(
        checkpoint["pose_model_config"], checkpoint["pose_statistics"], device
    )
    config = _paper_camera_config(checkpoint["camera_config"])
    model = VGGTSpatialCameraPredictor(pose_model, config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint, config
