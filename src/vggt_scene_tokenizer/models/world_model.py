from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SceneStatePredictorConfig:
    """Configuration for four-state next-scene-state prediction.

    The production tokenizer checkpoint emits 512 learned scene-query slots,
    each with width 512. All temporal states must use the same spatial anchor
    frame before they are passed to this model.
    """

    scene_tokens: int = 512
    scene_dim: int = 512
    context_frames: int = 4
    hidden_dim: int = 1152
    layers: int = 12
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.scene_tokens < 1:
            raise ValueError("scene_tokens must be positive")
        if self.scene_dim < 1 or self.hidden_dim < 1:
            raise ValueError("scene_dim and hidden_dim must be positive")
        if self.context_frames != 4:
            raise ValueError("This predictor is defined for exactly four context states")
        if self.layers < 1:
            raise ValueError("layers must be positive")
        if self.heads < 1 or self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if self.hidden_dim % 2:
            raise ValueError("hidden_dim must be even for broadcast position embeddings")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class _FeedForward(nn.Module):
    def __init__(self, dim: int, ratio: float, dropout: float) -> None:
        super().__init__()
        hidden_dim = int(dim * ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class FactorizedSceneStateBlock(nn.Module):
    """Temporal attention followed by attention across scene-query slots."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.scene_norm = nn.LayerNorm(dim)
        self.scene_attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(dim)
        self.feed_forward = _FeedForward(dim, mlp_ratio, dropout)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        batch, frames, scene_tokens, dim = states.shape

        temporal = states.permute(0, 2, 1, 3).reshape(batch * scene_tokens, frames, dim)
        normalized = self.temporal_norm(temporal)
        attended, _ = self.temporal_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        temporal = temporal + attended
        states = temporal.reshape(batch, scene_tokens, frames, dim).permute(0, 2, 1, 3)

        spatial = states.reshape(batch * frames, scene_tokens, dim)
        normalized = self.scene_norm(spatial)
        attended, _ = self.scene_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        spatial = spatial + attended
        spatial = spatial + self.feed_forward(self.output_norm(spatial))
        return spatial.reshape(batch, frames, scene_tokens, dim)


class SceneStatePredictionBackbone(nn.Module):
    """FR3D-style masked predictor for non-grid learned scene tokens.

    Unlike FR3D's image-grid state, this project's scene state consists of
    fixed Perceiver query slots. The block therefore uses learned slot
    embeddings instead of a 2D positional encoding.
    """

    def __init__(self, config: SceneStatePredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.scene_dim, config.hidden_dim)
        self.mask_token = nn.Parameter(torch.empty(1, 1, 1, config.hidden_dim))
        position_dim = config.hidden_dim // 2
        self.scene_slot_embedding = nn.Parameter(
            torch.empty(1, 1, config.scene_tokens, position_dim)
        )
        self.temporal_embedding = nn.Parameter(
            torch.empty(1, config.context_frames + 1, 1, position_dim)
        )
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.scene_slot_embedding, std=0.02)
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)
        self.layers = nn.ModuleList(
            [
                FactorizedSceneStateBlock(
                    config.hidden_dim,
                    config.heads,
                    config.mlp_ratio,
                    config.dropout,
                )
                for _ in range(config.layers)
            ]
        )

    def _validate_history(self, history: torch.Tensor) -> None:
        expected = (
            self.config.context_frames,
            self.config.scene_tokens,
            self.config.scene_dim,
        )
        if history.ndim != 4 or tuple(history.shape[1:]) != expected:
            raise ValueError(
                "history must have shape [B, context_frames, scene_tokens, scene_dim] "
                f"with trailing dimensions {expected}, got {tuple(history.shape)}"
            )
        if not history.is_floating_point():
            raise TypeError("history must be a floating-point tensor")

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        self._validate_history(history)
        encoded = self.input_projection(history)
        mask = self.mask_token.to(encoded).expand(
            history.shape[0],
            1,
            self.config.scene_tokens,
            -1,
        )
        states = torch.cat((encoded, mask), dim=1)
        temporal_position = self.temporal_embedding.to(encoded).expand(
            1,
            self.config.context_frames + 1,
            self.config.scene_tokens,
            -1,
        )
        scene_position = self.scene_slot_embedding.to(encoded).expand(
            1,
            self.config.context_frames + 1,
            self.config.scene_tokens,
            -1,
        )
        states = states + torch.cat((temporal_position, scene_position), dim=-1)
        for layer in self.layers:
            states = layer(states)
        return states[:, -1]


class _NextSceneStatePredictor(nn.Module):
    def __init__(self, config: SceneStatePredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = SceneStatePredictionBackbone(config)

    def rollout(self, history: torch.Tensor, steps: int) -> torch.Tensor:
        """Autoregressively predict `steps` states using a four-state window."""
        if steps < 1:
            raise ValueError("steps must be positive")
        self.backbone._validate_history(history)
        window = history
        predictions = []
        for _ in range(steps):
            next_state = self(window)
            predictions.append(next_state)
            window = torch.cat((window[:, 1:], next_state[:, None]), dim=1)
        return torch.stack(predictions, dim=1)


class ResidualNextSceneStatePredictor(_NextSceneStatePredictor):
    """Predict a change from the latest state and add it to that state.

    The residual head is zero-initialized, so the untrained model is exactly a
    copy-last predictor. This gives Waymo training a stable starting point.
    """

    def __init__(
        self,
        config: SceneStatePredictorConfig | None = None,
        *,
        zero_initialize_residual: bool = True,
    ) -> None:
        super().__init__(SceneStatePredictorConfig() if config is None else config)
        self.residual_head = nn.Linear(
            self.config.hidden_dim,
            self.config.scene_dim,
            bias=False,
        )
        if zero_initialize_residual:
            nn.init.zeros_(self.residual_head.weight)

    def predict_residual(self, history: torch.Tensor) -> torch.Tensor:
        return self.residual_head(self.backbone(history))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return history[:, -1] + self.predict_residual(history)


def build_next_scene_state_predictor(
    config: SceneStatePredictorConfig | None = None,
) -> ResidualNextSceneStatePredictor:
    """Build the paper's residual next-state predictor."""
    return ResidualNextSceneStatePredictor(config)
