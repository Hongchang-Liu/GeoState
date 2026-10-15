from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Sim3:
    """Similarity transform mapping source coordinates into a target frame."""

    scale: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor

    def to(self, *args: Any, **kwargs: Any) -> "Sim3":
        """Move contained tensors to the requested device or dtype."""
        return Sim3(
            self.scale.to(*args, **kwargs),
            self.rotation.to(*args, **kwargs),
            self.translation.to(*args, **kwargs),
        )


@dataclass
class AlignmentDiagnostics:
    """Per-sample diagnostics produced by robust camera-frame alignment."""
    rotation_error_deg: torch.Tensor
    rotation_error_p90_deg: torch.Tensor
    relative_baseline_error: torch.Tensor
    relative_baseline_error_p90: torch.Tensor
    center_error: torch.Tensor
    center_error_p90: torch.Tensor
    min_baseline: torch.Tensor
    condition: list[str]
    valid: torch.Tensor


@dataclass
class VGGTOutput:
    """Frozen VGGT features, depth, confidence, cameras, and world points."""
    patch_features: list[torch.Tensor]
    depth: torch.Tensor
    confidence: torch.Tensor
    world_points: torch.Tensor
    c2w: torch.Tensor
    intrinsics: torch.Tensor
    patch_size: int
    patch_hw: tuple[int, int]


@dataclass
class SceneEncoding:
    """View-decoupled scene state with the geometry needed for later queries."""
    memory: torch.Tensor
    memory_mask: torch.Tensor | None
    scene_tokens: torch.Tensor
    reference_c2w_student: torch.Tensor
    dataset_to_student: Sim3
    scene_scale: torch.Tensor
    input_depth: torch.Tensor
    input_confidence: torch.Tensor
    input_points_reference: torch.Tensor
    patch_hw: tuple[int, int]            # (H_patches, W_patches)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DepthPrediction:
    """Decoded camera z-depth."""
    z_depth: torch.Tensor
    log_z_depth: torch.Tensor
