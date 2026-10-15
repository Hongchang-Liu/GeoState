from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Architecture settings for scene encoding and query-conditioned decoding."""

    dim: int = 512
    scene_tokens: int = 512
    perceiver_layers: int = 8
    heads: int = 8
    mlp_ratio: float = 4.0

    point_fourier_bands: int = 10
    ray_fourier_bands: int = 6

    memory_mode: str = "perceiver"

    view_mode: str = "all"

    use_3d_pe: bool = True

    use_relative_camera_pose: bool = False
    use_input_patch_rays: bool = False
    camera_translation_fourier_bands: int = 4
    input_ray_fourier_bands: int = 4

    vggt_feature_projection_dim: int | None = None
    vggt_feature_fusion_hidden_dim: int | None = None

    vggt_dpt_decoder_layers: int = 16
    vggt_dpt_decoder_capture_layers: tuple[int, int, int, int] = (4, 8, 12, 16)
    vggt_dpt_decoder_hidden_dim: int = 1024
    vggt_dpt_state_conditioner_layers: int = 2
    vggt_dpt_branch_layers: int = 2

    decoder_arch: str = "frozen_vggt_dpt"


@dataclass
class DataConfig:
    """Dataset, cache, sampling, preprocessing, and camera-convention settings."""

    root: str = "data/dl3dv"
    cache_root: str = "data/cache/teacher"
    teacher_frames: int = 24
    teacher_sampling: str = "overlap_clip"

    min_input_views: int = 1
    max_input_views: int = 5
    image_resolution: int = 512

    confidence_quantile: float = 0.1
    source_c2w_convention: str = "opengl"


@dataclass
class AlignmentConfig:
    """Robust Sim(3) fitting and projection-validity thresholds."""

    max_rotation_error_deg: float = 10.0
    max_relative_baseline_error: float = 0.25
    max_relative_center_error: float = 0.25
    min_baseline: float = 1e-5

    min_overlap: float = 0.05
    max_projection_relative_error: float = 0.25


@dataclass
class TrainConfig:
    """Optimization, objective, and regularization settings."""

    seed: int = 42
    steps: int = 200000
    grad_accumulation: int = 9
    learning_rate: float = 1e-5
    weight_decay: float = 0.05

    # The total objective combines the active geometry terms using these weights.
    log_depth_weight: float = 1.0
    gradient_weight: float = 0.1

    anchor_view_dropout_probability: float = 0.4
    random_view_dropout_probability: float = 0.4

    output_dir: str = "outputs/stage1"


@dataclass
class VGGTConfig:
    """Paths and integrity threshold for the frozen VGGT-Omega dependency."""

    source_root: str = "third_party/VGGT-omega"
    checkpoint: str = "checkpoints/vggt_omega_1b_512.pt"
    checkpoint_min_bytes: int = 1_000_000_000


@dataclass
class ExperimentConfig:
    """Top-level configuration assembled from the model and pipeline sections."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    vggt: VGGTConfig = field(default_factory=VGGTConfig)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        """Load."""
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls(
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            alignment=AlignmentConfig(**raw.get("alignment", {})),
            train=TrainConfig(**raw.get("train", {})),
            vggt=VGGTConfig(**raw.get("vggt", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """To dict."""
        return asdict(self)
