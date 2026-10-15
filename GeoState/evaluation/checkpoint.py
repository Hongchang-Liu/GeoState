"""Load the paper checkpoints without importing training or visualization code."""

from __future__ import annotations

import warnings
from pathlib import Path

import torch

from GeoState.core.config import AlignmentConfig, ExperimentConfig, ModelConfig
from GeoState.models.tokenizer import GeometrySceneModel
from GeoState.models.vggt_adapter import VGGTOmegaAdapter


def load_geostate_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    vggt_checkpoint: str | Path | None = None,
) -> tuple[ExperimentConfig, VGGTOmegaAdapter, GeometrySceneModel]:
    """Build GeoState and load inference weights from a compatible checkpoint."""
    config = ExperimentConfig.load(config_path)
    if vggt_checkpoint is not None:
        config.vggt.checkpoint = str(vggt_checkpoint)

    # Evaluation does not use optimizer or scheduler state. Memory mapping keeps
    # those potentially large objects off the eager checkpoint-loading path.
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    saved_config = checkpoint.get("config", {})
    if "model" in saved_config:
        saved_model = saved_config["model"]
        active_fields = set(ModelConfig.__dataclass_fields__)
        inactive_fields = sorted(set(saved_model) - active_fields)
        if inactive_fields:
            warnings.warn(
                "Ignoring checkpoint model-config fields absent from the released "
                f"architecture: {inactive_fields}",
                stacklevel=2,
            )
        config.model = ModelConfig(
            **{key: value for key, value in saved_model.items() if key in active_fields}
        )
    if "alignment" in saved_config:
        # Older training snapshots store these unused knobs. They never
        # controlled alignment and are absent from the released configuration.
        legacy_fields = {
            "huber_delta_rotation_deg", "huber_delta_baseline", "huber_delta_center",
        }
        config.alignment = AlignmentConfig(
            **{
                key: value for key, value in saved_config["alignment"].items()
                if key not in legacy_fields
            }
        )

    adapter = VGGTOmegaAdapter(
        config.vggt.source_root,
        config.vggt.checkpoint,
        config.vggt.checkpoint_min_bytes,
        device,
    )
    model = GeometrySceneModel(adapter, config.model, config.alignment)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    missing = [key for key in missing if not key.startswith("vggt.")]
    unexpected = [key for key in unexpected if not key.startswith("vggt.")]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    return config, adapter, model.to(device).eval()
