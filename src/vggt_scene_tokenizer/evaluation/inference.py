"""Shared scene-state decoding used by all checkpoint evaluations."""

from __future__ import annotations

from typing import Any

import torch

from vggt_scene_tokenizer.core.types import SceneEncoding, Sim3


def scene_encoding_from_contexts(
    scene_tokens: torch.Tensor,
    contexts: list[dict[str, Any]],
    device: torch.device,
) -> SceneEncoding:
    """Reconstruct the geometry context required to query predicted states."""
    if scene_tokens.shape[0] != len(contexts):
        raise ValueError("Scene-token batch and context-item count disagree")
    height, width = contexts[0]["target_hw"]
    return SceneEncoding(
        memory=scene_tokens.new_empty(scene_tokens.shape[0], 0, scene_tokens.shape[-1]),
        memory_mask=None,
        scene_tokens=scene_tokens,
        reference_c2w_student=torch.stack(
            [context["reference_c2w_student"] for context in contexts]
        ).to(device).float(),
        dataset_to_student=Sim3(
            torch.stack(
                [context["dataset_to_student_scale"] for context in contexts]
            ).to(device).float().reshape(-1),
            torch.stack(
                [context["dataset_to_student_rotation"] for context in contexts]
            ).to(device).float(),
            torch.stack(
                [context["dataset_to_student_translation"] for context in contexts]
            ).to(device).float(),
        ),
        scene_scale=torch.stack(
            [context["scene_scale"] for context in contexts]
        ).to(device).float().reshape(-1),
        input_depth=scene_tokens.new_empty(len(contexts), 0, height, width),
        input_confidence=scene_tokens.new_empty(len(contexts), 0, height, width),
        input_points_reference=scene_tokens.new_empty(
            len(contexts), 0, height, width, 3
        ),
        patch_hw=(height // 16, width // 16),
        metadata={},
    )


def decode_metric_depth(
    geometry_model: torch.nn.Module,
    scene_tokens: torch.Tensor,
    contexts: list[dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    """Decode scene states at query cameras and restore metric depth scale."""
    encoding = scene_encoding_from_contexts(scene_tokens, contexts, device)
    prediction = geometry_model.decode_depth(
        encoding,
        torch.stack([context["target_c2w"] for context in contexts]).to(device).float(),
        torch.stack([context["target_intrinsics"] for context in contexts]).to(device).float(),
        tuple(contexts[0]["target_hw"]),
        query_is_dataset_camera=True,
    )
    depth = prediction.z_depth
    if depth.ndim == 4:
        depth = depth[:, 0]
    meters_per_unit = (
        encoding.scene_scale / encoding.dataset_to_student.scale.clamp_min(1e-8)
    )
    return depth * meters_per_unit[:, None, None]
