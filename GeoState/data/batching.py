from __future__ import annotations

import torch

from GeoState.data.common import select_target
from GeoState.core.types import DepthPrediction


def collate_scene_targets(
    batches: list[dict[str, object]],
    target_slot: int,
) -> dict[str, object]:
    """Collate equal-shaped scene tuples without changing per-scene contents."""
    if not batches:
        raise ValueError("A scene batch must contain at least one scene")
    selected = [select_target(batch, target_slot) for batch in batches]
    image_hw = tuple(int(value) for value in selected[0]["image_hw"])
    input_shape = tuple(torch.as_tensor(selected[0]["input_c2w"]).shape)
    if any(tuple(int(value) for value in batch["image_hw"]) != image_hw for batch in selected):
        raise ValueError("All scenes in a compute batch must have the same image size")
    if any(tuple(torch.as_tensor(batch["input_c2w"]).shape) != input_shape for batch in selected):
        raise ValueError("All scenes in a compute batch must have the same input-view count")

    tensor_fields = (
        "input_c2w",
        "input_intrinsics",
        "target_c2w_dataset",
        "target_intrinsics",
        "target_depth_teacher",
        "target_confidence",
        "target_valid",
        "target_c2w_teacher",
        "target_depth_dataset",
        "target_valid_dataset",
        "target_confidence_dataset",
        "target_scene_scale_dataset",
        "target_uses_dataset_depth",
    )
    result: dict[str, object] = {
        "_scene_batch_size": len(selected),
        "image_hw": image_hw,
        "scene_id": [str(batch["scene_id"]) for batch in selected],
        "dataset_name": [str(batch["dataset_name"]) for batch in selected],
        "input_frame_ids": [list(batch["input_frame_ids"]) for batch in selected],
        "input_frame_indices": [list(batch["input_frame_indices"]) for batch in selected],
        "teacher_register_frame_id": [
            str(batch["teacher_register_frame_id"]) for batch in selected
        ],
        "teacher_register_frame_index": [
            int(batch["teacher_register_frame_index"]) for batch in selected
        ],
        "teacher_scene_scale": [
            batch.get("teacher_scene_scale") for batch in selected
        ],
        "target_frame_id": [str(batch["target_frame_id"]) for batch in selected],
        "target_frame_index": [int(batch["target_frame_index"]) for batch in selected],
        "dataset_to_teacher": {
            key: torch.stack(
                [torch.as_tensor(batch["dataset_to_teacher"][key]) for batch in selected]
            )
            for key in ("scale", "rotation", "translation")
        },
    }
    if all("scene_tokenizer_view_mask" in batch for batch in selected):
        result["scene_tokenizer_view_mask"] = torch.stack(
            [torch.as_tensor(batch["scene_tokenizer_view_mask"]).bool() for batch in selected]
        )
    if all("input_scale_depth_dataset" in batch for batch in selected):
        result["input_scale_depth_dataset"] = torch.stack(
            [torch.as_tensor(batch["input_scale_depth_dataset"]) for batch in selected]
        )
        result["input_scale_valid_dataset"] = torch.stack(
            [torch.as_tensor(batch["input_scale_valid_dataset"]).bool() for batch in selected]
        )
        result["input_scale_view_index"] = torch.as_tensor(
            [int(batch["input_scale_view_index"]) for batch in selected],
            dtype=torch.long,
        )
    for field in (
        "reconstruction_task",
        "scene_token_view_count",
        "student_vggt_timestamp_count",
        "target_timestamp",
        "target_horizon_seconds",
    ):
        if all(field in batch for batch in selected):
            result[field] = [batch[field] for batch in selected]
    result.update(
        {
            field: torch.stack([torch.as_tensor(batch[field]) for batch in selected])
            for field in tensor_fields
        }
    )
    if all("dynamic_downweighted_fraction" in batch for batch in selected):
        result["dynamic_downweighted_fraction"] = torch.stack(
            [
                torch.as_tensor(batch["dynamic_downweighted_fraction"])
                for batch in selected
            ]
        )
    return result


def slice_depth_prediction(prediction: DepthPrediction, index: int) -> DepthPrediction:
    item = slice(index, index + 1)
    return DepthPrediction(
        z_depth=prediction.z_depth[item],
        log_z_depth=prediction.log_z_depth[item],
    )
