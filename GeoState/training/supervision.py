from __future__ import annotations

from dataclasses import dataclass

import torch

from GeoState.core.geometry import (
    apply_sim3,
    apply_sim3_to_c2w,
    compose_sim3,
    inverse_sim3,
    unproject_z_depth,
    zbuffer_render,
)
from GeoState.core.types import SceneEncoding, Sim3


@dataclass
class TargetSupervision:
    normalized_z_depth: torch.Tensor
    valid: torch.Tensor
    weight: torch.Tensor
    query_c2w_reference: torch.Tensor
    overlap: torch.Tensor
    projection_relative_error: torch.Tensor
    alignment_valid: torch.Tensor
    observed: torch.Tensor
    uses_dataset_depth: torch.Tensor


def _sim3_from_payload(payload: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> Sim3:
    return Sim3(
        payload["scale"].to(device=device, dtype=dtype).reshape(-1),
        payload["rotation"].to(device=device, dtype=dtype).reshape(-1, 3, 3),
        payload["translation"].to(device=device, dtype=dtype).reshape(-1, 3),
    )


def build_target_supervision(
    batch: dict[str, object],
    encoding: SceneEncoding,
    *,
    min_overlap: float = 0.05,
    max_projection_relative_error: float = 0.25,
) -> TargetSupervision:
    device = encoding.scene_tokens.device
    dtype = encoding.reference_c2w_student.dtype
    is_scene_batch = "_scene_batch_size" in batch

    def tensor(name: str, *, tensor_dtype: torch.dtype) -> torch.Tensor:
        value = torch.as_tensor(batch[name], device=device, dtype=tensor_dtype)
        return value if is_scene_batch else value.unsqueeze(0)

    teacher_depth = tensor("target_depth_teacher", tensor_dtype=dtype)
    teacher_confidence = tensor("target_confidence", tensor_dtype=dtype)
    teacher_hard_valid = tensor("target_valid", tensor_dtype=torch.bool)
    uses_dataset = (
        tensor("target_uses_dataset_depth", tensor_dtype=torch.bool).reshape(-1)
        if "target_uses_dataset_depth" in batch
        else torch.zeros(teacher_depth.shape[0], device=device, dtype=torch.bool)
    )
    dataset_depth = (
        tensor("target_depth_dataset", tensor_dtype=dtype)
        if "target_depth_dataset" in batch
        else teacher_depth
    )
    dataset_confidence = (
        tensor("target_confidence_dataset", tensor_dtype=dtype)
        if "target_confidence_dataset" in batch
        else teacher_confidence
    )
    dataset_valid = (
        tensor("target_valid_dataset", tensor_dtype=torch.bool)
        if "target_valid_dataset" in batch
        else teacher_hard_valid
    )
    selector = uses_dataset[:, None, None]
    target_depth = torch.where(selector, dataset_depth, teacher_depth)
    target_confidence = torch.where(selector, dataset_confidence, teacher_confidence)
    target_valid = torch.where(selector, dataset_valid, teacher_hard_valid)
    target_intrinsics = tensor("target_intrinsics", tensor_dtype=dtype)
    teacher_c2w = tensor("target_c2w_teacher", tensor_dtype=dtype)
    dataset_c2w = tensor("target_c2w_dataset", tensor_dtype=dtype)
    dataset_to_teacher = _sim3_from_payload(batch["dataset_to_teacher"], device, dtype)
    teacher_to_student = compose_sim3(
        encoding.dataset_to_student, inverse_sim3(dataset_to_teacher)
    )
    points_teacher = unproject_z_depth(target_depth, target_intrinsics, teacher_c2w)
    points_dataset = unproject_z_depth(target_depth, target_intrinsics, dataset_c2w)
    points_student_teacher = apply_sim3(teacher_to_student, points_teacher)
    points_student_dataset = apply_sim3(encoding.dataset_to_student, points_dataset)
    points_student = torch.where(
        uses_dataset[:, None, None, None],
        points_student_dataset,
        points_student_teacher,
    )
    query_student = apply_sim3_to_c2w(encoding.dataset_to_student, dataset_c2w)
    batch_size = target_depth.shape[0]
    height, width = target_depth.shape[-2:]
    rendered_depth, rendered_valid, rendered_weight = zbuffer_render(
        points_student.reshape(batch_size, -1, 3),
        target_intrinsics,
        query_student,
        (height, width),
        (target_confidence * target_valid).reshape(batch_size, -1),
    )
    valid = rendered_valid & (rendered_weight > 0)
    normalized_depth = rendered_depth / encoding.scene_scale[:, None, None]
    confidence_scale = torch.stack(
        [
            rendered_weight[index][valid[index]].median().clamp_min(1e-8)
            if valid[index].any()
            else rendered_weight.new_tensor(1.0)
            for index in range(batch_size)
        ]
    )
    weight = rendered_weight / confidence_scale[:, None, None]
    weight = weight.clamp(0.1, 10.0) * valid

    reference_w2c = torch.linalg.inv(encoding.reference_c2w_student)
    query_reference = reference_w2c @ query_student
    input_points = encoding.input_points_reference.reshape(batch_size, -1, 3)
    input_confidence = encoding.input_confidence.reshape(batch_size, -1)
    input_depth, input_valid, _ = zbuffer_render(
        input_points,
        target_intrinsics,
        query_reference,
        (height, width),
        input_confidence,
    )
    input_depth = input_depth / encoding.scene_scale[:, None, None]
    overlap_mask = input_valid & valid
    overlap = overlap_mask.float().mean(dim=(-2, -1))
    relative = torch.where(
        overlap_mask,
        (input_depth - normalized_depth).abs() / normalized_depth.clamp_min(1e-6),
        torch.nan,
    )
    projection_error = torch.stack(
        [values[torch.isfinite(values)].median() if torch.isfinite(values).any() else values.new_tensor(torch.inf) for values in relative]
    )
    alignment_valid = (overlap >= min_overlap) & (projection_error <= max_projection_relative_error)
    return TargetSupervision(
        normalized_depth,
        valid,
        weight,
        query_reference,
        overlap,
        projection_error,
        alignment_valid,
        overlap_mask,
        uses_dataset,
    )
