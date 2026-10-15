from __future__ import annotations

import math
from itertools import combinations

import torch
import torch.nn.functional as F

from vggt_scene_tokenizer.core.types import AlignmentDiagnostics, Sim3


def homogenize_transform(matrix: torch.Tensor) -> torch.Tensor:
    """Homogenize transform."""
    if matrix.shape[-2:] == (4, 4):
        return matrix
    if matrix.shape[-2:] != (3, 4):
        raise ValueError(f"Expected (...,3,4) or (...,4,4), got {matrix.shape}")
    bottom = torch.zeros(*matrix.shape[:-2], 1, 4, dtype=matrix.dtype, device=matrix.device)
    bottom[..., 0, 3] = 1
    return torch.cat((matrix, bottom), dim=-2)


def invert_rigid(matrix: torch.Tensor) -> torch.Tensor:
    """Invert rigid."""
    matrix = homogenize_transform(matrix)
    rotation = matrix[..., :3, :3]        # [..., 3, 3]
    translation = matrix[..., :3, 3]      # [..., 3]
    out = torch.zeros_like(matrix)
    out[..., :3, :3] = rotation.transpose(-1, -2)  # R^T
    # -R^T @ t
    out[..., :3, 3] = -(rotation.transpose(-1, -2) @ translation[..., None]).squeeze(-1)
    out[..., 3, 3] = 1
    return out


def paired_depth_scale(
    prediction: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor | None = None,
    *,
    min_valid_pixels: int = 128,
    max_target_depth: float = 80.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate dataset->prediction scale from matched depth pixels.

    ``prediction`` and ``target`` must describe the same camera and pixel grid.
    The returned scalar is ``median(prediction / target)`` in log space.  It is
    intended to fix the dataset/VGGT gauge once before scene-token encoding;
    it must not be applied to decoder output after prediction.

    Returns one scale, validity flag, and support count per batch element.
    Invalid rows receive a harmless scale of one and must be masked by callers.
    """
    if prediction.ndim == 4:
        if prediction.shape[1] != 1:
            raise ValueError("4D depth prediction must have one channel")
        prediction = prediction[:, 0]
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if prediction.ndim != 3 or target.ndim != 3:
        raise ValueError("Paired depths must be [B,H,W] or prediction [B,1,H,W]")
    if prediction.shape[0] != target.shape[0]:
        raise ValueError("Paired depth batch sizes disagree")
    if confidence is not None:
        if confidence.ndim == 4:
            if confidence.shape[1] != 1:
                raise ValueError("4D confidence must have one channel")
            confidence = confidence[:, 0]
        if confidence.ndim != 3 or confidence.shape[0] != prediction.shape[0]:
            raise ValueError("Confidence shape disagrees with prediction")

    target_hw = target.shape[-2:]
    if prediction.shape[-2:] != target_hw:
        prediction = F.interpolate(
            prediction[:, None].float(), target_hw, mode="bilinear", align_corners=False
        )[:, 0]
        if confidence is not None:
            confidence = F.interpolate(
                confidence[:, None].float(),
                target_hw,
                mode="bilinear",
                align_corners=False,
            )[:, 0]

    prediction = prediction.float()
    target = target.to(device=prediction.device, dtype=torch.float32)
    valid = (
        torch.isfinite(target)
        & (target > 0)
        & (target <= max_target_depth)
        & torch.isfinite(prediction)
        & (prediction > 0)
    )
    if confidence is not None:
        confidence = confidence.to(device=prediction.device, dtype=torch.float32)
        valid &= torch.isfinite(confidence) & (confidence > 0)

    supports = valid.flatten(1).sum(1)
    scales = prediction.new_ones(prediction.shape[0])
    scale_valid = supports >= int(min_valid_pixels)
    for index in range(prediction.shape[0]):
        if not bool(scale_valid[index]):
            continue
        values = valid[index]
        scales[index] = torch.exp(
            (
                torch.log(prediction[index][values])
                - torch.log(target[index][values])
            ).median()
        )
    scale_valid &= torch.isfinite(scales) & (scales > 0)
    scales = torch.where(scale_valid, scales, torch.ones_like(scales))
    return scales, scale_valid, supports


def transform_points(matrix: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Transform points."""
    rotation = matrix[..., :3, :3]          # [..., 3, 3]
    translation = matrix[..., :3, 3]        # [..., 3]

    while rotation.ndim < points.ndim + 1:
        rotation = rotation.unsqueeze(-3)
        translation = translation.unsqueeze(-2)
    # R @ points + t: [..., 3, 3] @ [..., N, 3, 1] -> [..., N, 3] + [..., N, 3]
    return (rotation @ points.unsqueeze(-1)).squeeze(-1) + translation


def camera_centers(c2w: torch.Tensor) -> torch.Tensor:
    """Camera centers."""
    return c2w[..., :3, 3]


def pixel_grid(
    height: int, width: int, *, device: torch.device | str, dtype: torch.dtype
) -> torch.Tensor:
    """Pixel grid."""
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    return torch.stack((u, v, torch.ones_like(u)), dim=-1)


def unproject_z_depth(depth: torch.Tensor, intrinsics: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """Unproject z depth."""
    if depth.shape[-1:] == (1,):
        depth = depth[..., 0]
    height, width = depth.shape[-2:]

    grid = pixel_grid(height, width, device=depth.device, dtype=depth.dtype)
    grid = grid.expand(*depth.shape[:-2], height, width, 3)

    inv_k = torch.linalg.inv(intrinsics)  # [..., 3, 3]
    while inv_k.ndim < grid.ndim + 1:
        inv_k = inv_k.unsqueeze(-3)
    camera_rays = (inv_k @ grid.unsqueeze(-1)).squeeze(-1)  # [..., H, W, 3]

    points_camera = camera_rays * depth[..., None]

    return transform_points(c2w, points_camera)


def project_world_points(
    points_world: torch.Tensor,   # [..., N, 3]
    intrinsics: torch.Tensor,     # [..., 3, 3]
    c2w: torch.Tensor,            # [..., 4, 4]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world points."""

    points_camera = transform_points(invert_rigid(c2w), points_world)  # [..., N, 3]

    rotation_k = intrinsics  # [..., 3, 3]
    while rotation_k.ndim < points_camera.ndim + 1:
        rotation_k = rotation_k.unsqueeze(-3)
    homogeneous = (rotation_k @ points_camera.unsqueeze(-1)).squeeze(-1)  # [..., N, 3]

    z = points_camera[..., 2]
    uv = homogeneous[..., :2] / homogeneous[..., 2:3].clamp_min(1e-8)

    return uv, z


def zbuffer_render(
    points_world: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
    image_hw: tuple[int, int],
    point_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zbuffer render."""
    if points_world.ndim != 3:
        raise ValueError("points_world must have shape [B,M,3]")
    batch, count, _ = points_world.shape
    height, width = image_hw

    uv, z = project_world_points(points_world, intrinsics, c2w)  # [B,M,2], [B,M]

    u = torch.floor(uv[..., 0]).long()
    v = torch.floor(uv[..., 1]).long()

    valid = (z > 1e-8) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    linear = (v * width + u).clamp(0, height * width - 1)

    result_depth = torch.full(
        (batch, height * width), torch.inf, device=points_world.device, dtype=points_world.dtype
    )

    source = torch.where(valid, z, torch.full_like(z, torch.inf))
    result_depth.scatter_reduce_(1, linear, source, reduce="amin", include_self=True)

    result_valid = torch.isfinite(result_depth)
    result_depth = torch.where(result_valid, result_depth, torch.zeros_like(result_depth))

    selected_weight = torch.zeros_like(result_depth)
    if point_weight is None:
        selected_weight[result_valid] = 1
    else:

        closest = valid & ((z - result_depth.gather(1, linear)).abs() <= 1e-5 * z.abs().clamp_min(1))

        selected_weight.scatter_reduce_(
            1, linear, torch.where(closest, point_weight, torch.zeros_like(point_weight)),
            reduce="amax", include_self=True
        )

    return (
        result_depth.view(batch, height, width),
        result_valid.view(batch, height, width),
        selected_weight.view(batch, height, width),
    )


def full_plucker_ray_map(
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
    scene_scale: torch.Tensor,
) -> torch.Tensor:
    """Full plucker ray map."""
    if c2w.ndim == 2:
        c2w = c2w.unsqueeze(0)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    batch = c2w.shape[0]
    height, width = image_hw

    grid = pixel_grid(height, width, device=c2w.device, dtype=c2w.dtype)
    grid = grid.expand(batch, height, width, 3)  # [B, H, W, 3]

    inv_k = torch.linalg.inv(intrinsics)[:, None, None]  # [B, 1, 1, 3, 3]
    directions_camera = (inv_k @ grid.unsqueeze(-1)).squeeze(-1)  # [B, H, W, 3]

    directions_world = (c2w[:, None, None, :3, :3] @ directions_camera.unsqueeze(-1)).squeeze(-1)  # [B, H, W, 3]
    directions_world = F.normalize(directions_world, dim=-1, eps=1e-8)

    scale = scene_scale.reshape(batch, 1, 1, 1).clamp_min(1e-8)
    origins = c2w[:, None, None, :3, 3].expand_as(directions_world) / scale  # [B, H, W, 3]

    moments = torch.linalg.cross(origins, directions_world, dim=-1)  # [B, H, W, 3]

    return torch.cat((directions_world, moments), dim=-1).permute(0, 3, 1, 2).contiguous()


def apply_sim3(sim3: Sim3, points: torch.Tensor) -> torch.Tensor:
    """Apply sim3."""
    scale = sim3.scale
    rotation = sim3.rotation
    translation = sim3.translation

    while scale.ndim < points.ndim - 1:
        scale = scale.unsqueeze(-1)
    while rotation.ndim < points.ndim + 1:
        rotation = rotation.unsqueeze(-3)
        translation = translation.unsqueeze(-2)
    return scale[..., None] * (rotation @ points.unsqueeze(-1)).squeeze(-1) + translation


def apply_sim3_to_c2w(sim3: Sim3, c2w: torch.Tensor) -> torch.Tensor:
    """Apply sim3 to c2w."""
    rotation = sim3.rotation
    scale = sim3.scale
    translation = sim3.translation
    if rotation.ndim == 2:
        rotation = rotation.unsqueeze(0)
        scale = scale.unsqueeze(0) if scale.ndim == 0 else scale
        translation = translation.unsqueeze(0)
    while rotation.ndim < c2w.ndim:
        rotation = rotation.unsqueeze(-3)
        scale = scale.unsqueeze(-1)
        translation = translation.unsqueeze(-2)
    out = c2w.clone()

    out[..., :3, :3] = rotation @ c2w[..., :3, :3]

    centers = c2w[..., :3, 3]
    out[..., :3, 3] = scale[..., None] * (rotation @ centers.unsqueeze(-1)).squeeze(-1) + translation
    return out


def inverse_sim3(sim3: Sim3) -> Sim3:
    """Inverse sim3."""
    rotation_t = sim3.rotation.transpose(-1, -2)
    inv_scale = sim3.scale.reciprocal()
    translation = -inv_scale[..., None] * (rotation_t @ sim3.translation[..., None]).squeeze(-1)
    return Sim3(inv_scale, rotation_t, translation)


def compose_sim3(left: Sim3, right: Sim3) -> Sim3:
    """Compose sim3."""
    scale = left.scale * right.scale
    rotation = left.rotation @ right.rotation
    translation = (
        left.scale[..., None] * (left.rotation @ right.translation[..., None]).squeeze(-1) + left.translation
    )
    return Sim3(scale, rotation, translation)


def _project_so3(matrix: torch.Tensor) -> torch.Tensor:

    u, _, vh = torch.linalg.svd(matrix)
    correction = (
        torch.eye(3, dtype=matrix.dtype, device=matrix.device)
        .expand(*matrix.shape[:-2], 3, 3)
        .clone()
    )
    correction[..., -1, -1] = torch.det(u @ vh)
    return u @ correction @ vh


def _rotation_angle(rotation: torch.Tensor) -> torch.Tensor:

    cosine = ((torch.diagonal(rotation, dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.acos(cosine)


def _huber_weights(residual: torch.Tensor, delta: float) -> torch.Tensor:

    return torch.where(residual <= delta, torch.ones_like(residual), delta / residual.clamp_min(1e-12))


@torch.no_grad()
def estimate_camera_sim3(
    c2w_source: torch.Tensor,
    c2w_destination: torch.Tensor,
    *,
    single_view_scale: torch.Tensor | None = None,
    max_rotation_error_deg: float = 10.0,
    max_relative_baseline_error: float = 0.25,
    max_relative_center_error: float = 1.0,
    min_baseline: float = 1e-5,
    iterations: int = 8,
) -> tuple[Sim3, AlignmentDiagnostics]:
    """Estimate camera sim3."""
    if c2w_source.ndim == 3:
        c2w_source = c2w_source.unsqueeze(0)
        c2w_destination = c2w_destination.unsqueeze(0)

    c2w_destination = c2w_destination.float()
    c2w_source = c2w_source.to(device=c2w_destination.device, dtype=torch.float32)

    if c2w_source.shape != c2w_destination.shape or c2w_source.shape[-2:] != (4, 4):
        raise ValueError("Camera tensors must share shape [B,N,4,4]")
    if c2w_source.shape[1] == 1:
        if single_view_scale is None:
            raise ValueError("Single-camera alignment requires an explicit scale")
        scale = torch.as_tensor(
            single_view_scale,
            device=c2w_source.device,
            dtype=c2w_source.dtype,
        ).reshape(-1)
        if scale.shape != (c2w_source.shape[0],):
            raise ValueError("single_view_scale must contain one value per batch element")
        source_r = c2w_source[:, 0, :3, :3]
        destination_r = c2w_destination[:, 0, :3, :3]
        rotation = _project_so3(destination_r @ source_r.transpose(-1, -2))
        source_center = camera_centers(c2w_source[:, 0])
        destination_center = camera_centers(c2w_destination[:, 0])
        translation = destination_center - scale[:, None] * (
            rotation @ source_center[..., None]
        ).squeeze(-1)
        valid = (
            torch.isfinite(scale)
            & (scale > 0)
            & torch.isfinite(rotation).all(dim=(-2, -1))
            & torch.isfinite(translation).all(dim=-1)
        )
        zeros = torch.zeros_like(scale)
        diagnostics = AlignmentDiagnostics(
            rotation_error_deg=zeros,
            rotation_error_p90_deg=zeros,
            relative_baseline_error=zeros,
            relative_baseline_error_p90=zeros,
            center_error=zeros,
            center_error_p90=zeros,
            min_baseline=zeros,
            condition=[
                "single-view-scale-hint" if bool(value) else "invalid-single-view-scale-hint"
                for value in valid
            ],
            valid=valid,
        )
        return Sim3(scale, rotation, translation), diagnostics

    scales, rotations, translations = [], [], []
    rot_errors, rot_p90s, baseline_errors, baseline_p90s = [], [], [], []
    center_errors, center_p90s, min_baselines = [], [], []
    conditions: list[str] = []
    valid_values = []

    for source, destination in zip(c2w_source, c2w_destination):
        count = source.shape[0]

        source_r = source[:, :3, :3]         # [N, 3, 3]
        destination_r = destination[:, :3, :3]  # [N, 3, 3]
        candidates = destination_r @ source_r.transpose(-1, -2)  # [N, 3, 3]

        weights = torch.ones(count, dtype=source.dtype, device=source.device)
        rotation = _project_so3(candidates.mean(0))
        for _ in range(iterations):
            residual = _rotation_angle(candidates @ rotation.transpose(-1, -2))
            weights = _huber_weights(residual, math.radians(5.0))
            rotation = _project_so3((weights[:, None, None] * candidates).sum(0) / weights.sum())

        rotation_residual = _rotation_angle(candidates @ rotation.transpose(-1, -2))

        source_c = camera_centers(source)       # [N, 3]
        destination_c = camera_centers(destination)  # [N, 3]

        pairs = list(combinations(range(count), 2))
        delta_source = torch.stack([source_c[j] - source_c[i] for i, j in pairs])
        delta_destination = torch.stack([destination_c[j] - destination_c[i] for i, j in pairs])
        source_norm = delta_source.norm(dim=-1)

        usable = source_norm > min_baseline
        destination_norm = delta_destination.norm(dim=-1)
        direction_usable = usable & (destination_norm > min_baseline)

        if count >= 3 and direction_usable.sum() >= 2:

            source_unit = delta_source[direction_usable] / source_norm[direction_usable, None]
            destination_unit = delta_destination[direction_usable] / destination_norm[direction_usable, None]

            orientation_matrix = (weights[:, None, None] * candidates).sum(0) / weights.sum()
            for _ in range(iterations):
                predicted_unit = (rotation @ source_unit[..., None]).squeeze(-1)
                cosine = (predicted_unit * destination_unit).sum(-1).clamp(-1, 1)
                direction_weights = _huber_weights(torch.acos(cosine), math.radians(20.0))

                baseline_matrix = (
                    direction_weights[:, None, None]
                    * destination_unit[..., :, None]
                    * source_unit[..., None, :]
                ).sum(0) / direction_weights.sum()
                baseline_rotation = _project_so3(baseline_matrix)

                rotation = _project_so3(2.0 * orientation_matrix + baseline_rotation)
            rotation_residual = _rotation_angle(candidates @ rotation.transpose(-1, -2))

        if not usable.any():

            scale = torch.ones((), device=source.device, dtype=source.dtype)
            baseline_error = torch.full((), torch.inf, device=source.device, dtype=source.dtype)
            translation = (destination_c - (rotation @ source_c[..., None]).squeeze(-1)).mean(0)
            condition = "degenerate-baseline"
            is_valid = False
        else:

            source_lengths = source_norm[usable]
            destination_lengths = delta_destination[usable].norm(dim=-1)
            pair_weights = torch.ones(source_lengths.shape[0], device=source.device, dtype=source.dtype)

            scale = (destination_lengths / source_lengths.clamp_min(1e-8)).median().clamp_min(1e-8)
            for _ in range(iterations):
                errors = (scale * source_lengths - destination_lengths).abs()
                normalizer = destination_lengths.median().clamp_min(1e-8)
                pair_weights = _huber_weights(errors / normalizer, 0.05)
                scale = (source_lengths * destination_lengths * pair_weights).sum() / (
                    (source_lengths.square() * pair_weights).sum().clamp_min(1e-12)
                )
                scale = scale.clamp_min(1e-8)

            relative_errors = (scale * source_lengths - destination_lengths).abs() / destination_lengths.clamp_min(1e-8)
            baseline_error = relative_errors.median()

            translation_candidates = destination_c - scale * (rotation @ source_c[..., None]).squeeze(-1)
            translation = translation_candidates.median(dim=0).values

            for _ in range(iterations):
                trans_residual = (translation_candidates - translation).norm(dim=-1)
                trans_scale = trans_residual.median().clamp_min(1e-8)
                trans_weights = _huber_weights(trans_residual / trans_scale, 1.5)
                translation = (trans_weights[:, None] * translation_candidates).sum(0) / trans_weights.sum()

            condition = "low-redundancy" if count == 2 else "ok"
            is_valid = bool(
                math.degrees(float(rotation_residual.median())) <= max_rotation_error_deg
                and float(torch.quantile(torch.rad2deg(rotation_residual), 0.9)) <= max_rotation_error_deg
                and float(baseline_error) <= max_relative_baseline_error
            )
            if not is_valid:
                condition = "residual-failure"

        predicted_centers = scale * (rotation @ source_c[..., None]).squeeze(-1) + translation
        destination_baseline_scale = delta_destination.norm(dim=-1).median().clamp_min(1e-8)
        center_error = (predicted_centers - destination_c).norm(dim=-1).median() / destination_baseline_scale
        center_error_p90 = torch.quantile(
            (predicted_centers - destination_c).norm(dim=-1), 0.9
        ) / destination_baseline_scale

        if is_valid and float(center_error) > max_relative_center_error:
            is_valid = False
            condition = "center-residual-failure"
        elif is_valid and float(center_error) > max_relative_baseline_error:
            condition = "center-warning" if count > 2 else "low-redundancy-center-warning"

        scales.append(scale)
        rotations.append(rotation)
        translations.append(translation)
        rot_errors.append(torch.rad2deg(rotation_residual.median()))
        rot_p90s.append(torch.quantile(torch.rad2deg(rotation_residual), 0.9))
        baseline_errors.append(baseline_error)
        baseline_p90s.append(
            torch.quantile(relative_errors, 0.9) if usable.any() else torch.full_like(baseline_error, torch.inf)
        )
        center_errors.append(center_error)
        center_p90s.append(center_error_p90)
        min_baselines.append(source_norm.min())
        conditions.append(condition)
        valid_values.append(is_valid)

    sim3 = Sim3(torch.stack(scales), torch.stack(rotations), torch.stack(translations))
    diagnostics = AlignmentDiagnostics(
        rotation_error_deg=torch.stack(rot_errors),
        rotation_error_p90_deg=torch.stack(rot_p90s),
        relative_baseline_error=torch.stack(baseline_errors),
        relative_baseline_error_p90=torch.stack(baseline_p90s),
        center_error=torch.stack(center_errors),
        center_error_p90=torch.stack(center_p90s),
        min_baseline=torch.stack(min_baselines),
        condition=conditions,
        valid=torch.tensor(valid_values, device=c2w_source.device, dtype=torch.bool),
    )
    return sim3, diagnostics


def robust_scene_scale(
    z_depth: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor | None = None,
    trim_quantile: float = 0.02,
) -> torch.Tensor:
    """Robust scene scale."""
    if z_depth.ndim == 5 and z_depth.shape[-1] == 1:
        z_depth = z_depth[..., 0]
    if confidence.shape != z_depth.shape:
        raise ValueError("confidence and z_depth must have identical shape")
    if valid is None:
        valid = torch.isfinite(z_depth) & (z_depth > 0) & torch.isfinite(confidence) & (confidence > 0)

    batch = z_depth.shape[0]
    output = []
    for index in range(batch):

        values = z_depth[index][valid[index]].float()
        weights = confidence[index][valid[index]].float().clamp_min(0)

        if values.numel() == 0:
            raise ValueError("Cannot compute scene scale without valid depth")

        order = torch.argsort(values)
        values, weights = values[order], weights[order]

        cumulative = torch.cumsum(weights, dim=0)
        total = cumulative[-1].clamp_min(1e-8)
        keep = (cumulative >= trim_quantile * total) & (cumulative <= (1 - trim_quantile) * total)
        if keep.any():
            values, weights = values[keep], weights[keep]
            cumulative = torch.cumsum(weights, dim=0)
            total = cumulative[-1].clamp_min(1e-8)

        median_index = torch.searchsorted(cumulative, 0.5 * total).clamp_max(values.numel() - 1)
        output.append(values[median_index])

    return torch.stack(output).to(z_depth.dtype)
