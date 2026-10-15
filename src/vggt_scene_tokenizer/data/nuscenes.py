"""nuScenes calibration and sparse LiDAR depth utilities."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch


def quaternion_matrix(value: list[float]) -> np.ndarray:
    """Convert a nuScenes scalar-first quaternion to a rotation matrix."""
    w, x, y, z = (float(item) for item in value)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0:
        raise ValueError("zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform(translation: list[float], rotation: list[float]) -> np.ndarray:
    """Build a homogeneous sensor-to-parent transform."""
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_matrix(rotation)
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


def global_from_sensor(
    record: dict,
    calibrations: dict[str, dict],
    ego_poses: dict[str, dict],
) -> np.ndarray:
    """Compose calibrated-sensor and ego transforms into the global frame."""
    calibration = calibrations[record["calibrated_sensor_token"]]
    ego_pose = ego_poses[record["ego_pose_token"]]
    return transform(ego_pose["translation"], ego_pose["rotation"]) @ transform(
        calibration["translation"], calibration["rotation"]
    )


def project_lidar_depth(
    dataroot: Path,
    lidar_record: dict,
    camera_record: dict,
    calibrations: dict[str, dict],
    ego_poses: dict[str, dict],
    pixel_transform: torch.Tensor,
    target_hw: tuple[int, int],
    minimum_depth_m: float = 0.5,
) -> torch.Tensor:
    """Project sparse z-depth; the caller selects the near-depth cutoff."""
    points = np.fromfile(dataroot / lidar_record["filename"], dtype=np.float32)
    if points.size % 5:
        raise ValueError(f"invalid nuScenes LiDAR file {lidar_record['filename']}")
    xyz = points.reshape(-1, 5)[:, :3].astype(np.float64)
    homogeneous = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1)
    global_from_lidar = global_from_sensor(lidar_record, calibrations, ego_poses)
    global_from_camera = global_from_sensor(camera_record, calibrations, ego_poses)
    camera_points = (
        np.linalg.inv(global_from_camera) @ global_from_lidar @ homogeneous.T
    ).T[:, :3]
    calibration = calibrations[camera_record["calibrated_sensor_token"]]
    intrinsics = pixel_transform.double().numpy() @ np.asarray(
        calibration["camera_intrinsic"], dtype=np.float64
    )
    z_depth = camera_points[:, 2]
    projected = (intrinsics @ camera_points.T).T
    u = np.rint(projected[:, 0] / np.maximum(z_depth, 1e-12)).astype(np.int64)
    v = np.rint(projected[:, 1] / np.maximum(z_depth, 1e-12)).astype(np.int64)
    height, width = target_hw
    positive_depth = (
        z_depth > 0.0 if minimum_depth_m == 0.0 else z_depth >= minimum_depth_m
    )
    valid = (
        np.isfinite(z_depth)
        & positive_depth
        & (z_depth <= 80.0)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    linear = v[valid] * width + u[valid]
    depth = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(depth, linear, z_depth[valid].astype(np.float32))
    depth[~np.isfinite(depth)] = -1.0
    return torch.from_numpy(depth.reshape(height, width))
