"""KITTI calibration and OXTS pose utilities used by evaluation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _read_calibration(path: Path) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, payload = line.split(":", 1)
        if payload := payload.strip():
            try:
                values[key] = np.fromstring(payload, sep=" ", dtype=np.float64)
            except ValueError:
                continue
    return values


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.reshape(3, 3)
    result[:3, 3] = translation.reshape(3)
    return result


def _rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def _oxts_pose(packet: np.ndarray, scale: float) -> np.ndarray:
    latitude, longitude, altitude, roll, pitch, yaw = packet[:6]
    earth_radius = 6_378_137.0
    translation = np.array(
        [
            scale * longitude * math.pi * earth_radius / 180.0,
            scale * earth_radius * math.log(math.tan((90.0 + latitude) * math.pi / 360.0)),
            altitude,
        ],
        dtype=np.float64,
    )
    return _transform(_rotation(roll, pitch, yaw), translation)
