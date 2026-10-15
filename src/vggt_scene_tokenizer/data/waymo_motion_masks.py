#!/usr/bin/env python3
"""Prepare Waymo motion masks for pixel-region validation statistics.

Project tracked 3D cuboids into the undistorted camera grid and rasterize their
visible silhouettes, with nearer cuboids painted last. Binary PNGs use 255 for
objects moving faster than 0.5 m/s and 0 for static/background pixels.
Temporal training-pool labels are constructed separately by waymo_pools.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vggt_scene_tokenizer.data.waymo import (
    CAMERA_NAMES,
    CV_TO_WAYMO_CAMERA,
    _camera_calibrations,
    _matrix,
    _pyarrow_parquet,
    _target_geometry,
)
from vggt_scene_tokenizer.data.waymo_pools import BOX_PREFIX, _box_corners


CAMERA_LABELS = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT"}
MOVABLE_TYPES = frozenset((1, 2, 4))  # vehicle, pedestrian, cyclist
BOX_EDGES = (
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
)


@dataclass(frozen=True)
class MaskConfig:
    resolution: int = 512
    patch_size: int = 16
    moving_threshold_mps: float = 0.5
    minimum_lidar_points: int = 1
    near_clip_m: float = 0.1


def _clipped_projection_polygon(
    corners_camera: np.ndarray,
    intrinsic: np.ndarray,
    image_hw: tuple[int, int],
    near: float,
) -> tuple[np.ndarray, float] | None:
    """Return projected convex silhouette and representative camera depth."""
    points: list[np.ndarray] = [p for p in corners_camera if p[2] >= near]
    for i, j in BOX_EDGES:
        a, b = corners_camera[i], corners_camera[j]
        a_front, b_front = a[2] >= near, b[2] >= near
        if a_front == b_front:
            continue
        alpha = (near - a[2]) / (b[2] - a[2])
        points.append(a + alpha * (b - a))
    if len(points) < 3:
        return None
    xyz = np.asarray(points, dtype=np.float64)
    projected = (intrinsic @ xyz.T).T
    uv = projected[:, :2] / projected[:, 2:3]
    if not np.isfinite(uv).all():
        return None
    height, width = image_hw
    # Near-plane intersections can project extremely far away. Clipping to a
    # generous finite canvas preserves an in-frame silhouette and avoids int32
    # overflow in OpenCV.
    uv[:, 0] = np.clip(uv[:, 0], -4 * width, 5 * width)
    uv[:, 1] = np.clip(uv[:, 1], -4 * height, 5 * height)
    hull = cv2.convexHull(np.rint(uv).astype(np.int32)).reshape(-1, 2)
    if len(hull) < 3:
        return None
    left, top = hull.min(axis=0)
    right, bottom = hull.max(axis=0)
    if right < 0 or bottom < 0 or left >= width or top >= height:
        return None
    return hull, float(np.median(xyz[:, 2]))


def _read_segment_rows(split_root: Path, segment_name: str):
    parquet = _pyarrow_parquet()
    camera_columns = [
        "key.frame_timestamp_micros",
        "key.camera_name",
        "[CameraImageComponent].pose.transform",
    ]
    camera_rows = parquet.read_table(
        split_root / "camera_image" / segment_name,
        columns=camera_columns,
        filters=[("key.camera_name", "in", list(CAMERA_NAMES))],
    ).to_pylist()
    cameras: dict[int, dict[int, np.ndarray]] = {}
    for row in camera_rows:
        timestamp = int(row["key.frame_timestamp_micros"])
        cameras.setdefault(timestamp, {})[int(row["key.camera_name"])] = _matrix(
            row["[CameraImageComponent].pose.transform"]
        )

    pose_rows = parquet.read_table(
        split_root / "vehicle_pose" / segment_name,
        columns=[
            "key.frame_timestamp_micros",
            "[VehiclePoseComponent].world_from_vehicle.transform",
        ],
    ).to_pylist()
    poses = {
        int(row["key.frame_timestamp_micros"]): _matrix(
            row["[VehiclePoseComponent].world_from_vehicle.transform"]
        )
        for row in pose_rows
    }

    box_columns = [
        "key.frame_timestamp_micros",
        "key.laser_object_id",
        *(BOX_PREFIX + f"box.center.{axis}" for axis in "xyz"),
        *(BOX_PREFIX + f"box.size.{axis}" for axis in "xyz"),
        BOX_PREFIX + "box.heading",
        BOX_PREFIX + "type",
        BOX_PREFIX + "num_lidar_points_in_box",
        BOX_PREFIX + "speed.x",
        BOX_PREFIX + "speed.y",
    ]
    box_rows = parquet.read_table(
        split_root / "lidar_box" / segment_name, columns=box_columns
    ).to_pylist()
    boxes: dict[int, list[dict[str, Any]]] = {}
    for row in box_rows:
        boxes.setdefault(int(row["key.frame_timestamp_micros"]), []).append(row)
    return cameras, poses, boxes


def _write_png_atomic(path: Path, mask: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise RuntimeError(f"PNG encoding failed: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(encoded.tobytes())
    os.replace(temporary, path)


def _segment_is_complete(success_path: Path, config: MaskConfig) -> bool:
    if not success_path.exists():
        return False
    try:
        previous = json.loads(success_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return previous.get("config") == asdict(config) and int(previous.get("png_count", -1)) > 0


def generate_segment(task: tuple[str, str, str, str, MaskConfig]) -> dict[str, Any]:
    cv2.setNumThreads(0)
    split, segment_name, waymo_root_value, output_root_value, config = task
    started = time.time()
    split_root = Path(waymo_root_value) / split
    segment_stem = Path(segment_name).stem
    segment_output = Path(output_root_value) / split / segment_stem
    success_path = segment_output / "_SUCCESS.json"
    if _segment_is_complete(success_path, config):
        result = json.loads(success_path.read_text())
        result["skipped"] = True
        return result

    calibrations = _camera_calibrations(split_root / "camera_calibration" / segment_name)
    image_hw, intrinsics, _ = _target_geometry(
        calibrations, config.resolution, config.patch_size
    )
    cameras, poses, boxes = _read_segment_rows(split_root, segment_name)
    timestamps = sorted(
        t for t, by_camera in cameras.items()
        if set(by_camera) == set(CAMERA_NAMES) and t in poses
    )
    if not timestamps:
        raise ValueError(f"No complete front-three timestamps: {split}/{segment_name}")

    for name in CAMERA_NAMES:
        (segment_output / CAMERA_LABELS[name]).mkdir(parents=True, exist_ok=True)

    dynamic_box_count = 0
    projected_dynamic_count = 0
    dynamic_pixels = 0
    png_count = 0
    for timestamp in timestamps:
        frame_boxes = []
        for row in boxes.get(timestamp, []):
            object_type = int(row[BOX_PREFIX + "type"])
            lidar_points = int(row[BOX_PREFIX + "num_lidar_points_in_box"] or 0)
            if object_type not in MOVABLE_TYPES or lidar_points < config.minimum_lidar_points:
                continue
            vx = float(row[BOX_PREFIX + "speed.x"] or 0.0)
            vy = float(row[BOX_PREFIX + "speed.y"] or 0.0)
            speed = math.hypot(vx, vy)
            dynamic = math.isfinite(speed) and speed > config.moving_threshold_mps
            dynamic_box_count += int(dynamic)
            frame_boxes.append((_box_corners(row), dynamic))

        world_from_vehicle_frame = poses[timestamp]
        for camera_name in CAMERA_NAMES:
            calibration = calibrations[camera_name]
            world_from_camera_cv = (
                cameras[timestamp][camera_name]
                @ np.asarray(calibration["vehicle_from_camera"], dtype=np.float64)
                @ CV_TO_WAYMO_CAMERA
            )
            camera_from_vehicle_frame = (
                np.linalg.inv(world_from_camera_cv) @ world_from_vehicle_frame
            )
            renderables = []
            for corners_vehicle, dynamic in frame_boxes:
                homogeneous = np.concatenate(
                    (corners_vehicle, np.ones((8, 1), dtype=np.float64)), axis=1
                )
                corners_camera = (camera_from_vehicle_frame @ homogeneous.T).T[:, :3]
                projected = _clipped_projection_polygon(
                    corners_camera,
                    intrinsics[camera_name].astype(np.float64),
                    image_hw,
                    config.near_clip_m,
                )
                if projected is not None:
                    hull, depth = projected
                    renderables.append((depth, hull, dynamic))
                    projected_dynamic_count += int(dynamic)

            # Painter's algorithm: farther boxes first, nearer boxes last.
            mask = np.zeros(image_hw, dtype=np.uint8)
            for _, hull, dynamic in sorted(renderables, key=lambda x: x[0], reverse=True):
                cv2.fillConvexPoly(mask, hull, 255 if dynamic else 0, lineType=cv2.LINE_8)
            dynamic_pixels += int(np.count_nonzero(mask))
            output_path = (
                segment_output / CAMERA_LABELS[camera_name] / f"{timestamp}.png"
            )
            _write_png_atomic(output_path, mask)
            png_count += 1

    result = {
        "format": "waymo_fr3d_motion_mask_v1",
        "split": split,
        "segment": segment_stem,
        "timestamps": len(timestamps),
        "cameras": [CAMERA_LABELS[name] for name in CAMERA_NAMES],
        "image_hw": list(image_hw),
        "png_count": png_count,
        "dynamic_box_observations": dynamic_box_count,
        "projected_dynamic_box_observations": projected_dynamic_count,
        "dynamic_pixels": dynamic_pixels,
        "dynamic_pixel_fraction": dynamic_pixels / float(png_count * image_hw[0] * image_hw[1]),
        "config": asdict(config),
        "seconds": time.time() - started,
        "output": str(segment_output),
        "skipped": False,
    }
    temporary = success_path.with_name(f".{success_path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, success_path)
    return result


def _segment_names(root: Path, split: str) -> list[str]:
    required = ("camera_image", "camera_calibration", "vehicle_pose", "lidar_box")
    common: set[str] | None = None
    for component in required:
        names = {p.name for p in (root / split / component).glob("*.parquet")}
        common = names if common is None else common & names
    return sorted(common or ())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waymo-root", default="data/waymo")
    parser.add_argument("--output-root", default="data/waymo_motion_masks")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        choices=("train", "validation"),
        help="Splits with annotated 3D tracks.",
    )
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--moving-threshold-mps", type=float, default=0.5)
    parser.add_argument("--minimum-lidar-points", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--max-segments", type=int)
    args = parser.parse_args()

    waymo_root = Path(args.waymo_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = MaskConfig(
        resolution=args.resolution,
        patch_size=args.patch_size,
        moving_threshold_mps=args.moving_threshold_mps,
        minimum_lidar_points=args.minimum_lidar_points,
    )
    tasks = []
    expected = {}
    for split in args.splits:
        names = _segment_names(waymo_root, split)
        expected[split] = len(names)
        tasks.extend((split, name, str(waymo_root), str(output_root), config) for name in names)
    if args.max_segments is not None:
        tasks = tasks[: args.max_segments]
    if not tasks:
        raise FileNotFoundError("No Waymo segments with all required components")

    run = {
        "format": "waymo_fr3d_motion_mask_run_v1",
        "waymo_root": str(waymo_root.resolve()),
        "output_root": str(output_root.resolve()),
        "splits": args.splits,
        "expected_segments": expected,
        "selected_segments": len(tasks),
        "config": asdict(config),
        "started_unix": time.time(),
    }
    (output_root / "run_config.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")

    completed = 0
    failed = []
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(generate_segment, task): task[:2] for task in tasks}
        for future in as_completed(futures):
            split, name = futures[future]
            try:
                row = future.result()
                rows.append(row)
                completed += 1
                print(
                    f"[{completed}/{len(tasks)}] {split}/{Path(name).stem} "
                    f"png={row['png_count']} dynamic={row['dynamic_pixel_fraction']:.5f} "
                    f"skipped={row['skipped']}",
                    flush=True,
                )
            except Exception as error:  # keep the full batch running
                failed.append({"split": split, "segment": Path(name).stem, "error": repr(error)})
                print(f"ERROR {split}/{name}: {error!r}", flush=True)

    rows.sort(key=lambda x: (x["split"], x["segment"]))
    manifest_path = output_root / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary = {
        **run,
        "finished_unix": time.time(),
        "completed_segments": len(rows),
        "failed_segments": failed,
        "png_count": sum(int(row["png_count"]) for row in rows),
        "dynamic_pixels": sum(int(row["dynamic_pixels"]) for row in rows),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
