from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from GeoState.data.waymo import CAMERA_NAMES, CV_TO_WAYMO_CAMERA, _camera_calibrations, _pyarrow_parquet


BOX_PREFIX = "[LiDARBoxComponent]."
TYPE_NAMES = {1: "vehicle", 2: "pedestrian", 4: "cyclist"}


@dataclass(frozen=True)
class PoolConfig:
    state_count: int = 11
    frame_stride: int = 2
    minimum_lidar_points: int = 3
    minimum_track_states: int = 2
    maximum_step_relative_error: float = 0.10
    vehicle_speed_threshold: float = 0.75
    pedestrian_speed_threshold: float = 0.25
    cyclist_speed_threshold: float = 0.50
    vehicle_speed_reference: float = 5.0
    pedestrian_speed_reference: float = 1.5
    cyclist_speed_reference: float = 3.0

    def speed_threshold(self, object_type: int) -> float:
        return {
            1: self.vehicle_speed_threshold,
            2: self.pedestrian_speed_threshold,
            4: self.cyclist_speed_threshold,
        }[object_type]

    def speed_reference(self, object_type: int) -> float:
        return {
            1: self.vehicle_speed_reference,
            2: self.pedestrian_speed_reference,
            4: self.cyclist_speed_reference,
        }[object_type]


@dataclass(frozen=True)
class BoxObservation:
    object_id: str
    object_type: int
    speed_mps: float
    visible_area_fraction: float


def _box_corners(row: dict[str, Any]) -> np.ndarray:
    center = np.array(
        [
            row[BOX_PREFIX + "box.center.x"],
            row[BOX_PREFIX + "box.center.y"],
            row[BOX_PREFIX + "box.center.z"],
        ],
        dtype=np.float64,
    )
    size = np.array(
        [
            row[BOX_PREFIX + "box.size.x"],
            row[BOX_PREFIX + "box.size.y"],
            row[BOX_PREFIX + "box.size.z"],
        ],
        dtype=np.float64,
    )
    signs = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    local = signs * (size * 0.5)
    heading = float(row[BOX_PREFIX + "box.heading"])
    cosine, sine = math.cos(heading), math.sin(heading)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return local @ rotation.T + center


def _camera_projection_data(
    calibration_path: Path,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    calibrations = _camera_calibrations(calibration_path)
    result: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    for name in CAMERA_NAMES:
        calibration = calibrations[name]
        vehicle_from_camera_cv = (
            np.asarray(calibration["vehicle_from_camera"], dtype=np.float64)
            @ CV_TO_WAYMO_CAMERA
        )
        result.append(
            (
                np.linalg.inv(vehicle_from_camera_cv),
                np.asarray(calibration["intrinsic"], dtype=np.float64),
                int(calibration["width"]),
                int(calibration["height"]),
            )
        )
    return result


def _projected_area_fraction(
    corners_vehicle: np.ndarray,
    cameras: Iterable[tuple[np.ndarray, np.ndarray, int, int]],
) -> float:
    homogeneous = np.concatenate(
        [corners_vehicle, np.ones((corners_vehicle.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    maximum = 0.0
    for camera_from_vehicle, intrinsic, width, height in cameras:
        points = (camera_from_vehicle @ homogeneous.T).T[:, :3]
        in_front = points[:, 2] > 0.1
        if not in_front.any():
            continue
        points = points[in_front]
        projected = (intrinsic @ points.T).T
        u = projected[:, 0] / projected[:, 2]
        v = projected[:, 1] / projected[:, 2]
        left = float(np.clip(u.min(), 0.0, float(width)))
        right = float(np.clip(u.max(), 0.0, float(width)))
        top = float(np.clip(v.min(), 0.0, float(height)))
        bottom = float(np.clip(v.max(), 0.0, float(height)))
        area = max(0.0, right - left) * max(0.0, bottom - top)
        maximum = max(maximum, area / float(width * height))
    return min(maximum, 1.0)


def _load_timestamps(vehicle_pose_path: Path, camera_image_path: Path) -> list[int]:
    parquet = _pyarrow_parquet()
    pose_table = parquet.read_table(
        vehicle_pose_path, columns=["key.frame_timestamp_micros"]
    )
    pose_timestamps = {int(value) for value in pose_table.column(0).to_pylist()}
    camera_table = parquet.read_table(
        camera_image_path,
        columns=["key.frame_timestamp_micros", "key.camera_name"],
        filters=[("key.camera_name", "in", list(CAMERA_NAMES))],
    )
    cameras_by_timestamp: dict[int, set[int]] = {}
    for row in camera_table.to_pylist():
        cameras_by_timestamp.setdefault(int(row["key.frame_timestamp_micros"]), set()).add(
            int(row["key.camera_name"])
        )
    return sorted(
        timestamp
        for timestamp, cameras in cameras_by_timestamp.items()
        if cameras == set(CAMERA_NAMES) and timestamp in pose_timestamps
    )


def _has_regular_state_steps(logical_timestamps: list[int], config: PoolConfig) -> bool:
    nominal_seconds = config.frame_stride / 10.0
    minimum = nominal_seconds * (1.0 - config.maximum_step_relative_error)
    maximum = nominal_seconds * (1.0 + config.maximum_step_relative_error)
    deltas = np.diff(np.asarray(logical_timestamps, dtype=np.int64)) / 1e6
    return bool(((deltas >= minimum) & (deltas <= maximum)).all())


def _load_observations(
    box_path: Path,
    calibration_path: Path,
    config: PoolConfig,
) -> dict[int, list[BoxObservation]]:
    parquet = _pyarrow_parquet()
    columns = [
        "key.frame_timestamp_micros",
        "key.laser_object_id",
        BOX_PREFIX + "box.center.x",
        BOX_PREFIX + "box.center.y",
        BOX_PREFIX + "box.center.z",
        BOX_PREFIX + "box.size.x",
        BOX_PREFIX + "box.size.y",
        BOX_PREFIX + "box.size.z",
        BOX_PREFIX + "box.heading",
        BOX_PREFIX + "type",
        BOX_PREFIX + "num_lidar_points_in_box",
        BOX_PREFIX + "speed.x",
        BOX_PREFIX + "speed.y",
        BOX_PREFIX + "speed.z",
    ]
    rows = parquet.read_table(box_path, columns=columns).to_pylist()
    cameras = _camera_projection_data(calibration_path)
    result: dict[int, list[BoxObservation]] = {}
    for row in rows:
        object_type = int(row[BOX_PREFIX + "type"])
        if object_type not in TYPE_NAMES:
            continue
        if int(row[BOX_PREFIX + "num_lidar_points_in_box"] or 0) < config.minimum_lidar_points:
            continue
        speed = math.sqrt(
            sum(
                float(row[BOX_PREFIX + f"speed.{axis}"] or 0.0) ** 2
                for axis in ("x", "y", "z")
            )
        )
        if not math.isfinite(speed):
            continue
        area = _projected_area_fraction(_box_corners(row), cameras)
        if area <= 0.0:
            continue
        timestamp = int(row["key.frame_timestamp_micros"])
        result.setdefault(timestamp, []).append(
            BoxObservation(
                object_id=str(row["key.laser_object_id"]),
                object_type=object_type,
                speed_mps=speed,
                visible_area_fraction=area,
            )
        )
    return result


def _score_window(
    observations_by_timestamp: dict[int, list[BoxObservation]],
    logical_timestamps: list[int],
    config: PoolConfig,
) -> dict[str, Any]:
    tracks: dict[str, list[BoxObservation]] = {}
    for timestamp in logical_timestamps:
        for observation in observations_by_timestamp.get(timestamp, ()):
            tracks.setdefault(observation.object_id, []).append(observation)

    dynamic_score = 0.0
    visible_moving_area_fraction = 0.0
    moving_track_count = 0
    max_track_speed_mps = 0.0
    moving_type_counts = {name: 0 for name in TYPE_NAMES.values()}
    for observations in tracks.values():
        if len(observations) < config.minimum_track_states:
            continue
        object_type = observations[0].object_type
        speeds = np.asarray([value.speed_mps for value in observations], dtype=np.float64)
        speed = float(np.median(speeds))
        threshold = config.speed_threshold(object_type)
        if speed < threshold:
            continue
        area = max(value.visible_area_fraction for value in observations)
        motion_strength = min(speed / config.speed_reference(object_type), 2.0)
        dynamic_score += math.sqrt(area) * motion_strength
        visible_moving_area_fraction += area
        moving_track_count += 1
        max_track_speed_mps = max(max_track_speed_mps, speed)
        moving_type_counts[TYPE_NAMES[object_type]] += 1

    return {
        "dynamic_score": dynamic_score,
        "moving_track_count": moving_track_count,
        "visible_moving_area_fraction": min(visible_moving_area_fraction, 1.0),
        "max_track_speed_mps": max_track_speed_mps,
        "moving_type_counts": moving_type_counts,
    }


def _score_segment(task: tuple[str, str, str, str, PoolConfig]) -> list[dict[str, Any]]:
    split, segment_name, waymo_root_value, prepared_root_value, config = task
    waymo_root = Path(waymo_root_value)
    prepared_root = Path(prepared_root_value)
    stem = Path(segment_name).stem
    if not (prepared_root / split / f"{stem}.pt").is_file():
        return []
    split_root = waymo_root / split
    timestamps = _load_timestamps(
        split_root / "vehicle_pose" / segment_name,
        split_root / "camera_image" / segment_name,
    )
    required_span = (config.state_count - 1) * config.frame_stride
    if len(timestamps) <= required_span:
        return []
    observations = _load_observations(
        split_root / "lidar_box" / segment_name,
        split_root / "camera_calibration" / segment_name,
        config,
    )
    records: list[dict[str, Any]] = []
    for first_slot in range(len(timestamps) - required_span):
        logical_slots = [
            first_slot + state_index * config.frame_stride
            for state_index in range(config.state_count)
        ]
        logical_timestamps = [timestamps[slot] for slot in logical_slots]
        if not _has_regular_state_steps(logical_timestamps, config):
            continue
        score = _score_window(observations, logical_timestamps, config)
        records.append(
            {
                "window_id": f"{stem}:{first_slot}:s{config.frame_stride}:n{config.state_count}",
                "split": split,
                "segment": stem,
                "first_slot": first_slot,
                "logical_slots": logical_slots,
                "logical_timestamps": logical_timestamps,
                "frame_stride": config.frame_stride,
                "state_count": config.state_count,
                "duration_seconds": (logical_timestamps[-1] - logical_timestamps[0]) / 1e6,
                **score,
            }
        )
    return records


def _segment_names(waymo_root: Path, prepared_root: Path, split: str) -> list[str]:
    components = ("lidar_box", "vehicle_pose", "camera_calibration", "camera_image")
    common: set[str] | None = None
    for component in components:
        names = {path.name for path in (waymo_root / split / component).glob("*.parquet")}
        common = names if common is None else common & names
    prepared = {f"{path.stem}.parquet" for path in (prepared_root / split).glob("*.pt")}
    return sorted((common or set()) & prepared)


def score_split(
    waymo_root: Path,
    prepared_root: Path,
    split: str,
    config: PoolConfig,
    workers: int,
) -> list[dict[str, Any]]:
    names = _segment_names(waymo_root, prepared_root, split)
    tasks = [
        (split, name, str(waymo_root), str(prepared_root), config)
        for name in names
    ]
    records: list[dict[str, Any]] = []
    if workers <= 1:
        for index, task in enumerate(tasks, 1):
            records.extend(_score_segment(task))
            if index % 25 == 0 or index == len(tasks):
                print(json.dumps({"split": split, "segments": index, "windows": len(records)}))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_score_segment, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), 1):
                records.extend(future.result())
                if index % 25 == 0 or index == len(futures):
                    print(json.dumps({"split": split, "segments": index, "windows": len(records)}))
    records.sort(key=lambda item: (item["segment"], item["first_slot"]))
    return records


def assign_pools(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the paper's exhaustive binary motion partition."""
    static: list[dict[str, Any]] = []
    dynamic: list[dict[str, Any]] = []
    for source in records:
        record = dict(source)
        if int(record["moving_track_count"]) == 0:
            record["pool"] = "static_temporal"
            static.append(record)
        else:
            record["pool"] = "dynamic_temporal"
            dynamic.append(record)
    return static, dynamic


def _atomic_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _summary(
    records: list[dict[str, Any]],
    static: list[dict[str, Any]],
    dynamic: list[dict[str, Any]],
) -> dict[str, Any]:
    scores = np.asarray([record["dynamic_score"] for record in records], dtype=np.float64)
    return {
        "windows": len(records),
        "segments": len({record["segment"] for record in records}),
        "pool_counts": {
            "static_temporal": len(static),
            "dynamic_temporal": len(dynamic),
        },
        "dynamic_score_quantiles": {
            str(quantile): float(np.quantile(scores, quantile))
            for quantile in (0.0, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99, 1.0)
        },
        "moving_track_windows": sum(int(record["moving_track_count"]) > 0 for record in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build static-temporal and dynamic-temporal Waymo window pools."
    )
    parser.add_argument("--waymo-root", default="data/waymo")
    parser.add_argument(
        "--prepared-root", default="data/waymo_prepared"
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--state-count", type=int, default=11)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--minimum-lidar-points", type=int, default=3)
    parser.add_argument("--minimum-track-states", type=int, default=2)
    parser.add_argument("--maximum-step-relative-error", type=float, default=0.10)
    args = parser.parse_args()

    config = PoolConfig(
        state_count=args.state_count,
        frame_stride=args.frame_stride,
        minimum_lidar_points=args.minimum_lidar_points,
        minimum_track_states=args.minimum_track_states,
        maximum_step_relative_error=args.maximum_step_relative_error,
    )
    if config.state_count < 2 or config.frame_stride < 1:
        raise ValueError("state-count must be >=2 and frame-stride must be >=1")

    waymo_root = Path(args.waymo_root)
    prepared_root = Path(args.prepared_root)
    output_root = Path(args.output_root)
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    if "train" not in splits:
        raise ValueError("The train split is required to derive fixed pool thresholds")

    scored = {
        split: score_split(
            waymo_root, prepared_root, split, config, max(1, int(args.workers))
        )
        for split in splits
    }
    summaries: dict[str, Any] = {}
    for split, records in scored.items():
        static, dynamic = assign_pools(records)
        exhaustive = sorted(
            [*static, *dynamic], key=lambda item: (item["segment"], item["first_slot"])
        )
        _atomic_jsonl(exhaustive, output_root / f"{split}_all_temporal.jsonl")
        _atomic_jsonl(static, output_root / f"{split}_static_temporal.jsonl")
        _atomic_jsonl(dynamic, output_root / f"{split}_dynamic_temporal.jsonl")
        summaries[split] = _summary(records, static, dynamic)
    metadata = {
        "format": "data.waymo_pools_v1",
        "waymo_root": str(waymo_root.resolve()),
        "prepared_root": str(prepared_root.resolve()),
        "config": asdict(config),
        "splits": summaries,
        "scoring": {
            "objects": TYPE_NAMES,
            "visibility": "maximum clipped projected 3D-box area over FRONT/FRONT_LEFT/FRONT_RIGHT",
            "score": "sum(sqrt(visible_area_fraction) * clipped_type_normalized_speed)",
            "static_rule": "zero visible moving tracks",
            "dynamic_rule": "one or more visible moving tracks",
        },
    }
    _atomic_json(metadata, output_root / "metadata.json")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
