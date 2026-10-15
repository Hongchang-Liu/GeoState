from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import cv2
except ModuleNotFoundError:  # optional for consumers of constants/prepared tensors
    cv2 = None
import numpy as np
import torch
from PIL import Image

from GeoState.data.common import _target_shape


CAMERA_NAMES = (1, 2, 3)  # FRONT, FRONT_LEFT, FRONT_RIGHT.
CV_TO_WAYMO_CAMERA = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _pyarrow_parquet():
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ImportError(
            "Waymo parquet support requires pyarrow. Add its install directory "
            "to PYTHONPATH (or install the `waymo` dependency extra)."
        ) from error
    return parquet


def discover_segment_names(split_root: str | Path) -> list[str]:
    root = Path(split_root)
    components = (
        "camera_image",
        "camera_calibration",
        "vehicle_pose",
        "lidar",
        "lidar_calibration",
    )
    common: set[str] | None = None
    for component in components:
        names = {path.name for path in (root / component).glob("*.parquet")}
        common = names if common is None else common & names
    return sorted(common or ())


def _matrix(values: object) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def _camera_calibrations(path: Path) -> dict[int, dict[str, object]]:
    parquet = _pyarrow_parquet()
    rows = parquet.read_table(path).to_pylist()
    calibrations: dict[int, dict[str, object]] = {}
    prefix = "[CameraCalibrationComponent]."
    for row in rows:
        name = int(row["key.camera_name"])
        if name not in CAMERA_NAMES:
            continue
        intrinsic = np.array(
            [
                [row[prefix + "intrinsic.f_u"], 0.0, row[prefix + "intrinsic.c_u"]],
                [0.0, row[prefix + "intrinsic.f_v"], row[prefix + "intrinsic.c_v"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.array(
            [
                row[prefix + "intrinsic.k1"],
                row[prefix + "intrinsic.k2"],
                row[prefix + "intrinsic.p1"],
                row[prefix + "intrinsic.p2"],
                row[prefix + "intrinsic.k3"],
            ],
            dtype=np.float64,
        )
        calibrations[name] = {
            "intrinsic": intrinsic,
            "distortion": distortion,
            "vehicle_from_camera": _matrix(row[prefix + "extrinsic.transform"]),
            "width": int(row[prefix + "width"]),
            "height": int(row[prefix + "height"]),
        }
    if set(calibrations) != set(CAMERA_NAMES):
        raise ValueError(f"Missing one of Waymo cameras {CAMERA_NAMES} in {path}")
    return calibrations


def _lidar_calibrations(path: Path) -> dict[int, dict[str, object]]:
    parquet = _pyarrow_parquet()
    rows = parquet.read_table(path).to_pylist()
    prefix = "[LiDARCalibrationComponent]."
    result: dict[int, dict[str, object]] = {}
    for row in rows:
        result[int(row["key.laser_name"])] = {
            "vehicle_from_laser": _matrix(row[prefix + "extrinsic.transform"]),
            "inclination_values": np.asarray(
                row[prefix + "beam_inclination.values"] or [], dtype=np.float64
            ),
            "inclination_min": float(row[prefix + "beam_inclination.min"]),
            "inclination_max": float(row[prefix + "beam_inclination.max"]),
        }
    return result


def _inclinations(calibration: dict[str, object], height: int) -> np.ndarray:
    values = np.asarray(calibration["inclination_values"], dtype=np.float64)
    if values.size:
        if values.size != height:
            raise ValueError("LiDAR beam inclination count disagrees with range image")
        return values[::-1].copy()
    minimum = float(calibration["inclination_min"])
    maximum = float(calibration["inclination_max"])
    # This is Waymo range_image_utils.compute_inclination followed by reverse.
    values = minimum + (np.arange(height, dtype=np.float64) + 0.5) / height * (
        maximum - minimum
    )
    return values[::-1].copy()


def _range_image_points(
    values: object,
    shape: object,
    calibration: dict[str, object],
) -> np.ndarray:
    dims = tuple(int(value) for value in shape)
    if len(dims) != 3 or dims[-1] < 1:
        raise ValueError(f"Unexpected Waymo range image shape {dims}")
    tensor = np.asarray(values, dtype=np.float32).reshape(dims)
    ranges = tensor[..., 0]
    valid = np.isfinite(ranges) & (ranges > 0.0)
    if not valid.any():
        return np.empty((0, 3), dtype=np.float32)
    height, width = dims[:2]
    inclinations = _inclinations(calibration, height)
    correction = math.atan2(
        float(calibration["vehicle_from_laser"][1, 0]),
        float(calibration["vehicle_from_laser"][0, 0]),
    )
    ratios = (np.arange(width, 0, -1, dtype=np.float64) - 0.5) / width
    azimuths = (ratios * 2.0 - 1.0) * math.pi - correction
    rows, columns = np.nonzero(valid)
    distance = ranges[rows, columns].astype(np.float64)
    inclination = inclinations[rows]
    azimuth = azimuths[columns]
    cos_inclination = np.cos(inclination)
    points_laser = np.stack(
        (
            np.cos(azimuth) * cos_inclination * distance,
            np.sin(azimuth) * cos_inclination * distance,
            np.sin(inclination) * distance,
        ),
        axis=-1,
    )
    transform = np.asarray(calibration["vehicle_from_laser"], dtype=np.float64)
    return (
        points_laser @ transform[:3, :3].T + transform[:3, 3]
    ).astype(np.float32)


def _target_geometry(
    calibrations: dict[int, dict[str, object]],
    resolution: int,
    patch_size: int,
) -> tuple[tuple[int, int], dict[int, np.ndarray], dict[int, tuple[np.ndarray, np.ndarray]]]:
    target_hw: tuple[int, int] | None = None
    target_intrinsics: dict[int, np.ndarray] = {}
    undistort_maps: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for name, calibration in calibrations.items():
        height, width = int(calibration["height"]), int(calibration["width"])
        item_hw = _target_shape(height, width, resolution, patch_size)
        if target_hw is None:
            target_hw = item_hw
        elif target_hw != item_hw:
            raise ValueError("The three Waymo cameras preprocess to different shapes")
        intrinsic = np.asarray(calibration["intrinsic"], dtype=np.float64)
        distortion = np.asarray(calibration["distortion"], dtype=np.float64)
        target_height, target_width = item_hw
        pixel_transform = np.diag(
            [target_width / width, target_height / height, 1.0]
        )
        target_intrinsics[name] = (pixel_transform @ intrinsic).astype(np.float32)
        # Produce the model-resolution undistorted image directly. This is
        # geometrically equivalent to full-resolution remap followed by resize,
        # but avoids remapping 2.46M pixels for every camera frame.
        map_x, map_y = cv2.initUndistortRectifyMap(
            intrinsic,
            distortion,
            None,
            target_intrinsics[name],
            (target_width, target_height),
            cv2.CV_32FC1,
        )
        undistort_maps[name] = (map_x, map_y)
    assert target_hw is not None
    return target_hw, target_intrinsics, undistort_maps


def _zbuffer_depth(
    points_vehicle: np.ndarray,
    world_from_vehicle: np.ndarray,
    world_from_camera_cv: np.ndarray,
    intrinsic: np.ndarray,
    image_hw: tuple[int, int],
    *,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    camera_from_vehicle = np.linalg.inv(world_from_camera_cv) @ world_from_vehicle
    points_camera = (
        points_vehicle.astype(np.float64) @ camera_from_vehicle[:3, :3].T
        + camera_from_vehicle[:3, 3]
    )
    z = points_camera[:, 2]
    valid = np.isfinite(points_camera).all(axis=1) & (z >= min_depth) & (z <= max_depth)
    points_camera = points_camera[valid]
    z = z[valid]
    if not z.size:
        return np.zeros(image_hw, dtype=np.uint16)
    u = intrinsic[0, 0] * points_camera[:, 0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * points_camera[:, 1] / z + intrinsic[1, 2]
    columns = np.rint(u).astype(np.int64)
    rows = np.rint(v).astype(np.int64)
    height, width = image_hw
    inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
    rows, columns, z = rows[inside], columns[inside], z[inside]
    flattened = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(flattened, rows * width + columns, z.astype(np.float32))
    depth = flattened.reshape(height, width)
    finite = np.isfinite(depth)
    encoded = np.zeros((height, width), dtype=np.uint16)
    encoded[finite] = np.clip(np.rint(depth[finite] * 100.0), 1, 65535).astype(
        np.uint16
    )
    return encoded


def prepare_waymo_segment(
    split_root: str | Path,
    output_path: str | Path,
    segment_name: str,
    *,
    resolution: int = 512,
    patch_size: int = 16,
    jpeg_quality: int = 95,
    min_depth: float = 0.5,
    max_depth: float = 80.0,
    evaluation_tenth_front_only: bool = False,
) -> dict[str, object]:
    """Materialize RGB and LiDAR depth; optionally restrict depth to eval queries."""
    parquet = _pyarrow_parquet()
    root = Path(split_root)
    output = Path(output_path)
    camera_calibration = _camera_calibrations(root / "camera_calibration" / segment_name)
    lidar_calibration = _lidar_calibrations(root / "lidar_calibration" / segment_name)
    image_hw, target_intrinsics, undistort_maps = _target_geometry(
        camera_calibration, resolution, patch_size
    )

    camera_columns = [
        "key.frame_timestamp_micros",
        "key.camera_name",
        "[CameraImageComponent].image",
        "[CameraImageComponent].pose.transform",
    ]
    camera_rows = parquet.read_table(
        root / "camera_image" / segment_name,
        columns=camera_columns,
        filters=[("key.camera_name", "in", list(CAMERA_NAMES))],
    ).to_pylist()
    cameras_by_timestamp: dict[int, dict[int, dict[str, object]]] = {}
    for row in camera_rows:
        timestamp = int(row["key.frame_timestamp_micros"])
        name = int(row["key.camera_name"])
        cameras_by_timestamp.setdefault(timestamp, {})[name] = row

    pose_rows = parquet.read_table(
        root / "vehicle_pose" / segment_name,
        columns=[
            "key.frame_timestamp_micros",
            "[VehiclePoseComponent].world_from_vehicle.transform",
        ],
    ).to_pylist()
    vehicle_poses = {
        int(row["key.frame_timestamp_micros"]): _matrix(
            row["[VehiclePoseComponent].world_from_vehicle.transform"]
        )
        for row in pose_rows
    }
    timestamps = sorted(
        timestamp
        for timestamp, rows in cameras_by_timestamp.items()
        if set(rows) == set(CAMERA_NAMES) and timestamp in vehicle_poses
    )
    if not timestamps:
        raise ValueError(f"No complete front-three-camera timestamps in {segment_name}")
    timestamp_to_slot = {timestamp: slot for slot, timestamp in enumerate(timestamps)}
    selected_depth_timestamps: set[int] | None = None
    selected_image_timestamps: set[int] | None = None
    if evaluation_tenth_front_only:
        values = np.asarray(timestamps, dtype=np.int64)

        def nearest(expected: int, *, enforce_tolerance: bool) -> int | None:
            position = int(np.searchsorted(values, expected))
            indices = [index for index in (position - 1, position) if 0 <= index < len(values)]
            if not indices:
                return None
            chosen = min(indices, key=lambda index: abs(int(values[index]) - expected))
            result = int(values[chosen])
            if enforce_tolerance and abs(result - expected) > 20_000:
                return None
            return result

        anchor = nearest(timestamps[0] + 9 * 200_000, enforce_tolerance=False)
        if anchor is None:
            raise ValueError(f"No tenth-frame anchor in {segment_name}")
        selected_depth_timestamps = {anchor}
        for step in range(1, 11):
            target = nearest(anchor + step * 200_000, enforce_tolerance=True)
            if target is not None:
                selected_depth_timestamps.add(target)
        selected_image_timestamps = set()
        for offset in (-600_000, -400_000, -200_000, 0):
            history = nearest(anchor + offset, enforce_tolerance=True)
            if history is None:
                raise ValueError(f"Missing strict history timestamp in {segment_name}")
            selected_image_timestamps.add(history)
        if len(selected_image_timestamps) != 4:
            raise ValueError(f"Repeated strict history timestamp in {segment_name}")

    world_from_camera = np.empty((len(timestamps), 3, 4, 4), dtype=np.float32)
    encoded_images: list[list[bytes]] = [[b""] * 3 for _ in timestamps]
    for timestamp in timestamps:
        slot = timestamp_to_slot[timestamp]
        for camera_slot, camera_name in enumerate(CAMERA_NAMES):
            row = cameras_by_timestamp[timestamp][camera_name]
            world_from_vehicle_image = _matrix(
                row["[CameraImageComponent].pose.transform"]
            )
            c2w = (
                world_from_vehicle_image
                @ np.asarray(
                    camera_calibration[camera_name]["vehicle_from_camera"],
                    dtype=np.float64,
                )
                @ CV_TO_WAYMO_CAMERA
            )
            world_from_camera[slot, camera_slot] = c2w.astype(np.float32)
            if selected_image_timestamps is not None and timestamp not in selected_image_timestamps:
                continue
            compressed = np.frombuffer(row["[CameraImageComponent].image"], dtype=np.uint8)
            image = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to decode {segment_name}:{timestamp}:{camera_name}")
            map_x, map_y = undistort_maps[camera_name]
            image = cv2.remap(
                image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            )
            success, jpeg = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
            )
            if not success:
                raise ValueError("OpenCV JPEG encoding failed")
            encoded_images[slot][camera_slot] = jpeg.tobytes()

    depth_cm = np.zeros((len(timestamps), 3, *image_hw), dtype=np.uint16)
    lidar_columns = [
        "key.frame_timestamp_micros",
        "key.laser_name",
        "[LiDARComponent].range_image_return1.values",
        "[LiDARComponent].range_image_return1.shape",
        "[LiDARComponent].range_image_return2.values",
        "[LiDARComponent].range_image_return2.shape",
    ]
    lidar_file = parquet.ParquetFile(root / "lidar" / segment_name)
    timestamp_rows: dict[int, list[dict[str, object]]] = {}

    def process_timestamp(timestamp: int, rows: list[dict[str, object]]) -> None:
        if timestamp not in timestamp_to_slot:
            return
        if selected_depth_timestamps is not None and timestamp not in selected_depth_timestamps:
            return
        points = []
        for row in rows:
            calibration = lidar_calibration[int(row["key.laser_name"])]
            for return_index in (1, 2):
                prefix = f"[LiDARComponent].range_image_return{return_index}."
                if row[prefix + "values"] is None:
                    continue
                points.append(
                    _range_image_points(
                        row[prefix + "values"], row[prefix + "shape"], calibration
                    )
                )
        if not points:
            return
        points_vehicle = np.concatenate(points, axis=0)
        slot = timestamp_to_slot[timestamp]
        for camera_slot, camera_name in enumerate(CAMERA_NAMES):
            if evaluation_tenth_front_only and camera_slot != 0:
                continue
            depth_cm[slot, camera_slot] = _zbuffer_depth(
                points_vehicle,
                vehicle_poses[timestamp],
                world_from_camera[slot, camera_slot].astype(np.float64),
                target_intrinsics[camera_name].astype(np.float64),
                image_hw,
                min_depth=min_depth,
                max_depth=max_depth,
            )

    for record_batch in lidar_file.iter_batches(batch_size=5, columns=lidar_columns):
        for row in record_batch.to_pylist():
            timestamp = int(row["key.frame_timestamp_micros"])
            timestamp_rows.setdefault(timestamp, []).append(row)
            if len(timestamp_rows[timestamp]) == len(lidar_calibration):
                process_timestamp(timestamp, timestamp_rows.pop(timestamp))
    for timestamp, rows in timestamp_rows.items():
        process_timestamp(timestamp, rows)

    payload = {
        "format": "waymo_front3_lidar_v1",
        "segment": Path(segment_name).stem,
        "timestamps": torch.tensor(timestamps, dtype=torch.int64),
        "camera_names": torch.tensor(CAMERA_NAMES, dtype=torch.int64),
        "images_jpeg": encoded_images,
        "world_from_camera_cv": torch.from_numpy(world_from_camera),
        "intrinsics": torch.stack(
            [torch.from_numpy(target_intrinsics[name]) for name in CAMERA_NAMES]
        ),
        "depth_cm": torch.from_numpy(depth_cm),
        "image_hw": image_hw,
        "undistorted": True,
        "depth_convention": "opencv_z_centimeters",
        "lidars": "all_five_first_and_second_returns",
        "depth_coverage": (
            "tenth_anchor_and_future_front_only"
            if evaluation_tenth_front_only else "all_frames_front_three"
        ),
        "image_coverage": (
            "four_strict_history_timestamps_front_three"
            if evaluation_tenth_front_only else "all_frames_front_three"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    valid_fraction = float((depth_cm > 0).mean())
    return {
        "segment": Path(segment_name).stem,
        "timestamps": len(timestamps),
        "image_hw": image_hw,
        "valid_depth_fraction": valid_fraction,
        "bytes": output.stat().st_size,
        "output": str(output),
    }


class WaymoPreparedDataset:
    """Scene-token tasks backed by prepared Waymo front-camera data.

    Frozen VGGT may see multiple timestamps. ``scene_tokenizer_view_mask`` is
    true only for cameras from one selected timestamp.  With a static temporal
    manifest, memory reads exactly that timestamp's front three cameras and
    the target is a future FRONT camera that never enters frozen VGGT.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        seed: int = 20260731,
        samples_per_segment: int = 128,
        decode_workers: int = 8,
        max_context_frame_delta: int = 10,
        static_temporal_manifest: str | Path,
    ) -> None:
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No prepared Waymo .pt segments under {self.root}")
        self.seed = int(seed)
        self.samples_per_segment = max(1, int(samples_per_segment))
        self.max_context_frame_delta = max(1, int(max_context_frame_delta))
        self.static_temporal_manifest = Path(static_temporal_manifest)
        self._static_records_by_segment: dict[str, list[dict[str, object]]] = {}
        self._static_records: list[dict[str, object]] = []
        if not self.static_temporal_manifest.is_file():
            raise FileNotFoundError(self.static_temporal_manifest)
        with self.static_temporal_manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if str(record.get("pool")) != "static_temporal":
                    raise ValueError("Static Waymo training received a non-static window")
                if int(record.get("state_count", 0)) != 11:
                    raise ValueError("Static Waymo windows must contain 11 states")
                self._static_records.append(record)
                self._static_records_by_segment.setdefault(
                    str(record["segment"]), []
                ).append(record)
        if not self._static_records:
            raise ValueError("Static Waymo temporal manifest is empty")
        available = {path.stem: path for path in self.files}
        missing = sorted(set(self._static_records_by_segment) - set(available))
        if missing:
            raise FileNotFoundError(
                f"Prepared Waymo payload missing for static segment {missing[0]}"
            )
        self.files = [available[name] for name in sorted(self._static_records_by_segment)]
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(decode_workers)))
        self._cached_path: Path | None = None
        self._cached_payload: dict[str, object] | None = None
        self._ordered_files = self._stable_file_order(self.files)
        self._last_refresh_block = -1

    def _stable_file_order(self, files: list[Path]) -> list[Path]:
        return sorted(
            files,
            key=lambda path: hashlib.sha1(
                f"{self.seed}:{path.name}".encode("utf-8")
            ).digest(),
        )

    def _refresh_files(self) -> None:
        current = sorted(self.root.glob("*.pt"))
        if self._static_records_by_segment:
            allowed = set(self._static_records_by_segment)
            current = [path for path in current if path.stem in allowed]
        if len(current) != len(self.files) or set(current) != set(self.files):
            self.files = current
            self._ordered_files = self._stable_file_order(current)

    @staticmethod
    def _decode(jpeg: bytes) -> torch.Tensor:
        with Image.open(io.BytesIO(jpeg)) as image:
            # Keep prefetched images compact.  Conversion to [0,1] float runs
            # once on the GPU after a pinned batched transfer.
            array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1)

    def _payload(self, sample_cursor: int) -> dict[str, object]:
        block = sample_cursor // self.samples_per_segment
        if block != self._last_refresh_block:
            self._refresh_files()
            self._last_refresh_block = block
        path = self._ordered_files[block % len(self._ordered_files)]
        if path != self._cached_path:
            self._cached_payload = torch.load(path, map_location="cpu", weights_only=False)
            self._cached_path = path
        assert self._cached_payload is not None
        return self._cached_payload

    def _build_static_future(
        self,
        payload: dict[str, object],
        record: dict[str, object],
        *,
        total_views: int,
        rng: random.Random,
        horizon_index: int | None = None,
        validation: bool = False,
    ) -> dict[str, object]:
        """Build a causal query from a motion-free 2-second temporal window."""
        if total_views < 3:
            raise ValueError("Static future Waymo tasks require at least three input views")
        logical_slots = [int(value) for value in record["logical_slots"]]
        if len(logical_slots) != 11:
            raise ValueError("Static future window must have 11 logical states")
        if horizon_index is None:
            horizon_index = rng.randint(1, 10)
            source_state = rng.randint(0, 10 - horizon_index)
        else:
            if not 1 <= horizon_index <= 10:
                raise ValueError("Future validation horizon index must be in [1,10]")
            source_state = 0
        target_state = source_state + horizon_index
        source_time_slot = logical_slots[source_state]
        target_time_slot = logical_slots[target_state]
        target_camera = 0  # FRONT only.

        # FRONT first fixes both the local camera origin and the paired scale.
        source_pairs = [(source_time_slot, camera) for camera in range(3)]
        context_candidates = [
            (time_slot, camera)
            for time_slot in logical_slots
            if time_slot not in (source_time_slot, target_time_slot)
            for camera in range(3)
        ]
        context_count = total_views - len(source_pairs)
        if context_count > len(context_candidates):
            raise ValueError("Not enough non-query static context views")
        context_pairs = rng.sample(context_candidates, context_count)
        input_pairs = source_pairs + context_pairs
        query_pair = (target_time_slot, target_camera)
        if query_pair in input_pairs:
            raise AssertionError("Future query RGB leaked into frozen VGGT input")
        if any(time == target_time_slot for time, _ in input_pairs):
            raise AssertionError("The entire future query timestamp must be hidden")
        token_mask = torch.tensor(
            [index < 3 for index in range(len(input_pairs))], dtype=torch.bool
        )

        timestamps = torch.as_tensor(payload["timestamps"])
        world_c2w = torch.as_tensor(payload["world_from_camera_cv"]).float()
        intrinsics = torch.as_tensor(payload["intrinsics"]).float()
        global_input_c2w = torch.stack(
            [world_c2w[time, camera] for time, camera in input_pairs]
        ).double()
        first_from_world = torch.linalg.inv(global_input_c2w[0])
        input_c2w = (first_from_world.unsqueeze(0) @ global_input_c2w).float()
        target_c2w = (
            first_from_world @ world_c2w[target_time_slot, target_camera].double()
        ).float()
        if not torch.allclose(input_c2w[0], torch.eye(4), atol=2e-4, rtol=0.0):
            raise AssertionError("Static source FRONT camera is not the local origin")

        jpeg_rows = payload["images_jpeg"]
        input_images = torch.stack(
            list(
                self._pool.map(
                    self._decode,
                    [jpeg_rows[time][camera] for time, camera in input_pairs],
                )
            )
        )
        input_intrinsics = torch.stack(
            [intrinsics[camera] for _, camera in input_pairs]
        )
        depth_rows = torch.as_tensor(payload["depth_cm"])
        input_scale_depth = depth_rows[source_time_slot, 0].float() / 100.0
        input_scale_valid = (
            torch.isfinite(input_scale_depth)
            & (input_scale_depth > 0.0)
            & (input_scale_depth <= 80.0)
        )
        target_depth = depth_rows[target_time_slot, target_camera].float() / 100.0
        target_valid = (
            torch.isfinite(target_depth)
            & (target_depth > 0.0)
            & (target_depth <= 80.0)
        )
        target_depth = torch.where(
            target_valid, target_depth, torch.zeros_like(target_depth)
        )
        target_confidence = target_valid.float()
        target_scale = (
            target_depth[target_valid].median()
            if target_valid.any()
            else target_depth.new_tensor(1.0)
        )
        target_timestamp = int(timestamps[target_time_slot])
        input_frame_ids = [
            f"{int(timestamps[time])}_cam{camera + 1}"
            for time, camera in input_pairs
        ]
        input_indices = [
            int(timestamps[time]) * 10 + camera + 1
            for time, camera in input_pairs
        ]
        source_timestamp = int(timestamps[source_time_slot])
        target_id = f"{target_timestamp}_cam1"
        if target_id in input_frame_ids:
            raise AssertionError("Future query frame id leaked into frozen VGGT")
        horizon_seconds = (target_timestamp - source_timestamp) / 1e6
        expected_seconds = horizon_index * 0.2
        if abs(horizon_seconds - expected_seconds) > 0.03:
            raise ValueError("Static temporal window no longer follows 0.2-second states")
        identity = {
            "scale": torch.tensor(1.0),
            "rotation": torch.eye(3),
            "translation": torch.zeros(3),
        }
        return {
            "scene_id": (
                f"{payload['segment']}/{source_timestamp}/future_{horizon_seconds:.1f}"
            ),
            "dataset_name": "waymo",
            "input_images": input_images,
            "input_c2w": input_c2w,
            "input_intrinsics": input_intrinsics,
            "input_frame_ids": input_frame_ids,
            "input_frame_indices": input_indices,
            "teacher_register_frame_index": input_indices[0],
            "teacher_register_frame_id": input_frame_ids[0],
            "teacher_scene_scale": float(target_scale),
            "input_scale_depth_dataset": input_scale_depth,
            "input_scale_valid_dataset": input_scale_valid,
            "input_scale_view_index": 0,
            "scene_tokenizer_view_mask": token_mask,
            "reconstruction_task": "future_static",
            "scene_token_view_count": 3,
            "student_vggt_timestamp_count": len({time for time, _ in input_pairs}),
            "target_timestamp": target_timestamp,
            "target_horizon_seconds": float(horizon_seconds),
            "target_frame_ids": [target_id],
            "target_frame_indices": [target_timestamp * 10 + 1],
            "target_c2w_dataset_multi": target_c2w.unsqueeze(0),
            "target_intrinsics_multi": intrinsics[0].unsqueeze(0),
            "target_depth_teacher_multi": target_depth.unsqueeze(0),
            "target_confidence_multi": target_confidence.unsqueeze(0),
            "target_valid_multi": target_valid.unsqueeze(0),
            "target_c2w_teacher_multi": target_c2w.unsqueeze(0),
            "target_depth_dataset_multi": target_depth.unsqueeze(0),
            "target_valid_dataset_multi": target_valid.unsqueeze(0),
            "target_confidence_dataset_multi": target_confidence.unsqueeze(0),
            "target_scene_scale_dataset_multi": target_scale.reshape(1),
            "target_uses_dataset_depth_multi": torch.ones(1, dtype=torch.bool),
            "dynamic_downweighted_fraction_multi": torch.zeros(1),
            "dataset_to_teacher": identity,
            "image_hw": tuple(int(value) for value in payload["image_hw"]),
            "static_temporal_window_id": str(record["window_id"]),
            "future_validation_tuple": bool(validation),
        }

    def _static_record(
        self, payload: dict[str, object], rng: random.Random
    ) -> dict[str, object]:
        records = self._static_records_by_segment[str(payload["segment"])]
        return records[rng.randrange(len(records))]

    def build_static_validation_tuple(
        self,
        example_index: int,
        horizon_index: int,
    ) -> dict[str, object]:
        if not self._static_records:
            raise RuntimeError("No static temporal manifest was configured")
        ordered = sorted(
            self._static_records,
            key=lambda item: hashlib.sha1(
                f"{self.seed}:validation:{item['window_id']}".encode("utf-8")
            ).digest(),
        )
        record = ordered[example_index % len(ordered)]
        path = self.root / f"{record['segment']}.pt"
        if path != self._cached_path:
            self._cached_payload = torch.load(
                path, map_location="cpu", weights_only=False
            )
            self._cached_path = path
        assert self._cached_payload is not None
        return self._build_static_future(
            self._cached_payload,
            record,
            total_views=3,
            rng=random.Random(self.seed + example_index * 1_000_003),
            horizon_index=horizon_index,
            validation=True,
        )

    def get_compatible_scene_batch(
        self, cursor: int, batch_size: int, input_views: int
    ) -> tuple[list[dict[str, object]], int]:
        if not 3 <= input_views <= 8:
            raise ValueError("Waymo frozen-VGGT total views must be in [3, 8]")
        payload = self._payload(cursor)
        batches = []
        for index in range(batch_size):
            rng = random.Random(self.seed + (cursor + index) * 1_000_003)
            batches.append(
                self._build_static_future(
                    payload,
                    self._static_record(payload, rng),
                    total_views=input_views,
                    rng=rng,
                )
            )
        return batches, cursor + batch_size

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)

    def __len__(self) -> int:
        return len(self.files)


def _prepare_main(args: argparse.Namespace) -> None:
    split_root = Path(args.waymo_root) / args.split
    output_root = Path(args.output_root) / args.split
    names = discover_segment_names(split_root)
    names = names[args.shard_index :: args.shard_count]
    if args.limit > 0:
        names = names[: args.limit]
    print(json.dumps({"segments": len(names), "split": args.split, "output": str(output_root)}))
    for index, name in enumerate(names):
        output = output_root / f"{Path(name).stem}.pt"
        if output.is_file() and not args.overwrite:
            print(json.dumps({"status": "exists", "index": index, "output": str(output)}))
            continue
        started = time.perf_counter()
        result = prepare_waymo_segment(
            split_root,
            output,
            name,
            resolution=args.resolution,
            patch_size=args.patch_size,
            jpeg_quality=args.jpeg_quality,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        )
        result.update(status="prepared", index=index, seconds=time.perf_counter() - started)
        print(json.dumps(result))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--waymo-root", default="data/waymo")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--min-depth", type=float, default=0.5)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    _prepare_main(parser.parse_args())


if __name__ == "__main__":
    main()
