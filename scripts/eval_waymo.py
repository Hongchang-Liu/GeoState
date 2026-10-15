#!/usr/bin/env python3
"""Evaluate a 4-state Waymo world model with a pure ten-step rollout.

The history VGGT pass sees exactly t, t-0.6, t-0.4, t-0.2, with either three
cameras or FRONT only at each timestamp.  Images are ordered exactly as online
4->1 training so that t/FRONT is the reference view.  The tokenizer returns chronological states
t-0.6, t-0.4, t-0.2, t.  Future images and future teacher states are never
constructed.  The predictor rolls out ten states autoregressively and each is
decoded at the corresponding calibrated future target camera (FRONT by default).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    import cv2
except ModuleNotFoundError:  # prepared-tensor evaluation does not require OpenCV
    cv2 = None
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
sys.path.insert(0, str(SCRIPT.parent.parent))

from GeoState.models.world_model import (  # noqa: E402
    SceneStatePredictorConfig,
    build_next_scene_state_predictor,
)
from GeoState.evaluation.checkpoint import load_geostate_checkpoint as _load_model  # noqa: E402
from GeoState.core.geometry import (  # noqa: E402
    paired_depth_scale,
)
from GeoState.core.types import VGGTOutput  # noqa: E402
from GeoState.data.waymo import (  # noqa: E402
    CAMERA_NAMES,
    CV_TO_WAYMO_CAMERA,
    _camera_calibrations,
    _matrix,
    _pyarrow_parquet,
    _target_geometry,
)
from GeoState.evaluation.inference import decode_metric_depth as _decode_meters  # noqa: E402
from GeoState.models.camera_prediction import (  # noqa: E402
    rotation_angle,
)
from GeoState.models.camera_conditioning import (  # noqa: E402
    load_vggt_spatial_camera_checkpoint,
)


HISTORY_OFFSETS_US = (-600_000, -400_000, -200_000, 0)
HORIZON_SECONDS = tuple(step / 5.0 for step in range(1, 11))
TRAINING_INPUT_STATE_ORDER = (3, 0, 1, 2)
CHRONOLOGICAL_FRONT_INDICES = tuple(
    TRAINING_INPUT_STATE_ORDER.index(state) * 3 for state in range(4)
)
TARGET_HW = (416, 624)


def _complete_timestamps(parquet, root: Path, filename: str) -> list[int]:
    camera_table = parquet.read_table(
        root / "camera_image" / filename,
        columns=["key.frame_timestamp_micros", "key.camera_name"],
        filters=[("key.camera_name", "in", list(CAMERA_NAMES))],
    )
    cameras: dict[int, set[int]] = defaultdict(set)
    for row in camera_table.to_pylist():
        cameras[int(row["key.frame_timestamp_micros"])].add(int(row["key.camera_name"]))
    pose_table = parquet.read_table(
        root / "vehicle_pose" / filename,
        columns=["key.frame_timestamp_micros"],
    )
    poses = {int(value.as_py()) for value in pose_table["key.frame_timestamp_micros"]}
    return sorted(
        timestamp for timestamp, names in cameras.items()
        if names == set(CAMERA_NAMES) and timestamp in poses
    )


def _resize_depth(depth: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        depth[None, None].float(), size=TARGET_HW, mode="nearest-exact"
    )[0, 0]


def _resize_intrinsics(value: torch.Tensor, source_hw: tuple[int, int]) -> torch.Tensor:
    source_h, source_w = source_hw
    transform = value.new_tensor([
        [TARGET_HW[1] / source_w, 0.0, 0.0],
        [0.0, TARGET_HW[0] / source_h, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return transform @ value


def _nearest_timestamps(
    available: list[int], expected: list[int], tolerance_us: int,
) -> tuple[list[int], list[int]]:
    values = np.asarray(available, dtype=np.int64)
    selected: list[int] = []
    errors: list[int] = []
    for target in expected:
        position = int(np.searchsorted(values, target))
        candidates = [index for index in (position - 1, position) if 0 <= index < len(values)]
        if not candidates:
            raise ValueError(f"no timestamp near {target}")
        chosen = min(candidates, key=lambda index: abs(int(values[index]) - target))
        value = int(values[chosen])
        error = value - target
        if abs(error) > tolerance_us:
            raise ValueError(
                f"nearest history timestamp error {error} us exceeds {tolerance_us} us"
            )
        selected.append(value)
        errors.append(error)
    if len(set(selected)) != len(selected):
        raise ValueError("history timestamps are not unique")
    return selected, errors


def _decode_prepared_jpeg(value: bytes) -> torch.Tensor:
    with Image.open(io.BytesIO(value)) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _history_from_prepared(
    path: Path, anchor_timestamp: int, tolerance_us: int,
) -> dict[str, Any]:
    # Some prepared RGB records store encoded images as plain bytes.
    with torch.serialization.safe_globals([bytes]):
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    timestamps = [int(value) for value in torch.as_tensor(payload["timestamps"]).tolist()]
    chronological, errors = _nearest_timestamps(
        timestamps,
        [anchor_timestamp + offset for offset in HISTORY_OFFSETS_US],
        tolerance_us,
    )
    index = {timestamp: offset for offset, timestamp in enumerate(timestamps)}
    # Training physically places t first to make it the VGGT/reference anchor.
    ordered = [chronological[state] for state in TRAINING_INPUT_STATE_ORDER]
    images = torch.stack([
        _decode_prepared_jpeg(payload["images_jpeg"][index[timestamp]][camera])
        for timestamp in ordered for camera in range(3)
    ])
    world_c2w = torch.stack([
        torch.as_tensor(payload["world_from_camera_cv"])[index[timestamp], camera].float()
        for timestamp in ordered for camera in range(3)
    ])
    result = {
        "images": images,
        "world_c2w": world_c2w,
        "intrinsics": torch.as_tensor(payload["intrinsics"]).float().clone(),
        "image_hw": tuple(int(value) for value in payload["image_hw"]),
        # Retain mmap-backed tensors so side-camera GT can reuse this one
        # payload load rather than reopening the 300--400 MB segment archive.
        "prepared_timestamps": torch.as_tensor(payload["timestamps"]),
        "prepared_camera_names": torch.as_tensor(payload["camera_names"]),
        "prepared_world_c2w": torch.as_tensor(payload["world_from_camera_cv"]),
        "prepared_depth_cm": torch.as_tensor(payload["depth_cm"]),
        "chronological_timestamps": chronological,
        "timestamp_errors_us": errors,
        "source": "full_prepared_payload",
    }
    del payload
    return result


def _history_from_raw(
    root: Path, segment: str, anchor_timestamp: int, tolerance_us: int,
) -> dict[str, Any]:
    parquet = _pyarrow_parquet()
    filename = f"{segment}.parquet"
    complete = _complete_timestamps(parquet, root, filename)
    chronological, errors = _nearest_timestamps(
        complete,
        [anchor_timestamp + offset for offset in HISTORY_OFFSETS_US],
        tolerance_us,
    )
    ordered = [chronological[state] for state in TRAINING_INPUT_STATE_ORDER]
    calibration = _camera_calibrations(root / "camera_calibration" / filename)
    image_hw, target_intrinsics, undistort_maps = _target_geometry(
        calibration, 512, 16
    )
    rows = parquet.read_table(
        root / "camera_image" / filename,
        columns=[
            "key.frame_timestamp_micros", "key.camera_name",
            "[CameraImageComponent].image", "[CameraImageComponent].pose.transform",
        ],
        filters=[
            ("key.frame_timestamp_micros", "in", ordered),
            ("key.camera_name", "in", list(CAMERA_NAMES)),
        ],
    ).to_pylist()
    by_key = {
        (int(row["key.frame_timestamp_micros"]), int(row["key.camera_name"])): row
        for row in rows
    }
    required = {(timestamp, camera) for timestamp in ordered for camera in CAMERA_NAMES}
    if required - set(by_key):
        raise ValueError(f"missing raw history camera rows: {sorted(required - set(by_key))}")
    images: list[torch.Tensor] = []
    world_c2w: list[torch.Tensor] = []
    for timestamp in ordered:
        for camera in CAMERA_NAMES:
            row = by_key[(timestamp, camera)]
            compressed = np.frombuffer(row["[CameraImageComponent].image"], dtype=np.uint8)
            image = cv2.imdecode(compressed, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"failed to decode {segment}:{timestamp}:cam{camera}")
            map_x, map_y = undistort_maps[camera]
            image = cv2.remap(
                image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(
                torch.from_numpy(image.astype(np.float32) / 255.0)
                .permute(2, 0, 1).contiguous()
            )
            global_c2w = (
                _matrix(row["[CameraImageComponent].pose.transform"])
                @ np.asarray(calibration[camera]["vehicle_from_camera"], dtype=np.float64)
                @ CV_TO_WAYMO_CAMERA
            )
            world_c2w.append(torch.from_numpy(global_c2w.astype(np.float32)))
    return {
        "images": torch.stack(images),
        "world_c2w": torch.stack(world_c2w),
        "intrinsics": torch.stack([
            torch.from_numpy(target_intrinsics[camera]).float() for camera in CAMERA_NAMES
        ]),
        "image_hw": tuple(int(value) for value in image_hw),
        "chronological_timestamps": chronological,
        "timestamp_errors_us": errors,
        "source": "official_raw_parquet",
    }


def _load_history(
    prepared_root: Path,
    raw_root: Path,
    segment: str,
    anchor_timestamp: int,
    tolerance_us: int,
) -> dict[str, Any]:
    prepared = prepared_root / f"{segment}.pt"
    if prepared.is_file():
        return _history_from_prepared(prepared, anchor_timestamp, tolerance_us)
    return _history_from_raw(raw_root, segment, anchor_timestamp, tolerance_us)


def _context(
    encoding: Any,
    target_c2w: torch.Tensor,
    target_intrinsics: torch.Tensor,
    target_hw: tuple[int, int],
) -> dict[str, Any]:
    return {
        "reference_c2w_student": encoding.reference_c2w_student[0].float().cpu(),
        "dataset_to_student_scale": encoding.dataset_to_student.scale[0].float().cpu(),
        "dataset_to_student_rotation": encoding.dataset_to_student.rotation[0].float().cpu(),
        "dataset_to_student_translation": encoding.dataset_to_student.translation[0].float().cpu(),
        "scene_scale": encoding.scene_scale[0].float().cpu(),
        "target_c2w": target_c2w.float().cpu(),
        "target_intrinsics": target_intrinsics.float().cpu(),
        "target_hw": target_hw,
    }


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float | int]:
    valid = torch.isfinite(target) & (target > 0) & (target <= 80.0)
    pixels = int(valid.sum())
    if pixels == 0:
        raise ValueError("no valid GT pixels")
    if not bool((torch.isfinite(prediction) & (prediction > 0))[valid].all()):
        raise ValueError("non-finite or non-positive prediction on valid GT pixels")
    prediction = prediction[valid].float()
    target = target[valid].float()
    relative = (prediction - target).abs() / target.clamp_min(1e-8)
    delta = torch.maximum(
        prediction / target.clamp_min(1e-8),
        target / prediction.clamp_min(1e-8),
    ).lt(1.25)
    return {
        "abs_rel": float(relative.mean()),
        "delta1": float(delta.float().mean()),
        "valid_pixels": pixels,
        "abs_rel_sum": float(relative.sum()),
        "delta1_sum": float(delta.sum()),
    }


def _repeat_vggt_batch(output: VGGTOutput, repeats: int = 2) -> VGGTOutput:
    """Repeat a singleton output so source-faithful batched rejection can run.

    ``encode_vggt_output`` deliberately raises for an invalid singleton, while
    online training returns a per-element validity mask for batches.  Repeating
    the already-computed output changes neither geometry nor the number of VGGT
    forwards; it only exposes the same validity diagnostic used by training.
    """
    if output.c2w.shape[0] != 1:
        raise ValueError("expected singleton VGGT output")

    def repeat(value: torch.Tensor) -> torch.Tensor:
        return value.expand(repeats, *value.shape[1:])

    return VGGTOutput(
        patch_features=[repeat(value) for value in output.patch_features],
        depth=repeat(output.depth),
        confidence=repeat(output.confidence),
        world_points=repeat(output.world_points),
        c2w=repeat(output.c2w),
        intrinsics=repeat(output.intrinsics),
        patch_size=output.patch_size,
        patch_hw=output.patch_hw,
    )


@torch.inference_mode()
def main() -> None:
    global TARGET_HW
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--sparse-cache-root", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument(
        "--history-cache-root",
        help="Optional prepared history-only cache; avoids rereading raw camera parquet.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene-checkpoint", required=True)
    parser.add_argument("--world-checkpoint", required=True)
    parser.add_argument(
        "--camera-checkpoint", required=True,
        help=(
            "Anchored FRONT-camera predictor used for the predicted-camera metrics."
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--world-device",
        help="Optional separate device for the world and camera predictors.",
    )
    parser.add_argument(
        "--expected-seed", type=int, default=20260804,
        help="Require the selection manifest to use this random seed.",
    )
    parser.add_argument("--timestamp-tolerance-ms", type=float, default=20.0)
    parser.add_argument(
        "--only-segment", action="append",
        help="Evaluate only this segment after validating the full fixed selection.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Split the validated selection into this many deterministic strided shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based strided shard to evaluate.",
    )
    args = parser.parse_args()

    TARGET_HW = (416, 624)
    # The strict helper's resize functions intentionally consult its module-level
    # target.  Keep RGB/depth/intrinsics on one explicit raster protocol.

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    segments = list(selection["selected_segments"])
    selection_expected_count = int(selection.get("selected_count", len(segments)))
    if (
        selection.get("seed") != args.expected_seed
        or int(selection.get("population_count", 0)) != 202
        or len(segments) != selection_expected_count
        or len(set(segments)) != len(segments)
        or segments != sorted(segments)
        or selection_expected_count != 100
    ):
        raise ValueError(
            f"invalid fixed seed-{args.expected_seed} selection: "
            f"{len(segments)} rows, declared {selection_expected_count}"
        )
    if args.only_segment:
        requested = list(dict.fromkeys(args.only_segment))
        unavailable = sorted(set(requested) - set(segments))
        if unavailable:
            raise ValueError(f"--only-segment is not in the fixed selection: {unavailable}")
        segments = requested
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError(
            f"invalid shard {args.shard_index}/{args.shard_count}; "
            "require shard_count >= 1 and 0 <= shard_index < shard_count"
        )
    segments = segments[args.shard_index::args.shard_count]
    # A directory target is safe for either an explicit subset or the complete
    # selected population: every segment writes a uniquely named payload.

    device = torch.device(args.device)
    world_device = torch.device(args.world_device or args.device)
    _config, _adapter, geometry_model = _load_model(
        args.config, Path(args.scene_checkpoint), device
    )
    geometry_model.eval().requires_grad_(False)
    checkpoint = torch.load(
        args.world_checkpoint, map_location="cpu", mmap=True, weights_only=True
    )
    predictor_config = SceneStatePredictorConfig(**checkpoint["config"])
    predictor = build_next_scene_state_predictor(predictor_config).to(
        world_device
    )
    predictor.load_state_dict(checkpoint["model"], strict=True)
    predictor.eval().requires_grad_(False)
    world_validation = checkpoint.get("validation")
    residual_base = str(checkpoint.get("residual_base", "previous"))
    if residual_base != "previous":
        raise ValueError("The paper protocol requires residual_base='previous'")
    del checkpoint

    camera_checkpoint = torch.load(
        args.camera_checkpoint, map_location="cpu", weights_only=True
    )
    if camera_checkpoint.get("format") != "vggt_spatial_camera_predictor_v1":
        raise ValueError("The paper protocol requires a VGGT-spatial camera predictor")
    if camera_checkpoint.get("camera_patch_context") == "joint_12view_current_front_output":
        raise ValueError("The paper camera predictor requires an independent current-FRONT VGGT pass")
    camera_predictor, camera_checkpoint, camera_config = load_vggt_spatial_camera_checkpoint(
        args.camera_checkpoint,
        world_device,
    )
    camera_feature_kind = "current_front_independent_vggt_spatial_6x8"
    if camera_config.history != 4:
        raise ValueError(
            f"Expected a four-frame camera predictor, got history={camera_config.history}"
        )
    camera_predictor.eval().requires_grad_(False)
    del camera_checkpoint

    sparse_root = Path(args.sparse_cache_root)
    prepared_root = Path(args.prepared_root)
    raw_root = Path(args.raw_root)
    tolerance_us = int(round(args.timestamp_tolerance_ms * 1000))
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    copy_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    predicted_camera_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    predicted_camera_copy_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    camera_error_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    rows: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        started = time.perf_counter()
        sparse = torch.load(
            sparse_root / f"{segment}.pt", map_location="cpu", mmap=True,
            weights_only=True,
        )
        anchor_timestamp = int(sparse["anchor_timestamp"])
        if args.history_cache_root:
            cached_history = torch.load(
                Path(args.history_cache_root) / f"{segment}.pt",
                map_location="cpu", mmap=True, weights_only=True,
            )
            chronological = torch.as_tensor(
                cached_history["timestamps"], dtype=torch.int64
            ).tolist()
            # The cache stores timestamps chronologically; mirror the normal
            # raw/prepared loader by physically presenting t first to VGGT.
            ordered_indices = [3, 0, 1, 2]
            images = torch.stack([
                _decode_prepared_jpeg(cached_history["images_jpeg"][index][camera])
                for index in ordered_indices for camera in range(3)
            ])
            world_c2w = torch.stack([
                torch.as_tensor(cached_history["world_from_camera_cv"])[index, camera].float()
                for index in ordered_indices for camera in range(3)
            ])
            history_item = {
                "images": images,
                "world_c2w": world_c2w,
                "intrinsics": torch.as_tensor(cached_history["intrinsics"]).float().clone(),
                "image_hw": tuple(int(value) for value in cached_history["image_hw"]),
                "chronological_timestamps": chronological,
                "timestamp_errors_us": torch.as_tensor(
                    cached_history["timestamp_errors_us"], dtype=torch.int64
                ).tolist(),
                "source": "prepared_history_cache",
            }
        else:
            history_item = _load_history(
                prepared_root, raw_root, segment, anchor_timestamp, tolerance_us
            )
        if history_item["chronological_timestamps"][-1] != anchor_timestamp:
            raise ValueError(f"history anchor mismatch for {segment}")
        history_camera_slots = (0, 1, 2)
        physical_history_indices = [
            state_position * 3 + camera
            for state_position in range(4)
            for camera in history_camera_slots
        ]
        images = history_item["images"][physical_history_indices].to(device)
        world_c2w = history_item["world_c2w"][physical_history_indices].double()
        views_per_state = len(history_camera_slots)
        history_view_count = 4 * views_per_state
        anchor_from_world = torch.linalg.inv(world_c2w[0])
        input_c2w = (anchor_from_world[None] @ world_c2w).float().to(device)
        input_c2w[0] = torch.eye(4, device=device)
        predicted_query_c2w = None
        camera_history_world = None
        future_seconds = None
        if camera_predictor is not None:
            chronological_front_indices = tuple(
                TRAINING_INPUT_STATE_ORDER.index(state) * views_per_state
                for state in range(4)
            )
            camera_history_world = world_c2w[
                list(chronological_front_indices)
            ].float().to(world_device)
            if not torch.allclose(
                camera_history_world[-1].double().cpu(), world_c2w[0],
                atol=1e-5, rtol=1e-6,
            ):
                raise ValueError(f"camera-predictor anchor mismatch for {segment}")
            future_seconds = (
                torch.arange(1, 11, device=world_device, dtype=torch.float32) * 0.2
            )
            if camera_feature_kind is None:
                predicted_query_c2w = camera_predictor(
                    camera_history_world[None], future_seconds
                )["predicted_relative"][0].float()
        intrinsics = history_item["intrinsics"].float()
        input_intrinsics = torch.stack([
            intrinsics[camera]
            for _ in range(4)
            for camera in history_camera_slots
        ]).to(device)
        view_states = torch.tensor(
            [
                state
                for state in TRAINING_INPUT_STATE_ORDER
                for _ in history_camera_slots
            ],
            device=device,
            dtype=torch.long,
        )[None]
        state_mask = torch.stack(
            [(view_states == state) for state in range(4)], dim=1
        )
        vggt_calls = 0
        camera_feature_aggregator_calls = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            history_output_single = geometry_model.vggt(images[None])
            vggt_calls += 1
            aggregated, patch_start = geometry_model.vggt.model.aggregator(
                images[None, :1]
            )
            camera_feature_aggregator_calls += 1
            patch = aggregated[-1][:, 0, patch_start:].float()
            patch_h, patch_w = images.shape[-2] // 16, images.shape[-1] // 16
            if patch.shape[1] != patch_h * patch_w:
                raise RuntimeError("camera feature patch grid mismatch")
            patch = patch.reshape(
                1, patch_h, patch_w, patch.shape[-1]
            ).permute(0, 3, 1, 2)
            current_front_features = F.adaptive_avg_pool2d(
                patch, (6, 8)
            ).flatten(2).transpose(1, 2).to(world_device)
            # Keep pose arithmetic in FP32 because global Waymo camera centers
            # make reduced-precision relative-pose subtraction inaccurate.
            with torch.autocast("cuda", enabled=False):
                predicted_query_c2w = camera_predictor(
                    camera_history_world[None].float(),
                    future_seconds.float(),
                    current_features=current_front_features.float(),
                )["predicted_relative"][0].float()
            # depth_cm[0] is the anchor-t FRONT LiDAR z-depth. No target or
            # future LiDAR is read to establish the metric gauge.
            anchor_lidar_depth = torch.as_tensor(sparse["depth_cm"])[0].float().to(device).div_(100.0)
            paired_scale, paired_scale_valid, paired_scale_support = paired_depth_scale(
                history_output_single.depth[:, 0],
                anchor_lidar_depth[None],
                history_output_single.confidence[:, 0],
            )
            # Training evaluates Sim(3) validity per element in a batch.  The
            # model intentionally throws for batch size one, so repeat the
            # already-computed tensors (not the VGGT forward) to recover that
            # exact validity mask and reject the same way as training.
            history_output = _repeat_vggt_batch(history_output_single)
            anchor_alignment_mask = torch.zeros(
                (2, history_view_count), dtype=torch.bool, device=device
            )
            anchor_alignment_mask[:, 0] = True
            history_states, history_encoding = geometry_model.encode_temporal_scene_states_from_vggt_output(
                history_output,
                input_c2w[None].expand(2, -1, -1, -1),
                input_intrinsics=input_intrinsics[None].expand(2, -1, -1, -1),
                state_view_mask=state_mask.expand(2, -1, -1),
                frame_ids=[[ 
                    f"{timestamp}_{CAMERA_NAMES[camera]}"
                    for timestamp in [
                        history_item["chronological_timestamps"][state]
                        for state in TRAINING_INPUT_STATE_ORDER
                    ]
                    for camera in history_camera_slots
                ]] * 2,
                reference_index=0,
                input_camera_conditioning_source="vggt",
                single_view_scale=paired_scale.expand(2),
                alignment_view_mask=anchor_alignment_mask,
                scene_scale_view_mask=torch.ones(
                    (2, history_view_count), dtype=torch.bool, device=device
                ),
            )
        if vggt_calls != 1:
            raise RuntimeError(f"future leakage audit failed: {vggt_calls} VGGT calls")
        expected_camera_calls = 1
        if camera_feature_aggregator_calls != expected_camera_calls:
            raise RuntimeError(
                "camera feature leakage audit failed: "
                f"expected {expected_camera_calls}, got {camera_feature_aggregator_calls}"
            )
        alignment = history_encoding.metadata["alignment"]
        # Match the training job's unified-validation policy: an alignment
        # diagnostic failure is recorded, but every logical sample remains in
        # the metric population.  In particular, do not condition evaluation
        # membership on alignment.valid.
        history_alignment_valid = bool(alignment.valid[0])
        history_states = history_states[:1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            window = history_states.float().to(world_device)
            predicted_states = []
            for _ in range(10):
                next_state = predictor(window)
                predicted_states.append(next_state)
                window = torch.cat((window[:, 1:], next_state[:, None]), dim=1)
            rollout = torch.stack(predicted_states, dim=1).to(device)


        source_hw = tuple(int(value) for value in sparse["image_hw"])
        target_intrinsics_source = torch.as_tensor(sparse["intrinsics"])[0].float()
        target_global = torch.as_tensor(
            sparse["target_world_from_camera_cv"]
        ).double()
        target_depths_m = (
            torch.as_tensor(sparse["depth_cm"])[1:].float().div_(100.0)
        )
        target_available = torch.as_tensor(sparse["target_available"]).bool()
        target_intrinsics = _resize_intrinsics(
            target_intrinsics_source, source_hw
        )
        row: dict[str, Any] = {
            "segment": segment,
            "anchor_timestamp": anchor_timestamp,
            "history_timestamps_chronological": history_item["chronological_timestamps"],
            "history_offsets_seconds": [-0.6, -0.4, -0.2, 0.0],
            "history_timestamp_errors_us": history_item["timestamp_errors_us"],
            "history_source": history_item["source"],
            "vggt_forward_calls": vggt_calls,
            "camera_feature_aggregator_calls": camera_feature_aggregator_calls,
            "history_alignment_condition": alignment.condition[0],
            "history_alignment_valid": history_alignment_valid,
            "input_scale_mode": "anchor_lidar_paired",
            "paired_lidar_scale": float(paired_scale[0]),
            "paired_lidar_scale_valid": bool(paired_scale_valid[0]),
            "paired_lidar_scale_support": int(paired_scale_support[0]),
            "included_in_metrics": True,
            "horizons": {},
        }
        for target_index, seconds in enumerate(HORIZON_SECONDS):
            key = f"t+{seconds:.1f}"
            if not bool(target_available[target_index]):
                row["horizons"][key] = {"available": False}
                continue
            target_c2w = (anchor_from_world @ target_global[target_index]).float()
            target_depth = _resize_depth(target_depths_m[target_index])
            context = _context(
                history_encoding,
                target_c2w,
                target_intrinsics,
                TARGET_HW,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction_m = _decode_meters(
                    geometry_model, rollout[:, target_index], [context], device
                )[0].float()
            metric = _metrics(prediction_m, target_depth.to(device))
            row["horizons"][key] = metric
            # Source-faithful copy-last baseline at every horizon: always use
            # the final observed state S_t from this exact 12-image history
            # encoding, changing only the calibrated future query camera.  It
            # shares the rollout branch's Sim(3), scene scale, and LiDAR mask.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                copy_prediction_m = _decode_meters(
                    geometry_model, history_states[:, 3], [context], device
                )[0].float()
            copy_metric = _metrics(copy_prediction_m, target_depth.to(device))
            row["horizons"][key]["training_protocol_copy_last"] = copy_metric
            if predicted_query_c2w is not None:
                predicted_context = _context(
                    history_encoding,
                    predicted_query_c2w[target_index],
                    target_intrinsics,
                    TARGET_HW,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted_camera_prediction_m = _decode_meters(
                        geometry_model,
                        rollout[:, target_index],
                        [predicted_context],
                        device,
                    )[0].float()
                    predicted_camera_copy_m = _decode_meters(
                        geometry_model,
                        history_states[:, 3],
                        [predicted_context],
                        device,
                    )[0].float()
                predicted_camera_metric = _metrics(
                    predicted_camera_prediction_m, target_depth.to(device)
                )
                predicted_camera_copy_metric = _metrics(
                    predicted_camera_copy_m, target_depth.to(device)
                )
                gt_query_device = target_c2w.to(predicted_query_c2w.device)
                translation_error = float(
                    (
                        predicted_query_c2w[target_index, :3, 3]
                        - gt_query_device[:3, 3]
                    ).norm()
                )
                relative_rotation = (
                    gt_query_device[:3, :3].transpose(0, 1)
                    @ predicted_query_c2w[target_index, :3, :3]
                )
                rotation_error_deg = float(
                    rotation_angle(relative_rotation) * (180.0 / np.pi)
                )
                row["horizons"][key]["predicted_camera_rollout"] = (
                    predicted_camera_metric
                )
                row["horizons"][key]["predicted_camera_copy_last"] = (
                    predicted_camera_copy_metric
                )
                row["horizons"][key]["camera_error"] = {
                    "translation_m": translation_error,
                    "rotation_deg": rotation_error_deg,
                }
                for accumulator, camera_metric in (
                    (predicted_camera_totals, predicted_camera_metric),
                    (predicted_camera_copy_totals, predicted_camera_copy_metric),
                ):
                    for name in (
                        "abs_rel", "delta1", "abs_rel_sum", "delta1_sum",
                        "valid_pixels",
                    ):
                        accumulator[key][name] += float(camera_metric[name])
                        accumulator["overall"][name] += float(camera_metric[name])
                    accumulator[key]["frames"] += 1
                    accumulator["overall"]["frames"] += 1
                for error_key, error_value in (
                    ("translation_m", translation_error),
                    ("rotation_deg", rotation_error_deg),
                ):
                    camera_error_totals[key][error_key] += error_value
                    camera_error_totals["overall"][error_key] += error_value
                camera_error_totals[key]["frames"] += 1
                camera_error_totals["overall"]["frames"] += 1
            for name in (
                "abs_rel", "delta1", "abs_rel_sum", "delta1_sum", "valid_pixels"
            ):
                copy_totals[key][name] += float(copy_metric[name])
                copy_totals["overall"][name] += float(copy_metric[name])
            copy_totals[key]["frames"] += 1
            copy_totals["overall"]["frames"] += 1
            for name in ("abs_rel", "delta1", "abs_rel_sum", "delta1_sum", "valid_pixels"):
                totals[key][name] += float(metric[name])
                totals["overall"][name] += float(metric[name])
            totals[key]["frames"] += 1
            totals["overall"]["frames"] += 1
        row["seconds"] = time.perf_counter() - started
        rows.append(row)
        del sparse, history_item, images, history_output_single, history_output, history_states, history_encoding, rollout
        torch.cuda.empty_cache()
        print(json.dumps({
            "completed": len(rows), "total": len(segments), "segment": segment,
            "seconds": row["seconds"], "vggt_calls": vggt_calls,
        }), flush=True)

    summary: dict[str, dict[str, float | int]] = {}
    copy_summary: dict[str, dict[str, float | int]] = {}
    for key in [*(f"t+{value:.1f}" for value in HORIZON_SECONDS), "overall"]:
        frames = int(totals[key]["frames"])
        pixels = int(totals[key]["valid_pixels"])
        summary[key] = {
            "abs_rel": totals[key]["abs_rel"] / max(1, frames),
            "delta1": totals[key]["delta1"] / max(1, frames),
            "evaluated_frames": frames,
            "pixel_pooled_abs_rel": totals[key]["abs_rel_sum"] / max(1, pixels),
            "pixel_pooled_delta1": totals[key]["delta1_sum"] / max(1, pixels),
            "valid_pixels": pixels,
        }
        copy_frames = int(copy_totals[key]["frames"])
        copy_pixels = int(copy_totals[key]["valid_pixels"])
        copy_summary[key] = {
            "abs_rel": copy_totals[key]["abs_rel"] / max(1, copy_frames),
            "delta1": copy_totals[key]["delta1"] / max(1, copy_frames),
            "evaluated_frames": copy_frames,
            "pixel_pooled_abs_rel": copy_totals[key]["abs_rel_sum"] / max(1, copy_pixels),
            "pixel_pooled_delta1": copy_totals[key]["delta1_sum"] / max(1, copy_pixels),
            "valid_pixels": copy_pixels,
        }
    full_selection_requested = not args.only_segment and args.shard_count == 1
    coverage_complete = (
        full_selection_requested
        and len(rows) == selection_expected_count
        and all(
            summary[f"t+{seconds:.1f}"]["evaluated_frames"] == selection_expected_count
            for seconds in HORIZON_SECONDS
        )
    )
    complete = coverage_complete
    predicted_camera_summary = None
    predicted_camera_copy_summary = None
    camera_pose_error_summary = None
    camera_induced_degradation = None
    world_vs_copy_predicted_camera = None
    if camera_predictor is not None:
        predicted_camera_summary = {}
        predicted_camera_copy_summary = {}
        camera_pose_error_summary = {}
        camera_induced_degradation = {}
        world_vs_copy_predicted_camera = {}
        for key in [
            *(f"t+{value:.1f}" for value in HORIZON_SECONDS), "overall"
        ]:
            for accumulator, destination in (
                (predicted_camera_totals, predicted_camera_summary),
                (predicted_camera_copy_totals, predicted_camera_copy_summary),
            ):
                frames = int(accumulator[key]["frames"])
                pixels = int(accumulator[key]["valid_pixels"])
                destination[key] = {
                    "abs_rel": accumulator[key]["abs_rel"] / max(1, frames),
                    "delta1": accumulator[key]["delta1"] / max(1, frames),
                    "evaluated_frames": frames,
                    "pixel_pooled_abs_rel": (
                        accumulator[key]["abs_rel_sum"] / max(1, pixels)
                    ),
                    "pixel_pooled_delta1": (
                        accumulator[key]["delta1_sum"] / max(1, pixels)
                    ),
                    "valid_pixels": pixels,
                }
            error_frames = int(camera_error_totals[key]["frames"])
            camera_pose_error_summary[key] = {
                "translation_mean_m": (
                    camera_error_totals[key]["translation_m"]
                    / max(1, error_frames)
                ),
                "rotation_mean_deg": (
                    camera_error_totals[key]["rotation_deg"]
                    / max(1, error_frames)
                ),
                "evaluated_frames": error_frames,
            }
            rollout_gt = summary[key]
            rollout_predicted = predicted_camera_summary[key]
            copy_gt = copy_summary[key]
            copy_predicted = predicted_camera_copy_summary[key]
            camera_induced_degradation[key] = {
                "rollout_abs_rel_absolute": (
                    rollout_predicted["abs_rel"] - rollout_gt["abs_rel"]
                ),
                "rollout_delta1_absolute": (
                    rollout_predicted["delta1"] - rollout_gt["delta1"]
                ),
                "copy_last_abs_rel_absolute": (
                    copy_predicted["abs_rel"] - copy_gt["abs_rel"]
                ),
                "copy_last_delta1_absolute": (
                    copy_predicted["delta1"] - copy_gt["delta1"]
                ),
            }
            world_vs_copy_predicted_camera[key] = {
                "rollout_minus_copy_abs_rel": (
                    rollout_predicted["abs_rel"] - copy_predicted["abs_rel"]
                ),
                "rollout_minus_copy_delta1": (
                    rollout_predicted["delta1"] - copy_predicted["delta1"]
                ),
            }
    result = {
        "scene_checkpoint": str(Path(args.scene_checkpoint).resolve()),
        "world_checkpoint": str(Path(args.world_checkpoint).resolve()),
        "target_camera": "FRONT",
        "history_view_mode": "three_view",
        "gt_aligned_vggt_camera_geometry": False,
        "input_camera_conditioning_source": "vggt",
        "camera_checkpoint": str(Path(args.camera_checkpoint).resolve()),
        "camera_feature_kind": camera_feature_kind,
        "camera_predictor_config": (
            asdict(camera_config) if camera_config is not None else None
        ),
        "devices": {
            "geometry_decoder": str(device),
            "world_camera_predictors": str(world_device),
        },
        "residual_base": residual_base,
        "world_checkpoint_validation": world_validation,
        "predictor_config": asdict(predictor_config),
        "selection": {
            "manifest": str(Path(args.selection).resolve()),
            "population_count": selection["population_count"],
            "selected_count": len(segments),
            "seed": selection["seed"],
            "population_split": selection.get("population_split"),
            "purpose": selection.get("purpose", "paper_random100"),
        },
        "history_timestamp_audit": {
            "configured_tolerance_ms": args.timestamp_tolerance_ms,
            "max_abs_error_us": max(
                abs(int(error))
                for row in rows for error in row["history_timestamp_errors_us"]
            ),
            "segments_exceeding_20ms": sum(
                any(abs(int(error)) > 20_000 for error in row["history_timestamp_errors_us"])
                for row in rows
            ),
        },
        "protocol": {
            "history": "t-0.6,t-0.4,t-0.2,t; three cameras per state",
            "history_timestamp_tolerance_ms": args.timestamp_tolerance_ms,
            "physical_vggt_order": "t first as reference, then t-0.6,t-0.4,t-0.2",
            "scene_state_order": "chronological t-0.6,t-0.4,t-0.2,t",
            "vggt": (
                f"one {history_view_count}-image history-only forward; camera predictor consumes "
                "only its contextualized current-FRONT patch output"
                if camera_feature_kind == "joint_12view_current_front_vggt_spatial_6x8"
                else f"one {history_view_count}-image history-only scene forward plus one independent "
                "current-FRONT-only aggregator call for the camera predictor"
                if camera_feature_kind is not None
                else f"exactly one {history_view_count}-image history-only forward per segment"
            ),
            "input_camera_geometry": (
                "native VGGT world-point placement; explicit camera/ray conditioning uses "
                "vggt c2w; dataset-to-student Sim(3) retained"
            ),
            "rollout": "ten-step pure autoregressive predictor rollout; predicted states feed later steps",
            "future_information": (
                "historical metric FRONT poses and current-FRONT frozen VGGT patches "
                "predict future query cameras; GT future cameras are retained only as "
                "a paired diagnostic; no future RGB, LiDAR scale, or GT scene state "
                "enters prediction"
            ),
            "queries": "paired GT and predicted FRONT cameras at strict t+0.2 through t+2.0",
            "scale": (
                "matched anchor-t FRONT VGGT/LiDAR z-depth log-ratio median, then "
                f"single-anchor rotation/translation alignment and {history_view_count}-view scene scale; "
                "no future LiDAR and no decoder-output alignment"
            ),
            "gt": "all projected Waymo LiDAR z-depth points, 0<depth<=80m; no dynamic downweighting",
            "target_hw": list(TARGET_HW),
            "preprocessing": "balanced_512",
            "primary_aggregation": "arithmetic mean of per-frame metrics",
        },
        "segments_selected": len(rows),
        "segments_evaluated": len(rows),
        "complete": complete,
        "coverage_complete": coverage_complete,
        "exploratory_selection": False,
        "segments_alignment_invalid_but_included": sum(
            not bool(row["history_alignment_valid"]) for row in rows
        ),
        "segments_paired_lidar_scale_invalid_but_included": sum(
            not bool(row["paired_lidar_scale_valid"]) for row in rows
        ),
        "paired_lidar_scale": {
            "mean": float(np.mean([row["paired_lidar_scale"] for row in rows])),
            "median": float(np.median([row["paired_lidar_scale"] for row in rows])),
            "min": float(np.min([row["paired_lidar_scale"] for row in rows])),
            "max": float(np.max([row["paired_lidar_scale"] for row in rows])),
            "support_mean": float(np.mean([
                row["paired_lidar_scale_support"] for row in rows
            ])),
            "support_min": int(np.min([
                row["paired_lidar_scale_support"] for row in rows
            ])),
        },
        "training_protocol_copy_last_t_plus_0_2": copy_summary["t+0.2"],
        "copy_last_summary": copy_summary,
        "summary": summary,
        "predicted_camera_rollout_summary": predicted_camera_summary,
        "predicted_camera_copy_last_summary": predicted_camera_copy_summary,
        "camera_pose_error_summary": camera_pose_error_summary,
        "camera_induced_degradation": camera_induced_degradation,
        "world_vs_copy_predicted_camera": world_vs_copy_predicted_camera,
        "segments": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"summary": summary}, indent=2), flush=True)
    if full_selection_requested and not coverage_complete:
        raise RuntimeError(
            f"Incomplete Waymo paper evaluation; see {output} for per-horizon "
            "frame counts and segment details"
        )


if __name__ == "__main__":
    main()
