#!/usr/bin/env python3
"""KITTI zero-shot rollout using a VGGT-World-derived depth protocol.

The scene/world/camera checkpoints match the Waymo evaluation, while every
history state contains only KITTI image_02.  Evaluation follows VGGT-World's
KITTI source: frame-wise AbsRel/delta1 on valid projected LiDAR pixels followed
by an arithmetic frame mean.  No output-scale alignment is applied.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
sys.path.insert(0, str(SCRIPT.parent.parent))

from GeoState.data.common import _target_shape  # noqa: E402
from GeoState.data.kitti import (  # noqa: E402
    _oxts_pose,
    _read_calibration,
    _transform,
)
from GeoState.core.geometry import paired_depth_scale  # noqa: E402
from GeoState.evaluation.checkpoint import load_geostate_checkpoint as _load_model  # noqa: E402
from GeoState.models.world_model import (  # noqa: E402
    SceneStatePredictorConfig,
    build_next_scene_state_predictor,
)
from GeoState.core.types import VGGTOutput  # noqa: E402
from GeoState.evaluation.inference import decode_metric_depth as _decode_meters  # noqa: E402
from GeoState.models.camera_prediction import (  # noqa: E402
    rotation_angle,
)
from GeoState.models.camera_conditioning import (  # noqa: E402
    load_vggt_spatial_camera_checkpoint,
)


HISTORY_SECONDS = (-0.6, -0.4, -0.2, 0.0)
HORIZON_SECONDS = tuple(index / 5.0 for index in range(1, 11))
TOLERANCE_SECONDS = 0.020
TRAINING_INPUT_STATE_ORDER = (3, 0, 1, 2)


def _sequence_dirs(split_root: Path) -> list[Path]:
    return sorted(
        Path(path).parents[1]
        for path in glob.glob(
            str(split_root / "**" / "image_02" / "data"),
            recursive=True,
        )
        if Path(path).is_dir()
    )


def _camera_geometry(
    date_dir: Path,
    drive_dir: Path,
    frame_ids: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    camera = _read_calibration(date_dir / "calib_cam_to_cam.txt")
    velo = _read_calibration(date_dir / "calib_velo_to_cam.txt")
    imu = _read_calibration(date_dir / "calib_imu_to_velo.txt")
    projection = camera["P_rect_02"].reshape(3, 4)
    intrinsics = projection[:, :3]
    rectification = np.eye(4, dtype=np.float64)
    rectification[:3, :3] = camera["R_rect_00"].reshape(3, 3)
    camera2_from_rect0 = np.eye(4, dtype=np.float64)
    camera2_from_rect0[:3, 3] = np.linalg.solve(intrinsics, projection[:, 3])
    camera_from_imu = (
        camera2_from_rect0
        @ rectification
        @ _transform(velo["R"], velo["T"])
        @ _transform(imu["R"], imu["T"])
    )
    oxts_dir = drive_dir / "oxts" / "data"
    first_packet = np.loadtxt(min(oxts_dir.glob("*.txt")), dtype=np.float64)
    scale = math.cos(first_packet[0] * math.pi / 180.0)
    global_from_imu0 = _oxts_pose(first_packet, scale)
    poses = []
    for frame_id in frame_ids:
        packet = np.loadtxt(oxts_dir / f"{frame_id}.txt", dtype=np.float64)
        global_from_imu = _oxts_pose(packet, scale)
        poses.append(
            np.linalg.inv(global_from_imu0)
            @ global_from_imu
            @ np.linalg.inv(camera_from_imu)
        )
    return (
        torch.from_numpy(np.stack(poses).astype(np.float32)),
        torch.from_numpy(intrinsics.astype(np.float32)),
    )


def _parse_timestamp(line: str) -> float:
    # KITTI timestamps have nanosecond text precision; datetime gives enough
    # precision for the requested 20-ms gate.
    value = line.strip()
    if not value:
        raise ValueError("empty KITTI timestamp")
    if len(value.split(".")[-1]) > 6:
        value = value[: value.index(".") + 7]
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f").timestamp()


def _select_frames(
    drive_dir: Path,
    anchor_frame: int,
) -> dict[str, Any]:
    rgb_dir = drive_dir / "image_02" / "data"
    available = sorted(int(path.stem) for path in rgb_dir.glob("*.png"))
    if not available:
        raise ValueError("no image_02 RGB frames")
    lines = (drive_dir / "image_02" / "timestamps.txt").read_text().splitlines()
    times = {index: _parse_timestamp(line) for index, line in enumerate(lines) if line.strip()}
    available = [frame for frame in available if frame in times]
    first = available[0]
    # VGGT-World's KITTI source defines logical 5-Hz states by a raw-frame
    # stride of two, not by searching for exact wall-clock multiples of 0.2 s.
    # KITTI runs near 9.7 Hz, so exact-time gating would incorrectly reject all
    # normal drives by the later horizons.  Gate instead against the drive's
    # measured median raw-frame period; this detects actual gaps/outliers while
    # preserving the source-faithful frame indices.
    anchor_frame = int(anchor_frame)
    anchor_selection = "explicit_audit_override"
    history = [anchor_frame - 6, anchor_frame - 4, anchor_frame - 2, anchor_frame]
    targets = [anchor_frame + 2 * index for index in range(1, 11)]
    chosen = [*history, *targets]
    missing = sorted(set(chosen) - set(available))
    if missing:
        raise ValueError(f"fixed VGGT-World stride frames missing: {missing}")
    adjacent = [times[right] - times[left] for left, right in zip(available, available[1:]) if right == left + 1]
    if not adjacent:
        raise ValueError("cannot estimate KITTI frame period")
    frame_period = float(np.median(adjacent))
    expected = [times[anchor_frame] + (frame - anchor_frame) * frame_period for frame in chosen]
    errors = [times[frame] - target for frame, target in zip(chosen, expected)]
    if max(map(abs, errors)) > TOLERANCE_SECONDS:
        raise ValueError(
            f"fixed-stride timestamp residual {max(map(abs, errors)) * 1000:.3f} ms "
            f"exceeds {TOLERANCE_SECONDS * 1000:.3f} ms"
        )
    return {
        "first_frame": first,
        "anchor_frame": anchor_frame,
        "anchor_seed": None,
        "anchor_selection": anchor_selection,
        "raw_frame_period_ms": frame_period * 1000.0,
        "history_frames": history,
        "target_frames": targets,
        "history_errors_ms": [value * 1000.0 for value in errors[:4]],
        "target_errors_ms": [value * 1000.0 for value in errors[4:]],
    }


def _pixel_transform(
    original_hw: tuple[int, int],
) -> tuple[torch.Tensor, dict[str, Any]]:
    height, width = original_hw
    # Match preprocess_frame(): clamp extreme aspect ratios to [0.5, 2.0]
    # with a centered source-space crop, then use balanced-512 patch count.
    if height / width < 0.5:
        crop_width = max(1, round(height / 0.5))
        crop_height = height
        source_left = (width - crop_width) // 2
        source_top = 0
    elif height / width > 2.0:
        crop_width = width
        crop_height = max(1, round(width * 2.0))
        source_left = 0
        source_top = (height - crop_height) // 2
    else:
        crop_width, crop_height = width, height
        source_left = source_top = 0
    target_height, target_width = _target_shape(
        crop_height, crop_width, 512, 16
    )
    scale_x, scale_y = target_width / crop_width, target_height / crop_height
    transform = torch.tensor([
        [scale_x, 0.0, -source_left * scale_x],
        [0.0, scale_y, -source_top * scale_y],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    return transform, {
        "mode": "balanced_512",
        "source_crop": (source_left, source_top, crop_width, crop_height),
        "target_hw": (target_height, target_width),
    }


def _load_rgb(path: Path, geometry: dict[str, Any]) -> torch.Tensor:
    height, width = geometry["target_hw"]
    with Image.open(path) as source:
        image = source.convert("RGB")
        left, top, crop_width, crop_height = geometry["source_crop"]
        image = image.crop((left, top, left + crop_width, top + crop_height))
        image = image.resize((width, height), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_depth(path: Path, geometry: dict[str, Any]) -> torch.Tensor:
    height, width = geometry["target_hw"]
    with Image.open(path) as source:
        encoded = np.asarray(source, dtype=np.uint16)
    if encoded.max(initial=0) <= 255:
        raise ValueError(f"not KITTI 1/256-m uint16: {path}")
    depth = encoded.astype(np.float32) / 256.0
    depth[encoded == 0] = -1.0
    image = Image.fromarray(depth)
    left, top, crop_width, crop_height = geometry["source_crop"]
    image = image.crop((left, top, left + crop_width, top + crop_height))
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy())


def _vggt_world_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float | int]:
    valid = torch.isfinite(target) & (target > 0) & (target <= 80.0)
    pixels = int(valid.sum())
    if not pixels:
        raise ValueError("no valid GT pixels")
    if not bool((torch.isfinite(prediction) & (prediction > 0))[valid].all()):
        raise ValueError("non-finite or non-positive prediction on valid GT pixels")
    scale = prediction.new_tensor(1.0)
    aligned = prediction * scale
    relative = (aligned[valid] - target[valid]).abs() / target[valid]
    ratio = torch.maximum(aligned[valid] / target[valid], target[valid] / aligned[valid])
    return {
        "abs_rel": float(relative.mean()),
        "delta1": float((ratio < 1.25).float().mean()),
        "metric_scale": float(scale),
        "valid_pixels": pixels,
        "abs_rel_sum": float(relative.sum()),
        "delta1_sum": float((ratio < 1.25).sum()),
    }


def _repeat_output(output: VGGTOutput, repeats: int = 2) -> VGGTOutput:
    def repeat(value: torch.Tensor) -> torch.Tensor:
        return value.expand(repeats, *value.shape[1:])
    return VGGTOutput(
        patch_features=[repeat(value) for value in output.patch_features],
        depth=repeat(output.depth), confidence=repeat(output.confidence),
        world_points=repeat(output.world_points), c2w=repeat(output.c2w),
        intrinsics=repeat(output.intrinsics), patch_size=output.patch_size,
        patch_hw=output.patch_hw,
    )


def _context(encoding: Any, pose: torch.Tensor, intrinsics: torch.Tensor, target_hw: tuple[int, int]) -> dict[str, Any]:
    return {
        "reference_c2w_student": encoding.reference_c2w_student[0].float().cpu(),
        "dataset_to_student_scale": encoding.dataset_to_student.scale[0].float().cpu(),
        "dataset_to_student_rotation": encoding.dataset_to_student.rotation[0].float().cpu(),
        "dataset_to_student_translation": encoding.dataset_to_student.translation[0].float().cpu(),
        "scene_scale": encoding.scene_scale[0].float().cpu(),
        "target_c2w": pose.float().cpu(),
        "target_intrinsics": intrinsics.float().cpu(),
        "target_hw": target_hw,
    }


def _accumulate(total: dict[str, dict[str, float]], key: str, metric: dict[str, float | int]) -> None:
    for name in ("abs_rel", "delta1", "abs_rel_sum", "delta1_sum", "valid_pixels"):
        total[key][name] += float(metric[name])
        total["overall"][name] += float(metric[name])
    total[key]["frames"] += 1
    total["overall"]["frames"] += 1


def _summarize(total: dict[str, dict[str, float]]) -> dict[str, dict[str, float | int]]:
    result = {}
    for key in [*(f"t+{value:.1f}" for value in HORIZON_SECONDS), "overall"]:
        frames = int(total[key]["frames"])
        pixels = int(total[key]["valid_pixels"])
        result[key] = {
            "abs_rel": total[key]["abs_rel"] / max(1, frames),
            "delta1": total[key]["delta1"] / max(1, frames),
            "evaluated_frames": frames,
            "pixel_pooled_abs_rel": total[key]["abs_rel_sum"] / max(1, pixels),
            "pixel_pooled_delta1": total[key]["delta1_sum"] / max(1, pixels),
            "valid_pixels": pixels,
        }
    return result


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene-checkpoint", required=True)
    parser.add_argument("--world-checkpoint", required=True)
    parser.add_argument("--camera-checkpoint", required=True)
    parser.add_argument("--kitti-root", required=True)
    parser.add_argument(
        "--manifest", required=True,
        help="Fixed 113-sequence paper manifest with exact history and target frames.",
    )
    parser.add_argument(
        "--depth-annotated-root", required=True,
        help="KITTI depth-completion annotated root containing train/ and val/ drive folders.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--world-device", help="Optional separate GPU for world and camera predictors.")
    parser.add_argument("--only-sequence", help="Diagnostic subset; not a paper-table result.")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if tuple(float(value) for value in manifest["history_seconds"]) != HISTORY_SECONDS:
        raise ValueError("KITTI manifest history does not match the paper protocol")
    if tuple(float(value) for value in manifest["horizon_seconds"]) != HORIZON_SECONDS:
        raise ValueError("KITTI manifest horizons do not match the paper protocol")
    manifest_rows = list(manifest["sequences"])
    manifest_names = [str(row["sequence"]) for row in manifest_rows]
    if (
        int(manifest.get("raw_population_count", 0)) != 151
        or int(manifest.get("selected_count", 0)) != 113
        or len(manifest_names) != 113
        or len(set(manifest_names)) != 113
        or manifest_names != sorted(manifest_names)
    ):
        raise ValueError("invalid fixed KITTI 151-to-113 paper manifest")
    manifest_by_name = {
        str(row["sequence"]): row for row in manifest_rows
    }

    device = torch.device(args.device)
    world_device = torch.device(args.world_device or args.device)
    _, _, model = _load_model(args.config, Path(args.scene_checkpoint), device)
    model.eval().requires_grad_(False)
    world_payload = torch.load(args.world_checkpoint, map_location="cpu", mmap=True, weights_only=True)
    world_config = SceneStatePredictorConfig(**world_payload["config"])
    predictor = build_next_scene_state_predictor(world_config).to(world_device)
    predictor.load_state_dict(world_payload["model"], strict=True)
    predictor.eval().requires_grad_(False)
    residual_base = str(world_payload.get("residual_base", "previous"))
    if residual_base != "previous":
        raise ValueError("The paper protocol requires residual_base='previous'")
    del world_payload

    camera_payload = torch.load(args.camera_checkpoint, map_location="cpu", weights_only=True)
    if camera_payload.get("format") != "vggt_spatial_camera_predictor_v1":
        raise ValueError("The paper protocol requires a VGGT-spatial camera predictor")
    if camera_payload.get("camera_patch_context") == "joint_12view_current_front_output":
        raise ValueError("The paper camera predictor requires an independent current-image VGGT pass")
    camera, camera_payload, camera_config = load_vggt_spatial_camera_checkpoint(
        args.camera_checkpoint,
        world_device,
    )
    camera_feature_kind = "independent"
    camera.eval().requires_grad_(False)
    del camera_payload

    root = Path(args.kitti_root)
    split_by_sequence: dict[str, str] = {}
    all_sequence_dirs = []
    for split in ("train", "val"):
        for path in _sequence_dirs(root / split):
            name = f"{path.parent.name}/{path.name}"
            if name in split_by_sequence:
                raise ValueError(f"duplicate train/val sequence {name}")
            split_by_sequence[name] = split
            all_sequence_dirs.append(path)
    all_sequence_dirs.sort(key=lambda path: f"{path.parent.name}/{path.name}")
    sequence_by_name = {
        f"{path.parent.name}/{path.name}": path for path in all_sequence_dirs
    }
    missing_manifest_sequences = sorted(set(manifest_names) - set(sequence_by_name))
    if missing_manifest_sequences:
        raise ValueError(
            "KITTI paper-manifest sequences are missing from the dataset: "
            f"{missing_manifest_sequences[:3]}"
        )
    sequence_dirs = [sequence_by_name[name] for name in manifest_names]
    raw_population_count = len(all_sequence_dirs)
    population_count = len(sequence_dirs)
    selected_sequence_names = [
        f"{path.parent.name}/{path.name}" for path in sequence_dirs
    ]
    if raw_population_count != int(manifest["raw_population_count"]):
        raise ValueError(
            f"KITTI population mismatch: found {raw_population_count}, "
            f"expected {manifest['raw_population_count']}"
        )
    if selected_sequence_names != manifest_names:
        missing = sorted(set(manifest_names) - set(selected_sequence_names))
        unexpected = sorted(set(selected_sequence_names) - set(manifest_names))
        raise ValueError(
            "KITTI data selection differs from the paper manifest: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    if args.only_sequence:
        if args.only_sequence not in sequence_by_name or args.only_sequence not in manifest_by_name:
            raise ValueError("--only-sequence is not in the fixed paper manifest")
        sequence_dirs = [sequence_by_name[args.only_sequence]]
    totals = {name: defaultdict(lambda: defaultdict(float)) for name in (
        "world_gt_camera", "copy_gt_camera", "world_pred_camera", "copy_pred_camera"
    )}
    camera_totals = defaultdict(lambda: defaultdict(float))
    rows, skipped = [], []
    observed_target_hws: set[tuple[int, int]] = set()
    for sequence_index, drive_dir in enumerate(sequence_dirs):
        started = time.perf_counter()
        date_dir = drive_dir.parent
        try:
            sequence_name = f"{date_dir.name}/{drive_dir.name}"
            sequence_split = split_by_sequence[sequence_name]
            gt_dir = (
                Path(args.depth_annotated_root) / sequence_split / drive_dir.name
                / "proj_depth" / "groundtruth" / "image_02"
            )
            expected_selection = manifest_by_name[sequence_name]
            selection = _select_frames(
                drive_dir,
                anchor_frame=int(expected_selection["anchor_frame"]),
            )
            actual_selection = {
                "source_split": sequence_split,
                "first_frame": int(selection["first_frame"]),
                "anchor_frame": int(selection["anchor_frame"]),
                "history_frames": [int(value) for value in selection["history_frames"]],
                "target_frames": [int(value) for value in selection["target_frames"]],
            }
            expected_selection_fields = {
                key: expected_selection[key] for key in actual_selection
            }
            if actual_selection != expected_selection_fields:
                raise ValueError(
                    f"frame selection differs from paper manifest for {sequence_name}"
                )
            history_frames = selection["history_frames"]
            target_frames = selection["target_frames"]
            all_frames = history_frames + target_frames
            frame_ids = [f"{value:010d}" for value in all_frames]
            missing = [value for value in frame_ids if not (gt_dir / f"{value}.png").is_file()]
            if missing:
                raise ValueError(f"missing GT depth frames: {missing}")
            poses_global, original_intrinsics = _camera_geometry(date_dir, drive_dir, frame_ids)
            with Image.open(drive_dir / "image_02" / "data" / f"{frame_ids[3]}.png") as image:
                original_hw = (image.height, image.width)
            pixel_transform, geometry = _pixel_transform(original_hw)
            output_hw = tuple(int(value) for value in geometry["target_hw"])
            observed_target_hws.add(output_hw)
            intrinsics = (pixel_transform @ original_intrinsics).to(device)
            # VGGT reference and camera predictor anchor are physical view zero = t.
            ordered_history = [history_frames[index] for index in TRAINING_INPUT_STATE_ORDER]
            images = torch.stack([
                _load_rgb(drive_dir / "image_02" / "data" / f"{frame:010d}.png", geometry)
                for frame in ordered_history
            ]).to(device)
            global_by_frame = {frame: pose for frame, pose in zip(all_frames, poses_global)}
            ordered_global = torch.stack([global_by_frame[frame] for frame in ordered_history]).double()
            anchor_from_global = torch.linalg.inv(ordered_global[0])
            input_c2w = (anchor_from_global[None] @ ordered_global).float().to(device)
            chronological_global = torch.stack([global_by_frame[frame] for frame in history_frames]).float().to(world_device)
            target_global = torch.stack([global_by_frame[frame] for frame in target_frames]).double()
            target_c2w = (anchor_from_global[None] @ target_global).float().to(device)
            anchor_depth = _load_depth(gt_dir / f"{history_frames[-1]:010d}.png", geometry).to(device)

            state_ids = torch.tensor(TRAINING_INPUT_STATE_ORDER, device=device)[None]
            state_mask = torch.stack([(state_ids == state) for state in range(4)], dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_single = model.vggt(images[None])
                paired_scale, paired_valid, paired_support = paired_depth_scale(
                    output_single.depth[:, 0], anchor_depth[None], output_single.confidence[:, 0]
                )
                if not bool(paired_valid[0]):
                    raise ValueError("invalid anchor VGGT/LiDAR paired scale")
                output = _repeat_output(output_single)
                alignment_mask = torch.zeros((2, 4), dtype=torch.bool, device=device)
                alignment_mask[:, 0] = True
                states, encoding = model.encode_temporal_scene_states_from_vggt_output(
                    output,
                    input_c2w[None].expand(2, -1, -1, -1),
                    input_intrinsics=intrinsics[None, None].expand(2, 4, -1, -1),
                    state_view_mask=state_mask.expand(2, -1, -1),
                    single_view_scale=paired_scale.expand(2),
                    alignment_view_mask=alignment_mask,
                    scene_scale_view_mask=torch.ones((2, 4), dtype=torch.bool, device=device),
                    frame_ids=[[f"{value:010d}" for value in ordered_history]] * 2,
                    reference_index=0,
                    input_camera_conditioning_source="vggt",
                )
            states = states[:1]
            window = states.float().to(world_device)
            predicted_states = []
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in HORIZON_SECONDS:
                    predicted = predictor(window)
                    predicted_states.append(predicted)
                    window = torch.cat((window[:, 1:], predicted[:, None]), dim=1)
            rollout = torch.stack(predicted_states, dim=1).to(device)


            future_seconds = torch.tensor(HORIZON_SECONDS, device=world_device, dtype=torch.float32)
            aggregated, patch_start = model.vggt.model.aggregator(images[None, :1])
            patch = aggregated[-1][:, 0, patch_start:].float()
            patch_h, patch_w = images.shape[-2] // 16, images.shape[-1] // 16
            patch = patch.reshape(1, patch_h, patch_w, -1).permute(0, 3, 1, 2)
            features = F.adaptive_avg_pool2d(patch, (6, 8)).flatten(2).transpose(1, 2).to(world_device)
            with torch.autocast("cuda", enabled=False):
                predicted_c2w = camera(
                    chronological_global[None].float(), future_seconds.float(),
                    current_features=features.float(),
                )["predicted_relative"][0].to(device)

            row = {
                "sequence_index": sequence_index,
                "sequence": f"{date_dir.name}/{drive_dir.name}",
                "source_split": sequence_split,
                **selection,
                "paired_scale": float(paired_scale[0]),
                "paired_support": int(paired_support[0]),
                "target_hw": list(output_hw),
                "horizons": {},
            }
            for target_index, seconds in enumerate(HORIZON_SECONDS):
                key = f"t+{seconds:.1f}"
                target_depth = _load_depth(gt_dir / f"{target_frames[target_index]:010d}.png", geometry).to(device)
                gt_context = _context(encoding, target_c2w[target_index], intrinsics, output_hw)
                pred_context = _context(encoding, predicted_c2w[target_index], intrinsics, output_hw)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predictions = {
                        "world_gt_camera": _decode_meters(model, rollout[:, target_index], [gt_context], device)[0].float(),
                        "copy_gt_camera": _decode_meters(model, states[:, 3], [gt_context], device)[0].float(),
                        "world_pred_camera": _decode_meters(model, rollout[:, target_index], [pred_context], device)[0].float(),
                        "copy_pred_camera": _decode_meters(model, states[:, 3], [pred_context], device)[0].float(),
                    }
                metrics = {
                    name: _vggt_world_metrics(
                        value,
                        target_depth,
                    )
                    for name, value in predictions.items()
                }
                gt_pose = target_c2w[target_index]
                translation = float((predicted_c2w[target_index, :3, 3] - gt_pose[:3, 3]).norm())
                relative_rotation = gt_pose[:3, :3].T @ predicted_c2w[target_index, :3, :3]
                rotation = float(rotation_angle(relative_rotation) * 180.0 / math.pi)
                row["horizons"][key] = {
                    "target_frame": target_frames[target_index],
                    "metrics": metrics,
                    "camera_error": {"translation_m": translation, "rotation_deg": rotation},
                }
                for name, metric in metrics.items():
                    _accumulate(totals[name], key, metric)
                for name, value in (("translation_m", translation), ("rotation_deg", rotation)):
                    camera_totals[key][name] += value
                    camera_totals["overall"][name] += value
                camera_totals[key]["frames"] += 1
                camera_totals["overall"]["frames"] += 1
            row["seconds"] = time.perf_counter() - started
            rows.append(row)
            print(json.dumps({"completed": len(rows), "skipped": len(skipped), "sequence": row["sequence"]}), flush=True)
        except Exception as error:
            skipped.append({
                "sequence": f"{date_dir.name}/{drive_dir.name}",
                "reason": str(error),
            })
            print(json.dumps({"skipped": skipped[-1]}), flush=True)
        torch.cuda.empty_cache()

    summaries = {name: _summarize(total) for name, total in totals.items()}
    camera_summary = {}
    for key in [*(f"t+{value:.1f}" for value in HORIZON_SECONDS), "overall"]:
        frames = int(camera_totals[key]["frames"])
        camera_summary[key] = {
            "translation_mean_m": camera_totals[key]["translation_m"] / max(1, frames),
            "rotation_mean_deg": camera_totals[key]["rotation_deg"] / max(1, frames),
            "evaluated_frames": frames,
        }
    result = {
        "scene_checkpoint": str(Path(args.scene_checkpoint).resolve()),
        "world_checkpoint": str(Path(args.world_checkpoint).resolve()),
        "camera_checkpoint": str(Path(args.camera_checkpoint).resolve()),
        "world_config": asdict(world_config),
        "camera_config": asdict(camera_config),
        "protocol": {
            "manifest": str(Path(args.manifest).resolve()),
            "split": "fixed eligible drives from the combined KITTI train+val population",
            "sequence_population": "train_val",
            "raw_population_count": raw_population_count,
            "population_count": population_count,
            "complete_tenth_window_required": True,
            "sequence_sample_size": population_count,
            "sequence_seed": None,
            "selected_sequences": [
                f"{path.parent.name}/{path.name}" for path in sequence_dirs
            ],
            "history": "t-0.6,t-0.4,t-0.2,t; KITTI image_02 single view per state",
            "input_camera_geometry": (
                "native VGGT point placement; explicit camera/ray conditioning uses "
                "vggt c2w; dataset-to-student Sim(3) retained"
            ),
            "anchor": "fixed tenth logical state: raw-frame offset 18 from first RGB",
            "anchor_seed": None,
            "fixed_anchor_offset": 18,
            "anchor_sampling": "fixed tenth logical state",
            "rollout": "ten-step pure autoregressive rollout to t+2.0s",
            "time_gate_ms": 20.0,
            "time_gate_definition": "selected raw-stride timestamps versus anchor plus raw-frame offset times per-drive median frame period",
            "gate_failure_policy": "skip entire sequence and record reason",
            "target_hw": (
                list(next(iter(observed_target_hws)))
                if len(observed_target_hws) == 1
                else [list(value) for value in sorted(observed_target_hws)]
            ),
            "preprocessing": "training-source balanced-512: centered source crop to supported aspect ratio, then balanced patch-budget resize",
            "scene_gauge": "one anchor-t matched-pixel VGGT/LiDAR scale; no future GT enters model",
            "metric": "decoder metric depth evaluated directly on valid KITTI LiDAR pixels; no output or per-target-frame scale alignment",
            "metric_scale_alignment": False,
            "aggregation": "arithmetic mean of per-frame AbsRel/delta1",
            "dynamic_policy": "all valid projected LiDAR pixels; no downweighting",
        },
        "sequences_discovered": len(sequence_dirs),
        "input_camera_conditioning_source": "vggt",
        "sequences_evaluated": len(rows),
        "sequences_skipped": len(skipped),
        "complete": not args.only_sequence and len(rows) == 113 and not skipped,
        "diagnostic_subset": bool(args.only_sequence),
        "skipped": skipped,
        "summaries": summaries,
        "camera_pose_error_summary": camera_summary,
        "sequences": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"evaluated": len(rows), "skipped": len(skipped), "overall": {name: value["overall"] for name, value in summaries.items()}}, indent=2), flush=True)
    if skipped or len(rows) != len(sequence_dirs):
        raise RuntimeError(
            f"Incomplete KITTI paper evaluation: {len(rows)}/{len(sequence_dirs)} "
            f"sequences evaluated; see {output} for skip reasons"
        )


if __name__ == "__main__":
    main()
