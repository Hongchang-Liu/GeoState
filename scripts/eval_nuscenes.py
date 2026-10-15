#!/usr/bin/env python3
"""nuScenes official-val zero-shot world rollout with sparse LiDAR depth GT."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
sys.path.insert(0, str(SCRIPT.parent.parent))

from eval_kitti import (  # noqa: E402
    _accumulate, _context, _load_rgb, _pixel_transform, _repeat_output,
    _vggt_world_metrics,
)
from GeoState.models.camera_prediction import (  # noqa: E402
    rotation_angle,
)
from GeoState.models.camera_conditioning import (  # noqa: E402
    load_vggt_spatial_camera_checkpoint,
)
from GeoState.core.geometry import paired_depth_scale  # noqa: E402
from GeoState.evaluation.checkpoint import load_geostate_checkpoint as _load_model  # noqa: E402
from GeoState.models.world_model import (  # noqa: E402
    SceneStatePredictorConfig, build_next_scene_state_predictor,
)
from GeoState.evaluation.inference import decode_metric_depth as _decode_meters  # noqa: E402
from GeoState.data.nuscenes import (  # noqa: E402
    global_from_sensor,
    project_lidar_depth,
)


ROLLOUT_STEP_SECONDS = 0.25
TRAINING_INPUT_STATE_ORDER = (3, 0, 1, 2)


def _horizon_key(seconds: float) -> str:
    return f"t+{seconds:g}"


def _summarize_dynamic(
    total: dict[str, dict[str, float]], horizons: tuple[float, ...],
) -> dict[str, dict[str, float | int]]:
    result = {}
    for key in [*(_horizon_key(value) for value in horizons), "overall"]:
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


def _state_at_seconds(
    anchor_state: torch.Tensor,
    rollout: torch.Tensor,
    seconds: float,
    rollout_step_seconds: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    position = float(seconds) / rollout_step_seconds
    lower_step = int(math.floor(position + 1e-8))
    upper_step = int(math.ceil(position - 1e-8))
    alpha = position - lower_step

    def state(step: int) -> torch.Tensor:
        return anchor_state if step == 0 else rollout[:, step - 1]

    lower = state(lower_step)
    upper = state(upper_step)
    interpolated = lower if upper_step == lower_step else lower.lerp(upper, alpha)
    return interpolated, {
        "lower_rollout_step": lower_step,
        "upper_rollout_step": upper_step,
        "interpolation_alpha": float(alpha),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene-checkpoint", required=True)
    parser.add_argument("--world-checkpoint", required=True)
    parser.add_argument("--camera-checkpoint", required=True)
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--world-device", help="Optional separate GPU for world and camera predictors.")
    parser.add_argument("--only-scene", help="Diagnostic subset; not a paper-table result.")
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    table_root = dataroot / "v1.0-trainval"
    manifest = json.loads(Path(args.manifest).read_text())
    history_seconds = tuple(float(value) for value in manifest["history_seconds"])
    horizon_seconds = tuple(float(value) for value in manifest["horizon_seconds"])
    if history_seconds != (-0.75, -0.5, -0.25, 0.0):
        raise ValueError(f"unexpected paper history timestamps: {history_seconds}")
    if horizon_seconds != (0.75, 1.25, 2.5):
        raise ValueError(f"unexpected paper forecast horizons: {horizon_seconds}")
    if int(manifest.get("official_scene_count", 0)) != 150:
        raise ValueError("paper manifest must declare all 150 official val scenes")
    manifest_scene_names = [str(row["scene_name"]) for row in manifest["scenes"]]
    if len(manifest_scene_names) != 150 or len(set(manifest_scene_names)) != 150:
        raise ValueError("paper manifest must contain 150 unique nuScenes scenes")
    rollout_steps = 10
    rows = manifest["scenes"]
    if args.only_scene:
        rows = [item for item in rows if item["scene_name"] == args.only_scene]
        if len(rows) != 1:
            raise ValueError("--only-scene is not in the fixed paper manifest")
    calibrations = {
        record["token"]: record for record in json.loads(
            (table_root / "calibrated_sensor.json").read_text()
        )
    }
    ego_poses = {
        record["token"]: record for record in json.loads(
            (table_root / "ego_pose.json").read_text()
        )
    }

    device = torch.device(args.device)
    world_device = torch.device(args.world_device or args.device)
    _, _, model = _load_model(args.config, Path(args.scene_checkpoint), device)
    model.eval().requires_grad_(False)
    payload = torch.load(args.world_checkpoint, map_location="cpu", mmap=True, weights_only=True)
    world_config = SceneStatePredictorConfig(**payload["config"])
    predictor = build_next_scene_state_predictor(world_config).to(world_device)
    predictor.load_state_dict(payload["model"], strict=True)
    predictor.eval().requires_grad_(False)
    residual_base = str(payload.get("residual_base", "previous"))
    if residual_base != "previous":
        raise ValueError("The paper protocol requires residual_base='previous'")
    del payload

    camera_payload = torch.load(args.camera_checkpoint, map_location="cpu", weights_only=True)
    if camera_payload.get("format") != "vggt_spatial_camera_predictor_v1":
        raise ValueError("The paper protocol requires a VGGT-spatial camera predictor")
    if camera_payload.get("camera_patch_context") == "joint_12view_current_front_output":
        raise ValueError("The paper camera predictor requires an independent current-image VGGT pass")
    camera, camera_payload, camera_config = load_vggt_spatial_camera_checkpoint(
        args.camera_checkpoint,
        world_device,
    )
    camera.config = replace(camera_config, step_seconds=ROLLOUT_STEP_SECONDS)
    camera.pose_model.config = replace(
        camera.pose_model.config, step_seconds=ROLLOUT_STEP_SECONDS
    )
    camera.eval().requires_grad_(False)
    del camera_payload

    totals = {
        name: defaultdict(lambda: defaultdict(float))
        for name in (
            "world_gt_camera", "copy_gt_camera",
            "world_pred_camera", "copy_pred_camera",
        )
    }
    camera_totals = defaultdict(lambda: defaultdict(float))
    completed, skipped = [], []
    observed_hws: set[tuple[int, int]] = set()
    for scene_index, item in enumerate(rows):
        started = time.perf_counter()
        try:
            cameras = item["camera"]
            lidars = item["lidar"]
            history_cameras, target_cameras = cameras[:4], cameras[4:]
            history_lidars, target_lidars = lidars[:4], lidars[4:]
            required_records = (*cameras, history_lidars[-1], *target_lidars)
            missing = [
                record["filename"] for record in required_records
                if not (dataroot / record["filename"]).is_file()
            ]
            if missing:
                raise FileNotFoundError(f"missing {len(missing)} sensor files; first={missing[0]}")
            with Image.open(dataroot / history_cameras[-1]["filename"]) as source:
                original_hw = (source.height, source.width)
            pixel_transform, geometry = _pixel_transform(original_hw)
            target_hw = tuple(int(value) for value in geometry["target_hw"])
            observed_hws.add(target_hw)
            intrinsics_original = torch.tensor(
                calibrations[history_cameras[-1]["calibrated_sensor_token"]]["camera_intrinsic"],
                dtype=torch.float32,
            )
            intrinsics = (pixel_transform @ intrinsics_original).to(device)
            ordered_history = [history_cameras[index] for index in TRAINING_INPUT_STATE_ORDER]
            images = torch.stack([
                _load_rgb(dataroot / record["filename"], geometry)
                for record in ordered_history
            ]).to(device)
            global_by_token = {
                record["token"]: torch.from_numpy(
                    global_from_sensor(record, calibrations, ego_poses)
                ).double()
                for record in cameras
            }
            ordered_global = torch.stack([global_by_token[record["token"]] for record in ordered_history])
            anchor_from_global = torch.linalg.inv(ordered_global[0])
            input_c2w = (anchor_from_global[None] @ ordered_global).float().to(device)
            chronological_global = torch.stack([
                global_by_token[record["token"]] for record in history_cameras
            ]).float().to(world_device)
            target_global = torch.stack([global_by_token[record["token"]] for record in target_cameras])
            target_c2w = (anchor_from_global[None] @ target_global).float().to(device)
            anchor_depth = project_lidar_depth(
                dataroot, history_lidars[-1], history_cameras[-1], calibrations,
                ego_poses, pixel_transform, target_hw,
            ).to(device)

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
                    frame_ids=[[record["token"] for record in ordered_history]] * 2,
                    reference_index=0,
                    input_camera_conditioning_source="vggt",
                )
            states = states[:1]
            anchor_state = states[:, -1]
            window = states.float().to(world_device)
            predicted_states = []
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(rollout_steps):
                    predicted = predictor(window)
                    predicted_states.append(predicted)
                    window = torch.cat((window[:, 1:], predicted[:, None]), dim=1)
            rollout = torch.stack(predicted_states, dim=1).to(device)
            query_states, state_time_details = zip(*[
                _state_at_seconds(
                    anchor_state, rollout, seconds, ROLLOUT_STEP_SECONDS
                )
                for seconds in horizon_seconds
            ])
            query_states = torch.stack(query_states, dim=1)


            aggregated, patch_start = model.vggt.model.aggregator(images[None, :1])
            patch = aggregated[-1][:, 0, patch_start:].float()
            patch_h, patch_w = images.shape[-2] // 16, images.shape[-1] // 16
            patch = patch.reshape(1, patch_h, patch_w, -1).permute(0, 3, 1, 2)
            current_features = (
                F.adaptive_avg_pool2d(patch, (6, 8)).flatten(2).transpose(1, 2).to(world_device)
            )

            camera_positions = [seconds / ROLLOUT_STEP_SECONDS for seconds in horizon_seconds]
            if any(abs(position - round(position)) > 1e-6 for position in camera_positions):
                raise ValueError("camera horizons must be multiples of the 0.25-second step")
            camera_window = chronological_global[None].float()
            camera_anchor_inverse = torch.linalg.inv(camera_window[:, -1])
            camera_rollout = []
            one_step_seconds = torch.tensor(
                [ROLLOUT_STEP_SECONDS], device=world_device, dtype=torch.float32
            )
            with torch.autocast("cuda", enabled=False):
                for _ in range(max(int(round(position)) for position in camera_positions)):
                    step_relative = camera(
                        camera_window,
                        one_step_seconds,
                        current_features=current_features.float(),
                    )["predicted_relative"][:, 0]
                    next_global = camera_window[:, -1] @ step_relative
                    camera_rollout.append(camera_anchor_inverse @ next_global)
                    camera_window = torch.cat(
                        (camera_window[:, 1:], next_global[:, None]), dim=1
                    )
            predicted_c2w = torch.stack([
                camera_rollout[int(round(position)) - 1][0]
                for position in camera_positions
            ]).to(device)

            result_row = {
                "scene_index": scene_index, "scene_name": item["scene_name"],
                "camera_errors_us": item["camera_errors_us"],
                "lidar_to_camera_errors_us": item["lidar_to_camera_errors_us"],
                "paired_scale": float(paired_scale[0]),
                "paired_support": int(paired_support[0]),
                "target_hw": list(target_hw), "horizons": {},
            }
            for target_index, seconds in enumerate(horizon_seconds):
                key = _horizon_key(seconds)
                target_depth = project_lidar_depth(
                    dataroot, target_lidars[target_index], target_cameras[target_index],
                    calibrations, ego_poses, pixel_transform, target_hw,
                    minimum_depth_m=0.0,
                ).to(device)
                gt_context = _context(
                    encoding, target_c2w[target_index], intrinsics, target_hw
                )
                pred_context = _context(
                    encoding, predicted_c2w[target_index], intrinsics, target_hw
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predictions = {
                        "world_gt_camera": _decode_meters(
                            model, query_states[:, target_index], [gt_context], device
                        )[0].float(),
                        "copy_gt_camera": _decode_meters(
                            model, states[:, 3], [gt_context], device
                        )[0].float(),
                        "world_pred_camera": _decode_meters(
                            model, query_states[:, target_index], [pred_context], device
                        )[0].float(),
                        "copy_pred_camera": _decode_meters(
                            model, states[:, 3], [pred_context], device
                        )[0].float(),
                    }
                metrics = {
                    name: _vggt_world_metrics(
                        prediction, target_depth
                    )
                    for name, prediction in predictions.items()
                }
                for name, metric in metrics.items():
                    _accumulate(totals[name], key, metric)
                gt_pose = target_c2w[target_index]
                translation = float(
                    (predicted_c2w[target_index, :3, 3] - gt_pose[:3, 3]).norm()
                )
                relative_rotation = (
                    gt_pose[:3, :3].T @ predicted_c2w[target_index, :3, :3]
                )
                rotation = float(rotation_angle(relative_rotation) * 180.0 / math.pi)
                for name, value in (
                    ("translation_m", translation), ("rotation_deg", rotation)
                ):
                    camera_totals[key][name] += value
                    camera_totals["overall"][name] += value
                camera_totals[key]["frames"] += 1
                camera_totals["overall"]["frames"] += 1
                result_row["horizons"][key] = {
                    "camera_timestamp": int(target_cameras[target_index]["timestamp"]),
                    "lidar_timestamp": int(target_lidars[target_index]["timestamp"]),
                    "state_time": state_time_details[target_index],
                    "camera_error": {
                        "translation_m": translation,
                        "rotation_deg": rotation,
                    },
                    "metrics": metrics,
                }
            result_row["seconds"] = time.perf_counter() - started
            completed.append(result_row)
            print(json.dumps({"completed": len(completed), "skipped": len(skipped), "scene": item["scene_name"]}), flush=True)
        except Exception as error:
            skipped.append({"scene_name": item["scene_name"], "reason": str(error)})
            print(json.dumps({"skipped": skipped[-1]}), flush=True)
        torch.cuda.empty_cache()

    summaries = {
        name: _summarize_dynamic(total, horizon_seconds)
        for name, total in totals.items()
    }
    camera_summary = {}
    for key in [*(_horizon_key(value) for value in horizon_seconds), "overall"]:
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
        "manifest": str(Path(args.manifest).resolve()),
        "protocol": {
            "split": (
                manifest["selection"]["population"] + "; "
                + manifest["selection"]["method"]
                if "selection" in manifest
                else "all 150 scenes in official nuScenes v1.0-trainval val; no random sampling"
            ),
            "selection_seed": (
                manifest["selection"].get("seed")
                if "selection" in manifest else None
            ),
            "history": (
                "CAM_FRONT single view at "
                + ",".join(f"t{value:+g}" for value in manifest["history_seconds"][:-1])
                + ",t; t is VGGT reference"
            ),
            "input_camera_geometry": (
                "native VGGT point placement; explicit camera/ray conditioning uses "
                "vggt c2w; dataset-to-student Sim(3) retained"
            ),
            "anchor": manifest.get(
                "anchor",
                "fixed tenth 0.2-second logical state, nearest asynchronous CAM_FRONT recorded with residual",
            ),
            "horizon_seconds": list(horizon_seconds),
            "rollout": (
                f"{rollout_steps}-step autoregressive rollout with each model application "
                f"assigned {ROLLOUT_STEP_SECONDS:g} seconds; "
                "linear latent interpolation between bracketing rollout states at exact requested times"
            ),
            "rollout_step_seconds": ROLLOUT_STEP_SECONDS,
            "camera_rollout": "autoregressive four-pose sliding window",
            "camera_rollout_step_seconds": ROLLOUT_STEP_SECONDS,
            "query_camera": "both future GT CAM_FRONT c2w and predicted CAM_FRONT c2w at exact requested times",
            "depth_gt": "closest LIDAR_TOP sweep, transformed via both sensor/ego poses to target CAM_FRONT; z-depth, 0<depth<=80m, nearest-pixel z-buffer",
            "preprocessing": "balanced-512",
            "scene_gauge": "one anchor-t same-pixel VGGT/LiDAR median scale; decoder output not rescaled for model input",
            "metric": "decoder metric depth evaluated directly on valid sparse LiDAR pixels; no output or per-target-frame scale alignment",
            "metric_scale_alignment": False,
            "aggregation": "arithmetic mean of frame metrics",
            "dynamic_policy": "no object downweighting or difficulty filtering",
        },
        "scene_start_index": 0,
        "input_camera_conditioning_source": "vggt",
        "scenes_requested": len(rows), "scenes_evaluated": len(completed),
        "scenes_skipped": len(skipped), "skipped": skipped,
        "complete": not args.only_scene and len(completed) == 150 and not skipped,
        "diagnostic_subset": bool(args.only_scene),
        "target_hw": [list(value) for value in sorted(observed_hws)],
        "summaries": summaries,
        "camera_pose_error_summary": camera_summary,
        "scenes": completed,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2)); os.replace(temporary, output)
    print(json.dumps({"evaluated": len(completed), "skipped": len(skipped), "overall": {name: value["overall"] for name, value in summaries.items()}}, indent=2), flush=True)
    if skipped or len(completed) != len(rows):
        raise RuntimeError(
            f"Incomplete nuScenes paper evaluation: {len(completed)}/{len(rows)} "
            f"scenes evaluated; see {output} for skip reasons"
        )


if __name__ == "__main__":
    main()
