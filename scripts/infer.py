#!/usr/bin/env python3
"""Predict ten future states and query specified or predicted cameras on one device."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vggt_scene_tokenizer.data.common import FrameRecord, preprocess_frame
from vggt_scene_tokenizer.evaluation.checkpoint import load_geostate_checkpoint
from vggt_scene_tokenizer.models.camera_conditioning import load_vggt_spatial_camera_checkpoint
from vggt_scene_tokenizer.models.world_model import (
    SceneStatePredictorConfig,
    build_next_scene_state_predictor,
)


HISTORY_SECONDS = (-0.6, -0.4, -0.2, 0.0)
STATE_ORDER = (3, 0, 1, 2)
FUTURE_SECONDS = np.arange(1, 11, dtype=np.float32) / 5


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_conditions(scene: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read RGB and K only; use the model's own pose predictions later."""
    frames = load_json(scene / "condition.json")["frames"]
    calibration = load_json(scene / "intrinsics.json")
    if tuple(float(frame["seconds"]) for frame in frames) != HISTORY_SECONDS:
        raise ValueError("Provide four chronological conditions at -0.6, -0.4, -0.2, 0.0 s")
    camera_order = [item["camera"] for item in frames[-1]["images"]]
    if not camera_order or camera_order[0] != "front":
        raise ValueError("The first camera at each timestamp must be named 'front'")
    images, intrinsics, state_ids = [], [], []
    for state in STATE_ORDER:
        frame = frames[state]
        if [item["camera"] for item in frame["images"]] != camera_order:
            raise ValueError("Use the same ordered cameras at all four timestamps")
        for item in frame["images"]:
            path = scene / item["path"]
            with Image.open(path) as image:
                image_hw = (image.height, image.width)
            if image_hw != tuple(calibration["image_hw"]):
                raise ValueError(f"Image dimensions disagree with the supplied K: {path}")
            # This placeholder is used only by RGB/K preprocessing; it is never
            # passed to the encoder as a condition camera pose.
            record = FrameRecord(
                frame_id=item["path"], image_path=path, c2w=torch.eye(4),
                intrinsics=torch.tensor(calibration["cameras"][item["camera"]]),
                original_hw=image_hw,
            )
            prepared = preprocess_frame(record, 512, 16)
            if not images:
                anchor_pixel_transform = prepared.pixel_transform
            images.append(prepared.image)
            intrinsics.append(prepared.intrinsics)
            state_ids.append(state)
    return (
        torch.stack(images)[None], torch.stack(intrinsics)[None],
        torch.tensor(state_ids)[None], anchor_pixel_transform,
    )


def load_query_camera(path: Path) -> torch.Tensor:
    query = load_json(path)
    if query["coordinate_frame"] != "last_condition_front":
        raise ValueError("Query c2w must be relative to the last condition's FRONT camera")
    if query["translation_units"] != "normalized_scene":
        raise ValueError("Query translations must use normalized_scene units")
    pose = torch.tensor(query["c2w"], dtype=torch.float32)
    if pose.shape != (4, 4) or not bool(torch.isfinite(pose).all()):
        raise ValueError("Query c2w must be a finite 4x4 matrix")
    return pose


def autocast(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def predict_query_cameras(
    model: torch.nn.Module,
    images: torch.Tensor,
    estimated_c2w: torch.Tensor,
    state_ids: torch.Tensor,
    camera_checkpoint: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict relative future poses using estimated FRONT history and current RGB."""
    # History is stored anchor-first for VGGT. Restore chronological FRONT poses
    # for the motion predictor, preserving VGGT's translation gauge throughout.
    front_indices = [int((state_ids[0] == state).nonzero()[0, 0]) for state in range(4)]
    anchor_from_world = torch.linalg.inv(estimated_c2w[:, 0].float())
    history = anchor_from_world[:, None] @ estimated_c2w[:, front_indices].float()
    # Match the released camera predictor's independent current-FRONT feature
    # pass; joint history patch features are not used as its visual input.
    with autocast(device):
        aggregated, patch_start = model.vggt.model.aggregator(images[:, :1])
        patch = aggregated[-1][:, 0, patch_start:].float()
        height, width = images.shape[-2:]
        patch = patch.reshape(1, height // 16, width // 16, -1).permute(0, 3, 1, 2)
        visual = F.adaptive_avg_pool2d(patch, (6, 8)).flatten(2).transpose(1, 2)
    del aggregated, patch
    camera, payload, _ = load_vggt_spatial_camera_checkpoint(str(camera_checkpoint), device)
    camera.eval().requires_grad_(False)
    del payload
    with torch.autocast(device.type, enabled=False):
        future = camera(
            history.float(), torch.tensor(FUTURE_SECONDS, device=device).float(),
            current_features=visual.float(),
        )["predicted_relative"].float()
    return future, history


def save_outputs(output: Path, depths: np.ndarray, current: np.ndarray, metadata: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "depths.npz", seconds=FUTURE_SECONDS,
        depth=depths, current_depth=current,
        intrinsics=np.asarray(metadata["query_intrinsics"], dtype=np.float32),
        query_c2w=np.asarray(metadata["query_c2w"], dtype=np.float32),
        current_query_c2w=np.asarray(metadata["current_query_c2w"], dtype=np.float32),
    )
    # One display range for the entire prediction; this does not rescale depth.
    all_depths = np.concatenate((current[None], depths))
    lo, hi = np.percentile(all_depths[np.isfinite(all_depths) & (all_depths > 0)], (2, 98))
    lo, hi = max(float(lo), 1e-6), max(float(hi), float(lo) + 1e-6)
    previews = []
    for index, depth in enumerate(all_depths):
        position = np.clip((1 / depth.clip(1e-6) - 1 / hi) / (1 / lo - 1 / hi), 0, 1)
        rgb = cv2.cvtColor(
            cv2.applyColorMap(np.rint(position * 255).astype(np.uint8), cv2.COLORMAP_TURBO),
            cv2.COLOR_BGR2RGB,
        )
        image = Image.fromarray(rgb)
        name = "depth_t.png" if index == 0 else f"depth_t_plus_{index / 5:.1f}.png"
        image.save(output / name)
        previews.append(image)
    previews[0].save(
        output / "rollout.gif", save_all=True, append_images=previews[1:],
        duration=200, loop=0,
    )
    metadata["preview_depth_range"] = [lo, hi]
    (output / "prediction.json").write_text(json.dumps(metadata, indent=2) + "\n")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="examples", help="Folder containing RGB and camera JSON files")
    parser.add_argument("--checkpoints", default="checkpoints", help="GeoState, world-model, and camera weight directory")
    parser.add_argument("--config", default="configs/geostate.yaml")
    parser.add_argument("--camera-mode", choices=("specified", "predicted"), default="specified",
                        help="Use a fixed specified viewpoint or ten camera-predictor viewpoints")
    parser.add_argument("--query-camera", help="Fixed query c2w JSON; default: <scene>/query_camera.json")
    parser.add_argument("--query-intrinsics", help="Query K on the supplied FRONT image grid; default: <scene>/query_intrinsics.json")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    scene, weights = Path(args.scene), Path(args.checkpoints)
    device = torch.device(args.device)
    images, input_k, state_ids, anchor_pixel_transform = load_conditions(scene)
    query_path = Path(args.query_camera) if args.query_camera else scene / "query_camera.json"
    query_k_path = Path(args.query_intrinsics) if args.query_intrinsics else scene / "query_intrinsics.json"
    query_normalized = load_query_camera(query_path) if args.camera_mode == "specified" else None
    query_calibration = load_json(query_k_path)
    # Keep the decoder on the same grid as the preprocessed VGGT input. Query
    # intrinsics follow the anchor FRONT's crop/resize, just as in evaluation.
    query_hw = tuple(images.shape[-2:])
    query_k = (
        anchor_pixel_transform
        @ torch.tensor(query_calibration["intrinsics"], dtype=torch.float32)
    )[None]

    _, _, model = load_geostate_checkpoint(args.config, weights / "geostate.pt", device)
    model.eval().requires_grad_(False)
    images, input_k, state_ids = images.to(device), input_k.to(device), state_ids.to(device)
    state_mask = torch.stack([state_ids == state for state in range(4)], dim=1)
    print("Encoding four condition times using VGGT-estimated poses...", flush=True)
    with autocast(device):
        features = model.vggt(images)
        # The estimated cameras themselves define the source frame. No dataset
        # poses or LiDAR scale enter this image-only inference path.
        states, encoding = model.encode_temporal_scene_states_from_vggt_output(
            features, features.c2w,
            input_intrinsics=input_k, state_view_mask=state_mask,
            reference_index=0, input_camera_conditioning_source="vggt",
        )
    if args.camera_mode == "predicted":
        print("Predicting future query cameras from VGGT-estimated FRONT history...", flush=True)
        future_queries, camera_history = predict_query_cameras(
            model, images, features.c2w, state_ids, weights / "camera_predictor.pt", device,
        )
        current_query = torch.eye(4, device=device)[None]
        future_normalized = future_queries.clone()
        future_normalized[..., :3, 3] /= encoding.scene_scale[:, None, None]
        current_normalized = current_query.clone()
    else:
        # decode_depth divides ray origins by scene_scale internally; convert
        # specified normalized translations to the VGGT source gauge once.
        current_normalized = query_normalized.to(device)[None]
        current_query = current_normalized.clone()
        current_query[:, :3, 3] *= encoding.scene_scale[:, None]
        future_queries = current_query[:, None].expand(-1, 10, -1, -1)
        future_normalized = current_normalized[:, None].expand(-1, 10, -1, -1)
        camera_history = None
    del images, features

    payload = torch.load(weights / "world_model.pt", map_location="cpu", mmap=True, weights_only=True)
    world = build_next_scene_state_predictor(SceneStatePredictorConfig(**payload["config"]))
    world.load_state_dict(payload["model"], strict=True)
    world = world.to(device).eval().requires_grad_(False)
    del payload
    if world.config.context_frames != 4:
        raise ValueError("This example requires a four-history-state world checkpoint")
    window = states.float()
    predictions = []
    print("Predicting ten future states, t+0.2 through t+2.0...", flush=True)
    with autocast(device):
        for _ in range(10):
            state = world(window)
            predictions.append(state)
            window = torch.cat((window[:, 1:], state[:, None]), dim=1)
    del world, window
    if device.type == "cuda":
        torch.cuda.empty_cache()

    depths = []
    query_k = query_k.to(device)
    print(f"Decoding states with {args.camera_mode} query cameras...", flush=True)
    with autocast(device):
        for index, state in enumerate([states[:, -1], *predictions]):
            encoding.scene_tokens = state
            query = current_query if index == 0 else future_queries[:, index - 1]
            depth = model.decode_depth(
                encoding, query, query_k, query_hw, query_is_dataset_camera=False,
            ).z_depth
            if depth.ndim == 4:
                depth = depth[:, 0]
            depths.append(depth[0].float().cpu().numpy())
    depth_array = np.stack(depths)
    if not np.isfinite(depth_array).all() or not (depth_array > 0).all():
        raise RuntimeError("The model returned non-finite or non-positive depths")
    metadata = {
        "scene": str(scene), "history_seconds": list(HISTORY_SECONDS),
        "future_seconds": FUTURE_SECONDS.astype(float).round(1).tolist(),
        "rollout_steps": 10, "camera_mode": args.camera_mode,
        "fixed_query_camera": args.camera_mode == "specified", "device": str(device),
        "coordinate_frame": "last_condition_front", "depth_units": "normalized_scene",
        "metric_scale_available": False,
        "condition_pose_source": "vggt", "condition_intrinsics_source": "supplied_calibration",
        "future_rgb_used": False, "condition_extrinsics_used": False,
        "query_camera_file": str(query_path) if args.camera_mode == "specified" else None,
        "query_c2w": future_normalized[0].cpu().tolist(),
        "current_query_c2w": current_normalized[0].cpu().tolist(),
        "query_intrinsics": query_k[0].cpu().tolist(), "image_hw": list(query_hw),
        "scene_scale_vggt": encoding.scene_scale.float().cpu().tolist(),
        "scene_checkpoint": str(weights / "geostate.pt"),
        "world_checkpoint": str(weights / "world_model.pt"),
        "camera_checkpoint": str(weights / "camera_predictor.pt") if camera_history is not None else None,
        "camera_history_pose_source": "vggt" if camera_history is not None else None,
        "camera_history_translation_units": "vggt" if camera_history is not None else None,
        "camera_history_c2w_vggt": camera_history[0].cpu().tolist() if camera_history is not None else None,
        "camera_feature_source": "independent_current_front" if camera_history is not None else None,
    }
    save_outputs(Path(args.output), depth_array[1:], depth_array[0], metadata)
    print(f"Saved ten future depth maps, rollout.gif, and depths.npz to {args.output}", flush=True)


if __name__ == "__main__":
    main()
