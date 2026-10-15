"""Shared Waymo data and decoding helpers for the paper training protocols."""

from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vggt_scene_tokenizer.core.types import SceneEncoding, Sim3
from vggt_scene_tokenizer.data.waymo import CAMERA_NAMES


TARGET_STATE_INDICES = tuple(range(4, 11))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _select_records(
    rows: list[dict[str, Any]], prepared_root: Path, *, limit: int, seed: int, validation: bool
) -> list[dict[str, Any]]:
    available = [row for row in rows if (prepared_root / f"{row['segment']}.pt").is_file()]
    if not available:
        raise FileNotFoundError("No manifest rows have a matching prepared segment")
    count = min(limit, len(available))
    if validation:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in available:
            groups.setdefault(row["segment"], []).append(row)
        for values in groups.values():
            values.sort(key=lambda value: (-float(value["dynamic_score"]), value["first_slot"]))
        selected: list[dict[str, Any]] = []
        cursor = 0
        while len(selected) < count:
            progressed = False
            for segment in sorted(groups):
                if cursor < len(groups[segment]):
                    selected.append(groups[segment][cursor])
                    progressed = True
                    if len(selected) == count:
                        break
            if not progressed:
                break
            cursor += 1
        return selected
    rng = random.Random(seed)
    rng.shuffle(available)
    return available[:count]


class ManifestWaymoDataset(Dataset[dict[str, Any]]):
    """Load fixed history/future windows selected by a Waymo manifest."""

    def __init__(
        self,
        prepared_root: str | Path,
        manifest: str | Path,
        motion_mask_root: str | Path,
        *,
        max_windows: int,
        seed: int,
        validation: bool,
        view_count: int = 33,
        image_view_count: int | None = None,
        target_steps: int = len(TARGET_STATE_INDICES),
    ) -> None:
        self.prepared_root = Path(prepared_root)
        self.motion_mask_root = Path(motion_mask_root)
        if target_steps < 1:
            raise ValueError("target_steps must be positive")
        self.target_state_indices = tuple(range(4, 4 + int(target_steps)))
        expected_view_count = 3 * (4 + int(target_steps))
        if view_count == 33 and target_steps != len(TARGET_STATE_INDICES):
            view_count = expected_view_count
        if view_count != expected_view_count:
            raise ValueError(
                f"A {target_steps}-step sequence requires {expected_view_count} views, got {view_count}"
            )
        self.view_count = int(view_count)
        self.image_view_count = self.view_count if image_view_count is None else int(image_view_count)
        if not 1 <= self.image_view_count <= self.view_count:
            raise ValueError("image_view_count must be between 1 and view_count")
        self.records = _select_records(
            _read_jsonl(manifest), self.prepared_root,
            limit=max_windows, seed=seed, validation=validation,
        )
        self._cached_path: Path | None = None
        self._cached_payload: dict[str, Any] | None = None

    @staticmethod
    def _decode(value: bytes) -> torch.Tensor:
        with Image.open(io.BytesIO(value)) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
        return torch.from_numpy(array).permute(2, 0, 1) / 255.0

    def _payload(self, path: Path) -> dict[str, Any]:
        if path != self._cached_path:
            self._cached_payload = torch.load(path, map_location="cpu", weights_only=False)
            self._cached_path = path
        assert self._cached_payload is not None
        return self._cached_payload

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.records[index]
        segment = row["segment"]
        payload = self._payload(self.prepared_root / f"{segment}.pt")
        logical_slots = tuple(int(value) for value in row["logical_slots"])
        required_states = 4 + len(self.target_state_indices)
        if len(logical_slots) != required_states:
            raise ValueError(f"{row['window_id']} does not contain {required_states} logical states")
        state_order = (3, 0, 1, 2, *self.target_state_indices)
        pairs = [(logical_slots[state], camera) for state in state_order for camera in range(3)]
        pairs = pairs[: self.view_count]
        images = torch.stack([
            self._decode(payload["images_jpeg"][slot][camera])
            for slot, camera in pairs[: self.image_view_count]
        ])
        world_c2w = torch.as_tensor(payload["world_from_camera_cv"]).double()
        physical_c2w = torch.stack([world_c2w[slot, camera] for slot, camera in pairs])
        local_c2w = (torch.linalg.inv(physical_c2w[0])[None] @ physical_c2w).float()
        local_c2w[0] = torch.eye(4)
        intrinsics = torch.as_tensor(payload["intrinsics"]).float()
        input_intrinsics = torch.stack([intrinsics[camera] for _, camera in pairs])
        timestamps = torch.as_tensor(payload["timestamps"], dtype=torch.int64)
        view_states = torch.tensor(
            [state for state in state_order for _ in range(3)], dtype=torch.long
        )[: self.view_count]
        target_front_indices = [int((view_states == state).nonzero()[0]) for state in self.target_state_indices]
        depths = torch.as_tensor(payload["depth_cm"])
        input_scale_depth = depths[logical_slots[3], 0].float() / 100.0
        input_scale_valid = (
            torch.isfinite(input_scale_depth) & (input_scale_depth > 0) & (input_scale_depth <= 80.0)
        )
        input_scale_depth = torch.where(input_scale_valid, input_scale_depth, torch.zeros_like(input_scale_depth))
        target_depths: list[torch.Tensor] = []
        target_valids: list[torch.Tensor] = []
        motion_masks: list[torch.Tensor] = []
        target_timestamps: list[int] = []
        for state in self.target_state_indices:
            target_slot = logical_slots[state]
            target_timestamp = int(timestamps[target_slot])
            target_timestamps.append(target_timestamp)
            target_depth = depths[target_slot, 0].float() / 100.0
            target_valid = torch.isfinite(target_depth) & (target_depth > 0) & (target_depth <= 80.0)
            target_depths.append(torch.where(target_valid, target_depth, torch.zeros_like(target_depth)))
            target_valids.append(target_valid)
            mask_path = self.motion_mask_root / segment / "FRONT" / f"{target_timestamp}.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing motion mask {mask_path}")
            with Image.open(mask_path) as mask_image:
                motion_mask = torch.from_numpy(np.asarray(mask_image, dtype=np.uint8).copy()) > 0
            if motion_mask.shape != target_depth.shape:
                raise ValueError(f"Motion mask shape {motion_mask.shape} != depth {target_depth.shape}")
            motion_masks.append(motion_mask)
        return {
            "window_id": row["window_id"], "segment": segment,
            "dynamic_score": float(row["dynamic_score"]), "images": images,
            "input_c2w": local_c2w, "input_intrinsics": input_intrinsics,
            "view_states": view_states,
            "frame_ids": tuple(f"{int(timestamps[slot])}_{CAMERA_NAMES[camera]}" for slot, camera in pairs),
            "target_state_indices": torch.tensor(self.target_state_indices, dtype=torch.long),
            "target_timestamps": torch.tensor(target_timestamps, dtype=torch.int64),
            "target_c2w": local_c2w[target_front_indices],
            "target_intrinsics": input_intrinsics[target_front_indices],
            "target_depth": torch.stack(target_depths), "target_valid": torch.stack(target_valids),
            "input_scale_depth": input_scale_depth, "input_scale_valid": input_scale_valid,
            "motion_mask": torch.stack(motion_masks),
        }

    def __len__(self) -> int:
        return len(self.records)


def _encoding_from_items(
    scene_tokens: torch.Tensor, items: list[dict[str, Any]], device: torch.device
) -> SceneEncoding:
    if scene_tokens.shape[0] != len(items):
        raise ValueError("Scene-token batch and context-item count disagree")
    height, width = items[0]["target_depth"].shape
    return SceneEncoding(
        memory=scene_tokens.new_empty(scene_tokens.shape[0], 0, scene_tokens.shape[-1]),
        memory_mask=None, scene_tokens=scene_tokens,
        reference_c2w_student=torch.stack([item["reference_c2w_student"] for item in items]).to(device).float(),
        dataset_to_student=Sim3(
            torch.stack([item["dataset_to_student_scale"] for item in items]).to(device).float().reshape(-1),
            torch.stack([item["dataset_to_student_rotation"] for item in items]).to(device).float(),
            torch.stack([item["dataset_to_student_translation"] for item in items]).to(device).float(),
        ),
        scene_scale=torch.stack([item["scene_scale"] for item in items]).to(device).float().reshape(-1),
        input_depth=scene_tokens.new_empty(len(items), 0, height, width),
        input_confidence=scene_tokens.new_empty(len(items), 0, height, width),
        input_points_reference=scene_tokens.new_empty(len(items), 0, height, width, 3),
        patch_hw=(height // 16, width // 16), metadata={},
    )


def _decode_meters(
    geometry_model: torch.nn.Module,
    scene_tokens: torch.Tensor,
    items: list[dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    encoding = _encoding_from_items(scene_tokens, items, device)
    prediction = geometry_model.decode_depth(
        encoding,
        torch.stack([item["target_c2w"] for item in items]).to(device).float(),
        torch.stack([item["target_intrinsics"] for item in items]).to(device).float(),
        tuple(items[0]["target_depth"].shape),
        query_is_dataset_camera=True,
    )
    depth = prediction.z_depth
    if depth.ndim == 4:
        depth = depth[:, 0]
    meters_per_unit = encoding.scene_scale / encoding.dataset_to_student.scale.clamp_min(1e-8)
    return depth * meters_per_unit[:, None, None]


def _region_log_loss(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if not bool(valid.any()):
        return pred.sum() * 0.0
    return (pred[valid].clamp_min(1e-3).log() - gt[valid].clamp_min(1e-3).log()).abs().mean()
