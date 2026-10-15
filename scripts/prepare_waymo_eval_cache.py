#!/usr/bin/env python3
"""Build the fixed tenth-frame Waymo evaluation cache from prepared segments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


HORIZON_SECONDS = tuple(step / 5.0 for step in range(1, 11))
ANCHOR_ONE_BASED = 10


def strict_timestamps(
    timestamps: list[int], tolerance_us: int,
) -> tuple[int, int, list[int | None], list[int | None]]:
    """Match the paper's tenth 5-Hz logical state and nearest future frames."""
    if not timestamps or timestamps != sorted(set(timestamps)):
        raise ValueError("prepared timestamps must be nonempty, sorted, and unique")
    values = np.asarray(timestamps, dtype=np.int64)

    def nearest(expected: int, enforce_tolerance: bool = True) -> tuple[int | None, int | None]:
        position = int(np.searchsorted(values, expected))
        candidates = [index for index in (position - 1, position) if 0 <= index < len(values)]
        if not candidates:
            return None, None
        chosen = min(candidates, key=lambda index: abs(int(values[index]) - expected))
        target = int(values[chosen])
        error = target - expected
        if enforce_tolerance and abs(error) > tolerance_us:
            return None, error
        return target, error

    expected_anchor = timestamps[0] + (ANCHOR_ONE_BASED - 1) * 200_000
    anchor, anchor_error = nearest(expected_anchor, enforce_tolerance=False)
    if anchor is None or anchor_error is None:
        raise ValueError("no complete tenth-frame anchor")
    targets, errors = [], []
    for seconds in HORIZON_SECONDS:
        target, error = nearest(anchor + int(round(seconds * 1_000_000)))
        targets.append(target)
        errors.append(error)
    present = [target for target in targets if target is not None]
    if len(set(present)) != len(present):
        raise ValueError("future query timestamps are not unique")
    return anchor, anchor_error, targets, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--timestamp-tolerance-ms", type=float, default=20.0)
    parser.add_argument("--only-segment", help="Build one cache for a diagnostic run.")
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    segments = list(selection["selected_segments"])
    if (
        selection.get("population_count") != 202
        or selection.get("seed") != 20260804
        or selection.get("selected_count") != 100
        or len(segments) != 100
        or len(set(segments)) != 100
        or segments != sorted(segments)
    ):
        raise ValueError("selection must be the fixed seed-20260804 Waymo Random100")
    if args.only_segment:
        if args.only_segment not in segments:
            raise ValueError("--only-segment is not in the fixed selection")
        segments = [args.only_segment]
    tolerance_us = int(round(args.timestamp_tolerance_ms * 1000.0))
    if tolerance_us < 0:
        raise ValueError("timestamp tolerance must be nonnegative")
    args.output_root.mkdir(parents=True, exist_ok=True)

    for number, segment in enumerate(segments, start=1):
        output = args.output_root / f"{segment}.pt"
        if output.exists():
            if not args.resume_existing:
                raise FileExistsError(f"Refusing to overwrite existing cache: {output}")
            existing = torch.load(output, map_location="cpu", mmap=True, weights_only=True)
            if (
                existing.get("format") != "waymo_tenth_frame_strict_sparse_from_prepared_v1"
                or existing.get("segment") != segment
            ):
                raise ValueError(f"Existing cache does not match the protocol: {output}")
            print(json.dumps({"completed": number, "total": len(segments), "segment": segment, "status": "existing"}), flush=True)
            continue
        source = args.prepared_root / f"{segment}.pt"
        # Prepared Waymo payloads may contain byte-valued sensor metadata.
        with torch.serialization.safe_globals([bytes]):
            payload = torch.load(source, map_location="cpu", mmap=True, weights_only=True)
        if payload.get("format") != "waymo_front3_lidar_v1":
            raise ValueError(f"unexpected prepared format: {source}")
        timestamps = [int(value) for value in torch.as_tensor(payload["timestamps"]).tolist()]
        anchor, anchor_error, targets, errors = strict_timestamps(timestamps, tolerance_us)
        index = {timestamp: position for position, timestamp in enumerate(timestamps)}
        depth = torch.as_tensor(payload["depth_cm"])
        cameras = torch.as_tensor(payload["world_from_camera_cv"])
        if depth.shape[:2] != (len(timestamps), 3) or cameras.shape != (len(timestamps), 3, 4, 4):
            raise ValueError(f"invalid prepared depth or camera shape: {source}")
        target_indices = [index[target] if target is not None else -1 for target in targets]
        empty_depth = torch.zeros_like(depth[0, 0])
        cache = {
            "format": "waymo_tenth_frame_strict_sparse_from_prepared_v1",
            "segment": segment,
            "complete_timestamp_count": len(timestamps),
            "anchor_one_based": ANCHOR_ONE_BASED,
            "anchor_timestamp": anchor,
            "anchor_timestamp_error_us": anchor_error,
            "target_timestamps": torch.tensor(
                [target if target is not None else -1 for target in targets], dtype=torch.int64
            ),
            "timestamp_errors_us": torch.tensor(
                [error if target is not None else -(2**63) for target, error in zip(targets, errors)],
                dtype=torch.int64,
            ),
            "target_available": torch.tensor([target is not None for target in targets]),
            "anchor_world_from_camera_cv": cameras[index[anchor]].clone(),
            "target_world_from_camera_cv": torch.stack([
                cameras[position, 0] if position >= 0 else torch.eye(4, dtype=cameras.dtype)
                for position in target_indices
            ]).clone(),
            "intrinsics": torch.as_tensor(payload["intrinsics"]).clone(),
            "depth_cm": torch.stack([
                depth[index[anchor], 0],
                *[depth[position, 0] if position >= 0 else empty_depth for position in target_indices],
            ]).clone(),
            "image_hw": tuple(int(value) for value in payload["image_hw"]),
            "depth_convention": "opencv_z_centimeters",
            "lidars": "all_five_first_and_second_returns",
        }
        temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
        try:
            torch.save(cache, temporary)
            os.link(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        print(json.dumps({"completed": number, "total": len(segments), "segment": segment}), flush=True)


if __name__ == "__main__":
    main()
