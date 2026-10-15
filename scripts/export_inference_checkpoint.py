#!/usr/bin/env python3
"""Export a minimal, path-free inference checkpoint for separate hosting."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vggt_scene_tokenizer.core.config import AlignmentConfig, ModelConfig


def select_payload(kind: str, source: dict, pose_source: dict | None = None) -> dict:
    """Discard optimizer state, local paths, and private run metadata."""
    if kind == "geostate":
        saved_config = source["config"]
        model_fields = set(ModelConfig.__dataclass_fields__)
        alignment_fields = set(AlignmentConfig.__dataclass_fields__)
        return {
            # The official frozen backbone is loaded independently at inference.
            "model": {
                key: value for key, value in source["model"].items()
                if not key.startswith("vggt.")
            },
            "config": {
                "model": {
                    key: value for key, value in saved_config["model"].items()
                    if key in model_fields
                },
                "alignment": {
                    key: value for key, value in saved_config["alignment"].items()
                    if key in alignment_fields
                },
            },
        }
    if kind == "world":
        return {
            "config": source["config"],
            "model": source["model"],
            "residual_base": source.get("residual_base", "previous"),
        }
    if kind == "camera":
        embedded_fields = ("pose_model_config", "pose_statistics")
        has_embedded = [key in source for key in embedded_fields]
        if any(has_embedded) and not all(has_embedded):
            raise ValueError("Camera checkpoint must contain both embedded pose metadata fields")
        if all(has_embedded):
            pose_metadata = {key: source[key] for key in embedded_fields}
        else:
            if pose_source is None:
                raise ValueError("Legacy camera export requires a base pose checkpoint")
            embedded_state = {
                key.removeprefix("pose_model."): value
                for key, value in source["model"].items()
                if key.startswith("pose_model.")
            }
            base_state = pose_source["model"]
            if embedded_state.keys() != base_state.keys() or any(
                embedded_state[key].shape != value.shape
                or embedded_state[key].dtype != value.dtype
                or not torch.equal(embedded_state[key], value)
                for key, value in base_state.items()
            ):
                raise ValueError("Camera embedded pose weights do not match the base checkpoint")
            pose_metadata = {
                "pose_model_config": pose_source["model_config"],
                "pose_statistics": pose_source["statistics"],
            }
        required = ("format", "camera_config", "model")
        optional = (
            "camera_patch_context",
            "scene_model_used_as_frozen_loss_only", "scene_tokens_as_camera_input",
        )
        return {
            **{key: source[key] for key in required},
            **{key: source[key] for key in optional if key in source},
            **pose_metadata,
        }
    raise ValueError(f"Unknown checkpoint kind: {kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("geostate", "world", "camera"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pose-checkpoint", type=Path,
        help="Base pose checkpoint for merging a legacy camera artifact; unnecessary for an already merged artifact.",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output.exists() or args.input.resolve() == args.output.resolve():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    source = torch.load(args.input, map_location="cpu", mmap=True, weights_only=True)
    pose_source = None
    if args.pose_checkpoint is not None and args.kind != "camera":
        parser.error("--pose-checkpoint is only valid with --kind camera")
    if args.kind == "camera" and not any(
        key in source for key in ("pose_model_config", "pose_statistics")
    ):
        pose_path = args.pose_checkpoint
        if pose_path is None:
            reference = source.get("pose_checkpoint")
            if not reference:
                raise ValueError("Pass --pose-checkpoint when merging a legacy camera artifact")
            pose_path = Path(reference)
            if not pose_path.is_absolute():
                pose_path = args.input.resolve().parent / pose_path
        if not pose_path.is_file():
            raise FileNotFoundError(f"Base pose checkpoint not found: {pose_path}")
        pose_source = torch.load(pose_path, map_location="cpu", mmap=True, weights_only=True)
    payload = select_payload(args.kind, source, pose_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=args.output.parent, prefix=f".{args.output.name}.", suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(payload, temporary_path)
        os.link(temporary_path, args.output)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"Exported {args.kind} inference checkpoint: {args.output}")


if __name__ == "__main__":
    main()
