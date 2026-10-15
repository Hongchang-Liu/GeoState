"""Prepare the released pretraining scenes from calibrated source datasets."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import io
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch


def _write_scene(path: Path, frames: list[dict], **metadata: object) -> None:
    path.mkdir(parents=True, exist_ok=True)
    temporary = path / "transforms.tmp"
    temporary.write_text(json.dumps({"camera_model": "OPENCV", **metadata, "frames": frames}, indent=2) + "\n")
    temporary.replace(path / "transforms.json")


def _copy_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        shutil.copyfile(source, destination)


def co3d_camera(viewpoint: dict, height: int, width: int) -> dict:
    """Convert PyTorch3D row-vector cameras and isotropic NDC to OpenCV c2w."""
    scale = min(width, height) / 2.0
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = np.asarray(viewpoint["R"], dtype=np.float64).T
    w2c[:3, 3] = np.asarray(viewpoint["T"], dtype=np.float64)
    w2c = np.diag([-1.0, -1.0, 1.0, 1.0]) @ w2c
    return {
        "w": width, "h": height,
        "fl_x": float(viewpoint["focal_length"][0]) * scale,
        "fl_y": float(viewpoint["focal_length"][1]) * scale,
        "cx": width / 2.0 - float(viewpoint["principal_point"][0]) * scale,
        "cy": height / 2.0 - float(viewpoint["principal_point"][1]) * scale,
        "transform_matrix": np.linalg.inv(w2c).tolist(),
    }


def prepare_co3d(raw_root: Path, records: list[dict], data_root: Path) -> None:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in records:
        category, sequence = Path(record["path"]).name.split("__", 1)
        grouped[category][sequence] = record
    for category, requested in sorted(grouped.items()):
        annotations = raw_root / category / "frame_annotations.jgz"
        with gzip.open(annotations, "rt", encoding="utf-8") as handle:
            rows = json.load(handle)
        by_sequence: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row["sequence_name"] in requested:
                by_sequence[row["sequence_name"]].append(row)
        for sequence, record in sorted(requested.items()):
            rows = sorted(by_sequence[sequence], key=lambda row: int(row["frame_number"]))
            if not rows:
                raise FileNotFoundError(f"CO3D annotations missing for {category}/{sequence}")
            destination = data_root / Path(record["path"]).relative_to("data")
            frames = []
            for row in rows:
                source = raw_root / row["image"]["path"]
                if not source.is_file():
                    continue
                name = source.name
                _copy_image(source, destination / "images" / name)
                height, width = map(int, row["image"]["size"])
                frames.append({"file_path": f"images/{name}", **co3d_camera(row["viewpoint"], height, width)})
            if len(frames) < 4:
                raise FileNotFoundError(f"Insufficient downloaded CO3D images: {category}/{sequence}")
            _write_scene(destination, frames)
            print(f"Prepared co3d/{destination.name}: {len(frames)} frames", flush=True)


def re10k_camera(camera: np.ndarray, width: int, height: int) -> dict:
    """Preserve the source OpenCV w2c for the manifest's opencv_w2c adapter."""
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :4] = camera[6:18].reshape(3, 4)
    return {
        "w": width, "h": height,
        "fl_x": float(camera[0]) * width, "fl_y": float(camera[1]) * height,
        "cx": float(camera[2]) * width, "cy": float(camera[3]) * height,
        "transform_matrix": w2c.tolist(),
    }


def prepare_re10k(raw_root: Path, records: list[dict], data_root: Path) -> None:
    requested = {Path(record["path"]).name: record for record in records}
    # The source download places shards directly under train/ or in one range folder.
    shards = sorted({*raw_root.glob("*.torch"), *raw_root.glob("*/*.torch")})
    if not shards:
        raise FileNotFoundError(f"RealEstate10K .torch shards missing: {raw_root}")
    for shard in shards:
        entries = torch.load(shard, map_location="cpu", weights_only=False)
        for entry in entries:
            key = str(entry["key"])
            if key not in requested:
                continue
            record = requested[key]
            destination = data_root / Path(record["path"]).relative_to("data")
            cameras = torch.as_tensor(entry["cameras"]).cpu().numpy()
            images = entry["images"]
            count = min(len(images), len(cameras))
            if count < 4:
                raise ValueError(f"Insufficient RealEstate10K frames: {key}")
            # Match extract_re10k_scenes.py: a centered 24-frame stride-two clip,
            # with a contiguous centered clip for shorter source sequences.
            sample_count = min(24, count)
            stride = 2 if count >= 2 * sample_count - 1 else 1
            span = stride * (sample_count - 1) + 1
            start = (count - span) // 2
            indices = np.arange(start, start + span, stride, dtype=np.int64)
            frames = []
            for output_index, source_index in enumerate(indices):
                value = images[int(source_index)]
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                image_bytes = value.tobytes() if isinstance(value, np.ndarray) else bytes(value)
                with Image.open(io.BytesIO(image_bytes)) as image:
                    width, height = image.size
                name = f"{output_index:06d}.jpg"
                target = destination / "images" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(image_bytes)
                frames.append({"file_path": f"images/{name}", **re10k_camera(cameras[int(source_index)], width, height)})
            _write_scene(destination, frames)
            del requested[key]
            print(f"Prepared re10k/{key}: {len(frames)} frames", flush=True)
        del entries
        if not requested:
            return
    raise FileNotFoundError(f"RealEstate10K shards missing manifest scenes: {sorted(requested)[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("co3d", "re10k"), required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/geostate_pretrain.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    records = [row for row in json.loads(args.manifest.read_text())["scenes"] if row["dataset"] == args.dataset]
    if not records:
        raise ValueError(f"Manifest contains no {args.dataset} scenes")
    functions = {"co3d": prepare_co3d, "re10k": prepare_re10k}
    functions[args.dataset](args.raw_root, records, args.data_root)


if __name__ == "__main__":
    main()
