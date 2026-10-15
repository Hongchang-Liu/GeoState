"""Prepare a storage-efficient WildRGB-D subset directly from remote ZIPs.

Read the released scene manifest, then fetch its RGB, metadata, and pose
members over HTTP ranges from the official category ZIPs.
"""

from __future__ import annotations

import argparse
import json
import struct
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np


BASE_URL = "https://huggingface.co/hongchi/wildrgbd/resolve/main/{category}.zip?download=true"


@dataclass(frozen=True)
class ZipMember:
    name: str
    method: int
    compressed_size: int
    local_offset: int
    name_size: int
    extra_size: int


def _get(url: str, start: int, end: int) -> bytes:
    # Include the range in the cache key so redirects preserve each request.
    request = urllib.request.Request(f"{url}&geostate_range={start}-{end}", headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError("The archive server must support HTTP byte ranges")
        data = response.read(end - start + 2)
    expected = end - start + 1
    if len(data) != expected:
        raise RuntimeError(f"Range response has {len(data)} bytes; expected {expected} for {start}-{end}")
    return data


def _zip64_offset(extra: bytes, *, compressed_size: int, uncompressed_size: int, local_offset: int) -> tuple[int, int, int]:
    cursor = 0
    while cursor + 4 <= len(extra):
        tag, size = struct.unpack_from("<HH", extra, cursor)
        payload = extra[cursor + 4 : cursor + 4 + size]
        cursor += 4 + size
        if tag != 1:
            continue
        offset = 0
        if uncompressed_size == 0xFFFFFFFF:
            uncompressed_size = struct.unpack_from("<Q", payload, offset)[0]; offset += 8
        if compressed_size == 0xFFFFFFFF:
            compressed_size = struct.unpack_from("<Q", payload, offset)[0]; offset += 8
        if local_offset == 0xFFFFFFFF:
            local_offset = struct.unpack_from("<Q", payload, offset)[0]
        return compressed_size, uncompressed_size, local_offset
    if 0xFFFFFFFF in (compressed_size, uncompressed_size, local_offset):
        raise ValueError("ZIP64 member is missing its ZIP64 extra field")
    return compressed_size, uncompressed_size, local_offset


def _central_directory(url: str) -> dict[str, ZipMember]:
    # One MiB is ample for either the classic or ZIP64 EOCD.
    request = urllib.request.Request(f"{url}&geostate_range=tail", headers={"Range": "bytes=-1048576"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError("The archive server must support HTTP byte ranges")
        tail = response.read(1048577)
        total = int(response.headers["Content-Range"].rsplit("/", 1)[1])
    marker = tail.rfind(b"PK\x06\x06")
    if marker >= 0:
        _, _, _, _, _, _, _, _, directory_size, directory_offset = struct.unpack_from("<4sQHHIIQQQQ", tail, marker)
    else:
        marker = tail.rfind(b"PK\x05\x06")
        if marker < 0:
            raise ValueError("Missing ZIP end-of-central-directory record")
        _, _, _, _, _, directory_size, directory_offset, _ = struct.unpack_from("<4sHHHHIIH", tail, marker)
    directory = _get(url, directory_offset, directory_offset + directory_size - 1)
    if directory_offset + directory_size > total:
        raise ValueError("Invalid ZIP central directory bounds")
    members: dict[str, ZipMember] = {}
    cursor = 0
    while cursor + 46 <= len(directory) and directory[cursor : cursor + 4] == b"PK\x01\x02":
        header = struct.unpack_from("<4sHHHHHHIIIHHHHHII", directory, cursor)
        method, compressed, uncompressed = header[4], header[8], header[9]
        name_size, extra_size, comment_size = header[10], header[11], header[12]
        local_offset = header[16]
        name_start = cursor + 46
        name = directory[name_start : name_start + name_size].decode("utf-8")
        extra = directory[name_start + name_size : name_start + name_size + extra_size]
        compressed, _, local_offset = _zip64_offset(
            extra, compressed_size=compressed, uncompressed_size=uncompressed, local_offset=local_offset
        )
        if not name.endswith("/"):
            members[name] = ZipMember(name, method, compressed, local_offset, name_size, extra_size)
        cursor += 46 + name_size + extra_size + comment_size
    return members


def _member_bytes(url: str, member: ZipMember) -> bytes:
    # Central- and local-header extra lengths are allowed to differ.  Fetch a
    # small header cushion, then locate the payload from the local header.
    end = member.local_offset + 30 + member.name_size + member.extra_size + member.compressed_size + 4096 - 1
    packet = _get(url, member.local_offset, end)
    if packet[:4] != b"PK\x03\x04":
        raise ValueError(f"Bad local ZIP header for {member.name}")
    local_name_size, local_extra_size = struct.unpack_from("<HH", packet, 26)
    start = 30 + local_name_size + local_extra_size
    payload = packet[start : start + member.compressed_size]
    if len(payload) != member.compressed_size:
        raise ValueError(f"Truncated ZIP member {member.name}")
    if member.method == 0:
        return payload
    if member.method == 8:
        return zlib.decompress(payload, -15)
    raise ValueError(f"Unsupported ZIP compression method {member.method} for {member.name}")


def _poses(payload: bytes) -> dict[str, list[list[float]]]:
    result: dict[str, list[list[float]]] = {}
    for line in payload.decode("utf-8").splitlines():
        values = line.split()
        if len(values) != 17:
            continue
        # Pose files may use ``126`` while RGB members use ``00126.png``.
        result[str(int(float(values[0])))] = np.asarray(values[1:], dtype=np.float64).reshape(4, 4).tolist()
    return result


def _intrinsics(payload: bytes) -> tuple[int, int, float, float, float, float]:
    data = json.loads(payload.decode("utf-8"))
    k = [float(value) for value in data["K"]]
    if len(k) != 9:
        raise ValueError("WildRGB-D metadata K must contain 9 values")
    # K is serialized column-major: [[fx,0,cx],[0,fy,cy],[0,0,1]].T.flatten().
    return int(data["w"]), int(data["h"]), k[0], k[4], k[6], k[7]


def prepare_wildrgbd(
    output_root: str | Path,
    *,
    records: list[dict],
    workers: int = 16,
) -> dict:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    frames_per_scene = 24
    selected: list[tuple[str, str, dict[str, ZipMember]]] = []
    requested: dict[str, list[str]] = {}
    for record in records:
        _, category, scene = Path(record["path"]).name.split("__", 2)
        requested.setdefault(category, []).append(scene)
    for category, scene_names in requested.items():
        members = _central_directory(BASE_URL.format(category=category))
        for scene in scene_names:
            prefix = f"{category}/scenes/{scene}"
            selected.append((category, prefix, members))

    prepared = existing = failed = 0
    errors: list[str] = []
    for category, prefix, members in selected:
        scene_name = f"wildrgbd__{category}__{prefix.rsplit('/', 1)[-1]}"
        scene_dir = output_root / scene_name
        transforms_path = scene_dir / "transforms.json"
        if transforms_path.is_file():
            try:
                if len(json.loads(transforms_path.read_text(encoding="utf-8"))["frames"]) == frames_per_scene:
                    existing += 1
                    continue
            except (OSError, ValueError, KeyError):
                pass
        url = BASE_URL.format(category=category)
        try:
            metadata = _intrinsics(_member_bytes(url, members[f"{prefix}/metadata"]))
            poses = _poses(_member_bytes(url, members[f"{prefix}/cam_poses.txt"]))
            rgb_members = [member for name, member in members.items() if name.startswith(f"{prefix}/rgb/") and name.endswith(".png")]
            rgb_members.sort(key=lambda member: int(Path(member.name).stem))
            usable = [member for member in rgb_members if str(int(Path(member.name).stem)) in poses]
            if len(usable) < frames_per_scene:
                raise ValueError(f"Only {len(usable)} RGB frames with poses")
            indices = np.linspace(0, len(usable) - 1, frames_per_scene).round().astype(int).tolist()
            chosen = [usable[index] for index in indices]
            images_dir = scene_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                blobs = list(executor.map(lambda member: _member_bytes(url, member), chosen))
            width, height, fx, fy, cx, cy = metadata
            frames = []
            for member, blob in zip(chosen, blobs):
                name = Path(member.name).name
                temporary = images_dir / f".{name}.tmp"
                temporary.write_bytes(blob)
                temporary.replace(images_dir / name)
                frame_id = Path(name).stem
                frames.append({"file_path": f"images/{name}", "frame_id": frame_id, "w": width, "h": height, "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "transform_matrix": poses[str(int(frame_id))]})
            payload = {"camera_model": "OPENCV", "source_dataset": "wildrgbd", "source_category": category, "frames": frames}
            temporary = transforms_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(transforms_path)
            prepared += 1
        except Exception as error:
            failed += 1
            if len(errors) < 20:
                errors.append(f"{scene_name}: {type(error).__name__}: {error}")
    manifest = {"requested": len(records), "prepared": prepared, "existing": existing, "failed": failed, "errors": errors}
    (output_root / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="data/wildrgbd_prepared")
    parser.add_argument("--manifest", type=Path, default=Path("manifests/geostate_pretrain.json"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    records = [row for row in json.loads(args.manifest.read_text())["scenes"] if row["dataset"] == "wildrgbd"]
    result = prepare_wildrgbd(args.output_root, records=records, workers=args.workers)
    print(json.dumps(result))
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
