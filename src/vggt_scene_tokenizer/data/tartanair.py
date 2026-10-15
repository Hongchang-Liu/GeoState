from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np


# TartanAir camera coordinates are NED: +x forward, +y right, +z down.
# This matrix maps an OpenCV camera vector (+x right,+y down,+z forward)
# into that NED camera frame.
NED_FROM_OPENCV = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def quaternion_xyzw_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError("quaternion must have shape [4] in xyzw order")
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("invalid zero or non-finite quaternion")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def tartanair_pose_to_opencv_c2w(pose: np.ndarray) -> np.ndarray:
    """Convert TartanAir ``tx ty tz qx qy qz qw`` to OpenCV c2w."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("TartanAir pose must contain seven finite values")
    world_from_ned = quaternion_xyzw_to_rotation(pose[3:])
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = world_from_ned @ NED_FROM_OPENCV
    c2w[:3, 3] = pose[:3]
    return c2w


def _trajectory_images(archive: zipfile.ZipFile) -> dict[str, list[str]]:
    trajectories: dict[str, list[str]] = defaultdict(list)
    for name in archive.namelist():
        if "/image_left/" not in name or not name.lower().endswith(".png"):
            continue
        trajectory, _ = name.split("/image_left/", 1)
        trajectories[trajectory].append(name)
    for images in trajectories.values():
        images.sort(key=lambda name: int(Path(name).name.split("_", 1)[0]))
    return dict(trajectories)


def _trajectory_poses(archive: zipfile.ZipFile) -> dict[str, np.ndarray]:
    trajectories = {}
    for name in archive.namelist():
        if not name.endswith("/pose_left.txt"):
            continue
        values = np.loadtxt(archive.open(name), dtype=np.float64)
        if values.ndim == 1:
            values = values[None]
        if values.shape[1] != 7:
            raise ValueError(f"Unexpected TartanAir pose shape {values.shape} in {name}")
        trajectories[name.removesuffix("/pose_left.txt")] = values
    return trajectories


def _clip_starts(frame_count: int, span: int, clip_count: int) -> list[int]:
    if frame_count < span or clip_count < 1:
        return []
    count = min(clip_count, frame_count - span + 1)
    return sorted(set(np.linspace(0, frame_count - span, count).round().astype(np.int64).tolist()))


def prepare_tartanair_archives(
    archive_root: str | Path,
    output_root: str | Path,
    *,
    frames_per_clip: int = 24,
    frame_stride: int = 2,
    clips_per_trajectory: int = 4,
) -> dict[str, object]:
    """Extract calibrated RGB clips and their pixel-aligned GT z-depth arrays."""
    archive_root, output_root = Path(archive_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    image_archives = {
        path.name.removesuffix("_image_left.zip"): path
        for path in archive_root.glob("*_image_left.zip")
    }
    depth_archives = {
        path.name.removesuffix("_depth_left.zip"): path
        for path in archive_root.glob("*_depth_left.zip")
    }
    records: list[dict[str, object]] = []
    span = 1 + (frames_per_clip - 1) * frame_stride
    for archive_id in sorted(image_archives.keys() & depth_archives.keys()):
        try:
            image_zip = zipfile.ZipFile(image_archives[archive_id])
            depth_zip = zipfile.ZipFile(depth_archives[archive_id])
        except zipfile.BadZipFile:
            records.append({"archive": archive_id, "status": "incomplete"})
            continue
        with image_zip, depth_zip:
            image_trajectories = _trajectory_images(image_zip)
            pose_trajectories = _trajectory_poses(depth_zip)
            depth_members = set(depth_zip.namelist())
            for trajectory in sorted(image_trajectories.keys() & pose_trajectories.keys()):
                images = image_trajectories[trajectory]
                poses = pose_trajectories[trajectory]
                valid_images = [
                    name
                    for name in images
                    if int(Path(name).name.split("_", 1)[0]) < len(poses)
                ]
                starts = _clip_starts(len(valid_images), span, clips_per_trajectory)
                for start in starts:
                    selected = [valid_images[start + offset * frame_stride] for offset in range(frames_per_clip)]
                    first_id = int(Path(selected[0]).name.split("_", 1)[0])
                    scene_name = f"tartanair__{trajectory.replace('/', '__')}__f{first_id:06d}"
                    scene_path = output_root / scene_name
                    transforms_path = scene_path / "transforms.json"
                    if transforms_path.is_file():
                        try:
                            existing = json.loads(transforms_path.read_text(encoding="utf-8"))
                            existing_frames = existing.get("frames", [])
                            if len(existing_frames) == frames_per_clip and all(
                                frame.get("depth_file_path")
                                and (scene_path / frame["depth_file_path"]).is_file()
                                for frame in existing_frames
                            ):
                                records.append({"scene": scene_name, "status": "existing"})
                                continue
                        except (OSError, json.JSONDecodeError):
                            pass
                    images_path = scene_path / "images"
                    images_path.mkdir(parents=True, exist_ok=True)
                    depths_path = scene_path / "depths"
                    depths_path.mkdir(parents=True, exist_ok=True)
                    frames = []
                    for member in selected:
                        frame_index = int(Path(member).name.split("_", 1)[0])
                        destination = images_path / Path(member).name
                        if not destination.is_file():
                            with image_zip.open(member) as source, destination.open("wb") as target:
                                shutil.copyfileobj(source, target)
                        depth_name = Path(member).name.replace("_left.png", "_left_depth.npy")
                        depth_member = f"{trajectory}/depth_left/{depth_name}"
                        if depth_member not in depth_members:
                            raise FileNotFoundError(f"Missing paired GT depth: {depth_member}")
                        depth_destination = depths_path / depth_name
                        if not depth_destination.is_file():
                            temporary_depth = depth_destination.with_suffix(".tmp")
                            with depth_zip.open(depth_member) as source, temporary_depth.open("wb") as target:
                                shutil.copyfileobj(source, target)
                            temporary_depth.replace(depth_destination)
                        frames.append(
                            {
                                "file_path": f"images/{destination.name}",
                                "frame_id": str(frame_index),
                                "transform_matrix": tartanair_pose_to_opencv_c2w(poses[frame_index]).tolist(),
                                "depth_file_path": f"depths/{depth_name}",
                                "depth_source_member": depth_member,
                                "depth_convention": "z",
                            }
                        )
                    payload = {
                        "camera_model": "OPENCV",
                        "w": 640,
                        "h": 480,
                        "fl_x": 320.0,
                        "fl_y": 320.0,
                        "cx": 320.0,
                        "cy": 240.0,
                        "source_dataset": "tartanair",
                        "source_trajectory": trajectory,
                        "source_archive": archive_id,
                        "gt_depth": {
                            "format": "npy_float32", "convention": "z", "unit": "meters",
                            "alignment": "pixel_aligned_with_file_path",
                        },
                        "frames": frames,
                    }
                    temporary = transforms_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    temporary.replace(transforms_path)
                    records.append({"scene": scene_name, "status": "prepared"})
    counts = {
        status: sum(record["status"] == status for record in records)
        for status in ("prepared", "existing", "incomplete")
    }
    result = {"scene_count": counts["prepared"] + counts["existing"], "status_counts": counts, "records": records}
    (output_root / "preparation_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare calibrated TartanAir RGB clips and GT z-depth")
    parser.add_argument("--archive-root", default="data/tartanair")
    parser.add_argument("--output-root", default="data/tartanair_prepared")
    parser.add_argument("--frames-per-clip", type=int, default=24)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--clips-per-trajectory", type=int, default=4)
    args = parser.parse_args()
    result = prepare_tartanair_archives(
        args.archive_root,
        args.output_root,
        frames_per_clip=args.frames_per_clip,
        frame_stride=args.frame_stride,
        clips_per_trajectory=args.clips_per_trajectory,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))


if __name__ == "__main__":
    main()
