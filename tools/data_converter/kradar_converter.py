"""K-Radar info generator tailored to the shared folder layout.

The user-provided structure is a flat list of scene folders (``1/``, ``2/``, …)
each containing modality subfolders such as ``cam-front/``, ``radar_tesseract/``,
``os1-128/`` and metadata like ``time_info/time_info.csv`` and
``info_label_rev2/``. This converter walks that layout to emit NuScenes-style
``*_infos_<split>.pkl`` files so existing configs can be reused with minimal
changes.
"""

import csv
import glob
import multiprocessing as mp
import os
from typing import Dict, Iterable, List, Optional


def _load_split_filter(root_path: str, version: str) -> Optional[set]:
    """Optional scene filter from ``{version}.txt`` if present.

    If a text file named after the ``version`` exists in ``root_path`` (e.g.,
    ``train.txt``), its lines are treated as the allowed scene folder names.
    Otherwise all scenes are used.
    """

    split_file = os.path.join(root_path, f"{version}.txt")
    if not os.path.isfile(split_file):
        return None

    with open(split_file, "r") as f:
        return {line.strip() for line in f if line.strip()}


def _find_time_file(scene_path: str) -> Optional[str]:
    """Locate a timestamp list (``time_info`` folder) if available."""

    for candidate in [
        os.path.join(scene_path, "time_info", "time_info.csv"),
        os.path.join(scene_path, "time_info", "time_info.txt"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    return None


def _load_timestamps(time_file: str) -> List[str]:
    """Load timestamps/frame IDs from a CSV/TXT file."""

    if time_file is None:
        return []

    timestamps: List[str] = []
    with open(time_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            value = row[0].strip()
            # Skip header rows if they contain non-digit text.
            if not value or not any(ch.isdigit() for ch in value):
                continue
            timestamps.append(value)
    return timestamps


def _stem_candidates(scene_path: str, folder: str) -> List[str]:
    """Infer frame stems from sensor folders when time_info is absent."""

    sensor_dir = os.path.join(scene_path, folder)
    stems: List[str] = []
    if not os.path.isdir(sensor_dir):
        return stems

    for path in glob.glob(os.path.join(sensor_dir, "*")):
        name = os.path.basename(path)
        stem, _sep, _ext = name.partition(".")
        if stem:
            stems.append(stem)
    stems.sort()
    return stems


def _scan_samples(root_path: str, version: str) -> List[Dict[str, str]]:
    """Discover all (scene, frame) pairs under ``root_path``.

    - Scene folders are immediate children (``1/``, ``2/``, …).
    - If ``time_info/time_info.csv`` exists, its first column defines frame IDs.
    - Otherwise frame stems are taken from the radar folder as a fallback.
    - An optional ``{version}.txt`` can restrict the scene list.
    """

    allowed_scenes = _load_split_filter(root_path, version)

    scene_dirs = [
        d
        for d in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, d)) and (allowed_scenes is None or d in allowed_scenes)
    ]
    scene_dirs.sort()

    samples: List[Dict[str, str]] = []
    for scene in scene_dirs:
        scene_path = os.path.join(root_path, scene)
        time_file = _find_time_file(scene_path)
        frame_ids = _load_timestamps(time_file)

        if not frame_ids:
            frame_ids = _stem_candidates(scene_path, "radar_tesseract")
        if not frame_ids:
            frame_ids = _stem_candidates(scene_path, "os1-128")
        if not frame_ids:
            raise RuntimeError(
                f"No frames found under {scene_path}. Add time_info.csv or ensure radar/os1-128 files are present."
            )

        for frame_id in frame_ids:
            samples.append({"scene": scene, "frame_id": frame_id})

    return samples


def _first_existing(base_dir: str, stem: str, exts: List[str]) -> Optional[str]:
    for ext in exts:
        candidate = os.path.join(base_dir, f"{stem}{ext}")
        if os.path.isfile(candidate):
            return candidate
    # glob fallback (handles additional dots in filenames)
    matches = glob.glob(os.path.join(base_dir, f"{stem}.*"))
    return matches[0] if matches else None


def _collect_camera_paths(scene_path: str, frame_id: str) -> Dict[str, Optional[str]]:
    cameras = {}
    for cam_name in ["cam-front", "cam-left", "cam-right", "cam-rear"]:
        cam_dir = os.path.join(scene_path, cam_name)
        cameras[cam_name] = _first_existing(cam_dir, frame_id, [".png", ".jpg", ".jpeg"]) if os.path.isdir(cam_dir) else None
    # Some releases pack all images under a single folder.
    shared_dir = os.path.join(scene_path, "images")
    if os.path.isdir(shared_dir):
        for cam_name in cameras:
            if cameras[cam_name] is None:
                cameras[cam_name] = _first_existing(shared_dir, frame_id, [".png", ".jpg", ".jpeg"])
    return cameras


def _process_sample(sample: Dict[str, str], root_path: str, max_sweeps: int) -> dict:
    """Convert one K-Radar frame into an info dict."""

    scene = sample["scene"]
    frame_id = sample["frame_id"]
    scene_path = os.path.join(root_path, scene)

    lidar_dir = os.path.join(scene_path, "os1-128")
    lidar_path = _first_existing(lidar_dir, frame_id, [".pcd", ".bin", ".npy"]) if os.path.isdir(lidar_dir) else None

    # Secondary lidar (if present)
    lidar2_dir = os.path.join(scene_path, "os2-64")
    lidar2_path = _first_existing(lidar2_dir, frame_id, [".pcd", ".bin", ".npy"]) if os.path.isdir(lidar2_dir) else None

    radar_dir = os.path.join(scene_path, "radar_tesseract")
    radar_path = _first_existing(radar_dir, frame_id, [".pcd", ".bin", ".npy"]) if os.path.isdir(radar_dir) else None

    calib_dir = os.path.join(scene_path, "info_calib")
    calib_path = _first_existing(calib_dir, frame_id, [".json", ".txt", ".yaml"]) if os.path.isdir(calib_dir) else None

    label_dir = os.path.join(scene_path, "info_label_rev2")
    label_path = _first_existing(label_dir, frame_id, [".json", ".txt", ".csv"]) if os.path.isdir(label_dir) else None

    cameras = _collect_camera_paths(scene_path, frame_id)

    return {
        "scene_id": scene,
        "frame_id": frame_id,
        "timestamp": frame_id,
        "lidar_path": lidar_path,
        "lidar2_path": lidar2_path,
        "radar_path": radar_path,
        "calib_path": calib_path,
        "label_path": label_path,
        "camera_paths": cameras,
        "num_sweeps": max_sweeps,
    }


def create_kradar_infos(
    root_path: str,
    info_prefix: str,
    version: str,
    max_sweeps: int = 10,
    out_dir: Optional[str] = None,
    workers: int = 4,
) -> None:
    """Generate info files compatible with a K-Radar dataset class.

    It mirrors ``create_nuscenes_infos`` but walks the K-Radar folder layout
    described above, producing ``{out_dir}/{info_prefix}_infos_{version}.pkl``
    that list per-frame sensor paths and metadata.
    """

    if out_dir is None:
        out_dir = root_path
    os.makedirs(out_dir, exist_ok=True)

    samples = list(_scan_samples(root_path, version))
    if not samples:
        raise RuntimeError("K-Radar samples not found. Please check the dataset paths.")

    with mp.Pool(processes=workers) as pool:
        infos: List[dict] = pool.starmap(
            _process_sample, [(sample, root_path, max_sweeps) for sample in samples]
        )

    import pickle

    info_path = os.path.join(out_dir, f"{info_prefix}_infos_{version}.pkl")
    with open(info_path, "wb") as f:
        pickle.dump(infos, f)

    print(f"K-Radar infos saved to {info_path} ({len(infos)} samples)")
