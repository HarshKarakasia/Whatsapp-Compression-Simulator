"""Uniform frame sampling from a video (Phase 4)."""
from pathlib import Path

import cv2
import numpy as np


def extract_frames(video_path: Path | str, n: int = 16) -> np.ndarray:
    """Uniformly sample n frames across the video. Returns uint8 RGB array
    of shape [n, H, W, 3].

    Reads sequentially rather than seeking by frame index -- seeking can
    land on the wrong frame for videos with B-frames / variable framerate,
    and these clips are short enough that sequential reading is cheap.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"video reports 0 frames: {video_path}")

    target_indices = set(np.linspace(0, total - 1, n).astype(int).tolist())
    frames_by_index = {}
    idx = 0
    while len(frames_by_index) < len(target_indices):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx in target_indices:
            frames_by_index[idx] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()

    sorted_indices = sorted(np.linspace(0, total - 1, n).astype(int).tolist())
    if len(frames_by_index) < len(set(sorted_indices)):
        raise ValueError(
            f"only read {len(frames_by_index)}/{len(set(sorted_indices))} target frames from {video_path}"
        )
    return np.stack([frames_by_index[i] for i in sorted_indices], axis=0)
