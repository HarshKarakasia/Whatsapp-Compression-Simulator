"""Face detection + alignment crop (Phase 4).

Picked mediapipe over facenet-pytorch's MTCNN for this project -- see
DECISIONS.md: facenet-pytorch's packaging is broken on current
pip/setuptools, mediapipe installed and ran cleanly.
"""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import BaseOptions, vision

REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_PATH = REPO_ROOT / "data_cache" / "models" / "blaze_face_short_range.tflite"


def _make_detector(model_path: Path | str) -> vision.FaceDetector:
    options = vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path=str(model_path)))
    return vision.FaceDetector.create_from_options(options)


def _square_box(x: int, y: int, w: int, h: int, scale: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Expand (x,y,w,h) to a square box scaled by `scale`, centered on the
    original box, clamped to image bounds."""
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * scale
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1 = int(round(cx + side / 2))
    y1 = int(round(cy + side / 2))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img_w, x1), min(img_h, y1)
    return x0, y0, x1, y1


def _center_square_fallback(img_h: int, img_w: int) -> tuple[int, int, int, int]:
    """No face detected -- fall back to the frame's center square crop."""
    side = min(img_h, img_w)
    x0 = (img_w - side) // 2
    y0 = (img_h - side) // 2
    return x0, y0, x0 + side, y0 + side


def align_crop(
    frames: np.ndarray,
    scale: float = 1.3,
    size: int = 224,
    model_path: Path | str = _DEFAULT_MODEL_PATH,
) -> torch.Tensor:
    """Detect, crop (with margin `scale`), and resize each frame to a face
    chip. Returns float32 tensor [N, 3, size, size] in [0, 1]. Frames with
    no detected face fall back to a center-square crop of the whole frame."""
    detector = _make_detector(model_path)
    n, img_h, img_w = frames.shape[0], frames.shape[1], frames.shape[2]
    out = np.empty((n, size, size, 3), dtype=np.uint8)

    for i in range(n):
        frame = frames[i]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = detector.detect(mp_image)

        if result.detections:
            best = max(result.detections, key=lambda d: d.categories[0].score)
            bbox = best.bounding_box
            x0, y0, x1, y1 = _square_box(bbox.origin_x, bbox.origin_y, bbox.width, bbox.height, scale, img_w, img_h)
        else:
            x0, y0, x1, y1 = _center_square_fallback(img_h, img_w)

        crop = frame[y0:y1, x0:x1]
        out[i] = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

    detector.close()
    tensor = torch.from_numpy(out).permute(0, 3, 1, 2).float() / 255.0
    return tensor
