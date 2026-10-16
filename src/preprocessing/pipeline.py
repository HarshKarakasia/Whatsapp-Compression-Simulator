"""Video -> model-ready tensors, end to end (Phase 4)."""
from pathlib import Path

import yaml

from src.preprocessing.align import align_crop
from src.preprocessing.dct import block_dct
from src.preprocessing.frames import extract_frames

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | str | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "preprocessing.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def preprocess_video(video_path: Path | str, config: dict | None = None) -> dict:
    """Returns {'rgb': [n_frames,3,size,size] float32, 'dct': [n_frames,784,64] float32}."""
    config = config or load_config()
    frames = extract_frames(video_path, n=config["n_frames"])
    rgb = align_crop(
        frames,
        scale=config["face_crop"]["scale"],
        size=config["face_crop"]["size"],
        model_path=REPO_ROOT / config["face_crop"]["model_path"],
    )
    dct = block_dct(rgb, block=config["dct"]["block"])
    return {"rgb": rgb, "dct": dct}
