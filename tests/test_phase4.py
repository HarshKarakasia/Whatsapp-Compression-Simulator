"""Phase 4 checkpoint tests: frame extraction, face alignment, block DCT."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.align import align_crop  # noqa: E402
from src.preprocessing.dct import block_dct  # noqa: E402
from src.preprocessing.pipeline import preprocess_video  # noqa: E402

SAMPLE_FILES = [
    "data_cache/raw/ffpp/original/000.mp4",
    "data_cache/raw/ffpp/deepfakes/000_003.mp4",
    "data_cache/raw/celebdf/id0_0000.mp4",
]


@pytest.fixture(scope="module")
def preprocessed():
    for f in SAMPLE_FILES:
        assert (REPO_ROOT / f).is_file(), f"missing sample file: {f}"
    return {f: preprocess_video(f) for f in SAMPLE_FILES}


def test_rgb_and_dct_shapes_on_real_files(preprocessed):
    for f, out in preprocessed.items():
        assert out["rgb"].shape == (16, 3, 224, 224), f
        assert out["dct"].shape == (16, 784, 64), f
        assert out["rgb"].dtype == torch.float32
        assert out["dct"].dtype == torch.float32


def test_rgb_values_in_unit_range(preprocessed):
    for f, out in preprocessed.items():
        assert out["rgb"].min() >= 0.0 and out["rgb"].max() <= 1.0, f


def test_report_artifacts_exist():
    out_dir = REPO_ROOT / "reports" / "phase4_samples"
    grid = out_dir / "face_crop_grid.png"
    heatmap = out_dir / "dct_heatmap.png"
    assert grid.is_file() and grid.stat().st_size > 1024
    assert heatmap.is_file() and heatmap.stat().st_size > 1024


def test_block_dct_constant_block_has_only_dc_energy():
    """A flat-color image's 8x8 blocks should carry energy only in the DC
    (index 0) coefficient -- a basic correctness check on the DCT math."""
    flat = np.full((1, 16, 16, 3), 128, dtype=np.uint8)
    dct = block_dct(flat, block=8)
    assert dct.shape == (1, 4, 64)
    ac_energy = dct[..., 1:].abs().max().item()
    dc_energy = dct[..., 0].abs().min().item()
    assert ac_energy < 1e-3
    assert dc_energy > 100  # DC coeff scales with block sum for a constant block


def test_align_crop_fallback_on_no_face_frame():
    """A synthetic frame with no face should fall back to a center-square
    crop rather than raising."""
    blank = np.zeros((1, 100, 200, 3), dtype=np.uint8)
    out = align_crop(blank, scale=1.3, size=64)
    assert out.shape == (1, 3, 64, 64)
