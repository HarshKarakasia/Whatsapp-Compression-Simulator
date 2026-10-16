"""Phase 1 checkpoint tests: multi-codec, multi-generation transform chain."""
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.transform import build_chain_tree, load_config, load_paths  # noqa: E402


@pytest.fixture(scope="module")
def chain_outputs():
    config = load_config()
    source = REPO_ROOT / "data_cache" / "synthetic" / "test_input.mp4"
    assert source.is_file(), "Phase 0 synthetic clip is missing"
    outputs = build_chain_tree(source, config=config)
    return outputs, config


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_expected_file_count(chain_outputs):
    outputs, config = chain_outputs
    n_codecs = len(config["codecs"])
    depth = config["depth"]
    expected = sum(n_codecs**d for d in range(1, depth + 1))
    assert len(outputs) == expected
    assert all(p.is_file() for p in outputs)


def test_every_file_plays(chain_outputs):
    outputs, _ = chain_outputs
    paths_cfg = load_paths()
    ffprobe = paths_cfg["ffprobe_binary"]
    for p in outputs:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_streams", str(p)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"ffprobe failed on {p}: {proc.stderr}"
        assert "codec_type=video" in proc.stdout, f"no video stream in {p}"


def test_no_two_files_byte_identical(chain_outputs):
    outputs, _ = chain_outputs
    hashes = [_md5(p) for p in outputs]
    assert len(hashes) == len(set(hashes)), "found byte-identical output files"


def test_hop3_smaller_than_hop1_same_lineage(chain_outputs):
    outputs, config = chain_outputs
    codecs = list(config["codecs"].keys())
    first_codec = codecs[0]

    hop1 = next(p for p in outputs if p.stem.endswith(f"__{first_codec}_G1"))
    hop3 = next(
        p
        for p in outputs
        if p.stem.endswith(f"__{first_codec}_G1__{first_codec}_G2__{first_codec}_G3")
    )
    assert hop3.stat().st_size < hop1.stat().st_size
