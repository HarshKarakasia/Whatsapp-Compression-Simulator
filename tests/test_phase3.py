"""Phase 3 checkpoint tests: real dataset sample, probing, leakage-safe split.

Scope note: the full 84-file transform chain was smoke-tested on one real
video (not re-run here -- it takes ~2m43s) rather than run on all 45 real
sources, per the agreed Phase 3 cost tradeoff (see DECISIONS.md). This test
file checks that smoke test's already-generated output, plus the full
(cheap) probe + leakage-aware split across all 45 real sources.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.split import assert_no_leakage, split_by_source  # noqa: E402
from src.pipeline.probe import load_paths  # noqa: E402

AGREED_BUDGET = 45


@pytest.fixture(scope="module")
def sample_manifest():
    path = REPO_ROOT / "data_cache" / "raw" / "sample_manifest.json"
    assert path.is_file(), "Phase 3 sample was not downloaded"
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def real_metadata():
    path = REPO_ROOT / "data_cache" / "manifest" / "metadata_real.jsonl"
    assert path.is_file(), "real metadata manifest was not built"
    with open(path) as f:
        return [json.loads(line) for line in f]


def test_downloaded_count_matches_agreed_budget(sample_manifest):
    assert len(sample_manifest) == AGREED_BUDGET
    for row in sample_manifest:
        assert (REPO_ROOT / row["local_path"]).is_file()


def test_sample_has_both_datasets_and_labels(sample_manifest):
    datasets = {row["dataset"] for row in sample_manifest}
    labels = {row["label"] for row in sample_manifest}
    assert datasets == {"ffpp", "celebdf"}
    assert labels == {"real", "fake"}


def test_real_manifest_probed_without_errors(sample_manifest, real_metadata):
    assert len(real_metadata) == len(sample_manifest)
    required = {"file", "source_id", "router_target", "codec_name", "width", "height", "label"}
    for row in real_metadata:
        assert required.issubset(row.keys())
        assert row["width"] and row["height"]
        # none of these are transformed -- all clean/G0 at this scope
        assert row["router_target"] == "clean_G0"


def test_real_split_is_leakage_free(real_metadata):
    splits = split_by_source(real_metadata)
    assert_no_leakage(splits)  # should not raise
    total = sum(len(v) for v in splits.values())
    assert total == len(real_metadata)


def test_chain_tree_smoke_test_output_exists_and_plays():
    """Checks the already-generated output of the one real-video smoke
    test (see DECISIONS.md) without re-running the ~2m43s chain build."""
    chain_dir = REPO_ROOT / "data_cache" / "chains"
    real_chain_files = sorted(chain_dir.glob("000__*.mp4"))
    assert len(real_chain_files) == 84, "expected the full 84-file chain for the real smoke-test video"

    ffprobe = load_paths()["ffprobe_binary"]
    for p in real_chain_files[:5]:  # spot-check a few, not all 84, to keep this test fast
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_streams", str(p)], capture_output=True, text=True
        )
        assert proc.returncode == 0, f"ffprobe failed on {p}: {proc.stderr}"
        assert "codec_type=video" in proc.stdout
