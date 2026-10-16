"""Phase 0 checkpoint tests: repo scaffold, synthetic clip, DECISIONS.md."""
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DIRS = [
    "configs",
    "external",
    "src/pipeline",
    "src/data",
    "src/preprocessing",
    "src/models",
    "src/losses",
    "src/train",
    "src/eval",
    "src/api",
    "ui",
    "tests",
    "data_cache",
    "reports",
]


def test_repo_dirs_exist():
    for rel in EXPECTED_DIRS:
        assert (REPO_ROOT / rel).is_dir(), f"missing expected directory: {rel}"


def test_synthetic_clip_exists_and_is_valid_video():
    clip = REPO_ROOT / "data_cache" / "synthetic" / "test_input.mp4"
    assert clip.is_file(), f"missing synthetic test clip: {clip}"

    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", str(clip)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"ffprobe failed on {clip}: {proc.stderr}"
    assert "codec_type=video" in proc.stdout, "no video stream found in synthetic clip"


def test_decisions_md_has_nonempty_phase0_section():
    decisions = REPO_ROOT / "DECISIONS.md"
    assert decisions.is_file(), "DECISIONS.md does not exist"

    text = decisions.read_text()
    match = re.search(
        r"^##\s*Phase 0.*?\n(.*?)(?=^##\s|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match, "no '## Phase 0' section found in DECISIONS.md"

    body = match.group(1).strip()
    assert len(body) > 0, "'## Phase 0' section in DECISIONS.md is empty"
