"""FFprobe metadata extraction with actual (not requested) stream properties
(Phase 2). Measures what an encoder really produced -- resolution, fps,
bitrate, keyframe interval -- rather than echoing back the requested recipe.
"""
import json
import re
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Phase 1 encodes lineage into the filename as repeated "__<codec>_G<depth>"
# tags (see build_chain_tree); the last tag gives the router target.
LINEAGE_HOP_RE = re.compile(r"__([a-z0-9]+)_G(\d+)")


def load_paths(path: Path | str | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "paths.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def router_target_for(path: Path) -> str:
    matches = LINEAGE_HOP_RE.findall(path.stem)
    if not matches:
        return "clean_G0"
    last_codec, depth = matches[-1]
    return f"{last_codec}_G{depth}"


def _ffprobe_json(path: Path, ffprobe_binary: str) -> dict:
    proc = subprocess.run(
        [ffprobe_binary, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)


def _keyframe_indices(path: Path, ffprobe_binary: str) -> list[int]:
    proc = subprocess.run(
        [ffprobe_binary, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=key_frame", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    flags = [int(x.strip().rstrip(",")) for x in proc.stdout.splitlines() if x.strip()]
    return [i for i, v in enumerate(flags) if v == 1]


def probe_file(path: Path, source_id: str, ffprobe_binary: str) -> dict:
    """Extract actual stream properties for one file. Raises if it's not a
    valid, decodable video (ffprobe failure, no video stream)."""
    path = Path(path)
    info = _ffprobe_json(path, ffprobe_binary)
    video_streams = [s for s in info["streams"] if s.get("codec_type") == "video"]
    if not video_streams:
        raise ValueError(f"no video stream found in {path}")
    stream = video_streams[0]
    fmt = info.get("format", {})

    keyframes = _keyframe_indices(path, ffprobe_binary)
    if len(keyframes) >= 2:
        intervals = [b - a for a, b in zip(keyframes, keyframes[1:])]
        actual_gop = sum(intervals) / len(intervals)
    else:
        # fewer than 2 keyframes observed in this clip -> interval can't be
        # measured within the clip's length (e.g. encoder's default keyint
        # exceeds total frame count).
        actual_gop = None

    num, _, den = stream.get("r_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if den and float(den) != 0 else float(num)

    bit_rate = stream.get("bit_rate") or fmt.get("bit_rate")
    duration = fmt.get("duration")

    try:
        rel_file = str(path.relative_to(REPO_ROOT))
    except ValueError:
        rel_file = str(path)

    return {
        "file": rel_file,
        "source_id": source_id,
        "router_target": router_target_for(path),
        "codec_name": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps,
        "bitrate_kbps": round(float(bit_rate) / 1000, 2) if bit_rate else None,
        "keyframe_count": len(keyframes),
        "actual_gop": actual_gop,
        "duration_sec": float(duration) if duration else None,
    }


def build_metadata(
    file_paths: list[Path], source_id: str, ffprobe_binary: str | None = None
) -> list[dict]:
    ffprobe_binary = ffprobe_binary or load_paths()["ffprobe_binary"]
    return [probe_file(p, source_id, ffprobe_binary) for p in file_paths]


def write_metadata_jsonl(rows: list[dict], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_phase1_synthetic_manifest() -> list[dict]:
    """Probe the Phase 0 synthetic source + every Phase 1 chain file, and
    write data_cache/manifest/metadata.jsonl. All rows share one clearly-
    synthetic source_id, since every file here derives from the single
    Phase 0 test clip -- see PROGRESS.md / DECISIONS.md."""
    from src.pipeline.transform import load_config  # local import avoids a cycle at module load

    config = load_config()
    source = REPO_ROOT / "data_cache" / "synthetic" / "test_input.mp4"
    chain_dir = REPO_ROOT / config["output_dir"]

    ffprobe_binary = load_paths()["ffprobe_binary"]
    source_id = "synthetic_test_input"

    # chain_dir is shared with other sources (e.g. Phase 3's real-video
    # smoke test), so only glob this source's own stem-prefixed outputs.
    paths = [source] + sorted(chain_dir.glob(f"{source.stem}__*.mp4"))
    rows = build_metadata(paths, source_id, ffprobe_binary)
    write_metadata_jsonl(rows, REPO_ROOT / "data_cache" / "manifest" / "metadata.jsonl")
    return rows


if __name__ == "__main__":
    rows = build_phase1_synthetic_manifest()
    print(f"wrote {len(rows)} rows to data_cache/manifest/metadata.jsonl")
