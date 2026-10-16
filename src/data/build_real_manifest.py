"""Phase 3: probe the real Kaggle sample and build a leakage-checked,
source_id-aware train/val/test split. All rows here are genuinely real
(FF++ c23 / Celeb-DF v2) -- kept in a separate file from Phase 2's
synthetic metadata.jsonl so real and synthetic data are never blended
into one artifact. Scope: clean/G0 only -- the full multi-generation
transform chain was smoke-tested on one real video (see DECISIONS.md's
Phase 3 decision), not run on all 45, per the agreed cost tradeoff.
"""
import json
from pathlib import Path

from src.data.split import assert_no_leakage, split_by_source
from src.pipeline.probe import load_paths, probe_file

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_real_manifest() -> list[dict]:
    sample_manifest_path = REPO_ROOT / "data_cache" / "raw" / "sample_manifest.json"
    with open(sample_manifest_path) as f:
        sample_rows = json.load(f)

    ffprobe_binary = load_paths()["ffprobe_binary"]
    rows = []
    for sr in sample_rows:
        meta = probe_file(REPO_ROOT / sr["local_path"], sr["source_id"], ffprobe_binary)
        meta["label"] = sr["label"]
        meta["dataset"] = sr["dataset"]
        meta["identity"] = sr["identity"]
        if "method" in sr:
            meta["method"] = sr["method"]
        rows.append(meta)

    out_path = REPO_ROOT / "data_cache" / "manifest" / "metadata_real.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return rows


if __name__ == "__main__":
    rows = build_real_manifest()
    splits = split_by_source(rows)
    assert_no_leakage(splits)
    print(f"{len(rows)} real rows probed, written to data_cache/manifest/metadata_real.jsonl")
    for name, split_rows in splits.items():
        sids = sorted({r["source_id"] for r in split_rows})
        print(f"  {name}: {len(split_rows)} rows, {len(sids)} source_ids -> {sids}")
