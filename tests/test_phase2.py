"""Phase 2 checkpoint tests: ffprobe metadata + leakage-aware splitter."""
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.split import LeakageError, assert_no_leakage, split_by_source  # noqa: E402
from src.pipeline.probe import build_phase1_synthetic_manifest  # noqa: E402

ROUTER_TARGET_RE = re.compile(r"^(clean_G0|[a-z0-9]+_G[1-9]\d*)$")


@pytest.fixture(scope="module")
def manifest_rows():
    rows = build_phase1_synthetic_manifest()
    manifest_path = REPO_ROOT / "data_cache" / "manifest" / "metadata.jsonl"
    assert manifest_path.is_file()
    with open(manifest_path) as f:
        file_rows = [json.loads(line) for line in f]
    return rows, file_rows


def test_metadata_has_85_rows(manifest_rows):
    rows, file_rows = manifest_rows
    assert len(rows) == 85
    assert len(file_rows) == 85


def test_every_router_target_well_formed(manifest_rows):
    rows, _ = manifest_rows
    for row in rows:
        assert ROUTER_TARGET_RE.match(row["router_target"]), row["router_target"]
    # exactly one clean_G0 (the original, untransformed source)
    clean = [r for r in rows if r["router_target"] == "clean_G0"]
    assert len(clean) == 1


def test_metadata_fields_present(manifest_rows):
    rows, _ = manifest_rows
    required = {"file", "source_id", "router_target", "codec_name", "width", "height", "fps"}
    for row in rows:
        assert required.issubset(row.keys())
        assert row["width"] and row["height"]


# --- splitter leakage tests (synthetic dummy fixture, distinct from metadata.jsonl) ---

def _dummy_rows(n_sources: int, rows_per_source: int) -> list[dict]:
    """Clearly-synthetic fixture rows, used only to exercise split ratios
    with more than one source_id -- metadata.jsonl itself has just one."""
    return [
        {"source_id": f"dummy_src_{s}", "row": r}
        for s in range(n_sources)
        for r in range(rows_per_source)
    ]


def test_split_by_source_no_leakage_on_normal_path():
    rows = _dummy_rows(n_sources=20, rows_per_source=5)
    splits = split_by_source(rows)
    assert_no_leakage(splits)  # should not raise

    total = sum(len(v) for v in splits.values())
    assert total == len(rows)

    # every source_id's rows land together in exactly one split
    for split_rows in splits.values():
        if not split_rows:
            continue
        unique_sources = {r["source_id"] for r in split_rows}
        assert len(split_rows) == len(unique_sources) * 5


def test_split_by_source_respects_approximate_ratios():
    rows = _dummy_rows(n_sources=20, rows_per_source=1)
    splits = split_by_source(rows, ratios=(0.70, 0.15, 0.15))
    assert len(splits["train"]) == 14
    assert len(splits["val"]) == 3
    assert len(splits["test"]) == 3


def test_assert_no_leakage_raises_when_forced():
    rows = _dummy_rows(n_sources=5, rows_per_source=2)
    splits = split_by_source(rows)

    # deliberately force one source_id's row into a different split too
    leaked_source = rows[0]["source_id"]
    leaked_row = next(r for r in rows if r["source_id"] == leaked_source)
    for name, split_rows in splits.items():
        if not any(r["source_id"] == leaked_source for r in split_rows):
            split_rows.append(leaked_row)
            break

    with pytest.raises(LeakageError):
        assert_no_leakage(splits)
