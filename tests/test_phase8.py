"""Phase 8 checkpoint tests: metrics, slicing, comparison runner.

No real trained checkpoint exists yet, so this tests against synthetic
prediction arrays only, per BUILD.md's explicit allowance for this case.
"""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.benchmark import VARIANTS, comparison_table_by_depth, comparison_table_seen_unseen  # noqa: E402
from src.eval.metrics import accuracy, auc, ece  # noqa: E402
from src.eval.slicing import depth_of, slice_by_depth, slice_by_seen_unseen  # noqa: E402


def test_auc_perfect_and_random():
    y_true = [0, 0, 1, 1]
    assert auc(y_true, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert auc(y_true, [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_auc_undefined_single_class_returns_nan():
    result = auc([1, 1, 1], [0.6, 0.7, 0.8])
    assert result != result  # NaN


def test_accuracy_basic():
    assert accuracy([0, 1, 1, 0], [0.1, 0.9, 0.6, 0.4]) == pytest.approx(1.0)
    assert accuracy([0, 1, 1, 0], [0.9, 0.1, 0.1, 0.9]) == pytest.approx(0.0)


def test_ece_zero_when_perfectly_calibrated():
    # prediction exactly equals the empirical bin accuracy
    y_true = [0, 0, 1, 1]
    y_prob = [0.0, 0.0, 1.0, 1.0]
    assert ece(y_true, y_prob) == pytest.approx(0.0, abs=1e-6)


def test_depth_of_parses_router_targets():
    assert depth_of("clean_G0") == 0
    assert depth_of("h264_G1") == 1
    assert depth_of("av1_G3") == 3
    with pytest.raises(ValueError):
        depth_of("not_a_router_target")


def test_slice_by_depth_bucket_counts():
    rows = [
        {"router_target": "clean_G0"},
        {"router_target": "h264_G1"},
        {"router_target": "h264_G1"},
        {"router_target": "av1_G3"},
    ]
    slices = slice_by_depth(rows)
    assert {d: len(v) for d, v in slices.items()} == {0: 1, 1: 2, 3: 1}


def test_slice_by_seen_unseen():
    rows = [{"router_target": rt} for rt in ["clean_G0", "h264_G1", "av1_G3", "vp9_G2"]]
    slices = slice_by_seen_unseen(rows, seen_router_targets={"clean_G0", "h264_G1"})
    assert len(slices["seen"]) == 2
    assert len(slices["unseen"]) == 2


def test_comparison_table_covers_all_variants_and_depths():
    rows = [
        {"router_target": "clean_G0", "label": 0, "pred_moe": 0.1, "pred_single_head": 0.2, "pred_no_con": 0.3},
        {"router_target": "h264_G1", "label": 1, "pred_moe": 0.8, "pred_single_head": 0.6, "pred_no_con": 0.5},
        {"router_target": "h264_G1", "label": 0, "pred_moe": 0.3, "pred_single_head": 0.4, "pred_no_con": 0.5},
        {"router_target": "av1_G3", "label": 1, "pred_moe": 0.7, "pred_single_head": 0.5, "pred_no_con": 0.4},
    ]
    table = comparison_table_by_depth(rows)
    variants_seen = {r["variant"] for r in table}
    assert variants_seen == set(VARIANTS)
    depths_seen = {r["key"] for r in table if r["slice"] == "depth"}
    assert depths_seen == {"G0", "G1", "G3"}
    for r in table:
        assert r["n"] > 0
        assert 0 <= r["acc"] <= 1
        assert r["ece"] >= 0

    su_table = comparison_table_seen_unseen(rows, seen_router_targets={"clean_G0", "h264_G1"})
    assert {r["key"] for r in su_table} == {"seen", "unseen"}


def test_synthetic_report_artifacts_exist():
    table_path = REPO_ROOT / "reports" / "SYNTHETIC_benchmark_table.txt"
    chart_path = REPO_ROOT / "reports" / "SYNTHETIC_degradation_chart.png"
    assert table_path.is_file()
    assert chart_path.is_file() and chart_path.stat().st_size > 1024
    text = table_path.read_text()
    assert "SYNTHETIC" in text and "NOT REAL RESULTS" in text
