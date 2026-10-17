"""Phase 7 checkpoint tests: local training-loop dry-run.

Uses small settings (2 steps, n_frames=2) to keep this fast and CPU-
friendly for routine test runs -- the real dry-run (5 steps, n_frames=4,
~100s, see DECISIONS.md) is what actually demonstrated the loop on a
realistic-ish shape; this test just needs to prove the mechanics
(doesn't crash, logs, checkpoints, reloads) on every run.
"""
import csv
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.model import DeepfakeDetector, load_config  # noqa: E402
from src.train.dataset import SourceGroupedBatcher, load_real_manifest  # noqa: E402
from src.train.train import load_checkpoint, train  # noqa: E402


def test_source_grouped_batcher_respects_counts():
    rows = load_real_manifest()
    batcher = SourceGroupedBatcher(rows, sources_per_batch=3, views_per_source=2, seed=0)
    batch = batcher.sample_batch()
    assert len(batch) == 6
    source_ids = [r["source_id"] for r in batch]
    assert len(set(source_ids)) == 3
    for sid in set(source_ids):
        assert source_ids.count(sid) == 2


@pytest.fixture(scope="module")
def dry_run_result(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("phase7")
    logs = train(
        n_steps=2,
        sources_per_batch=2,
        views_per_source=1,
        n_frames_override=2,
        checkpoint_path=tmp_dir / "test_checkpoint.pt",
        log_path=tmp_dir / "test_log.csv",
        save_optimizer_state=False,
    )
    return logs, tmp_dir


def test_dry_run_completes_n_steps_without_error(dry_run_result):
    logs, _ = dry_run_result
    assert len(logs) == 2
    for row in logs:
        assert torch.isfinite(torch.tensor(row["total"]))


def test_loss_is_logged_to_csv(dry_run_result):
    _, tmp_dir = dry_run_result
    log_path = tmp_dir / "test_log.csv"
    assert log_path.is_file()
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert "total" in rows[0]


def test_checkpoint_written_and_reloadable(dry_run_result):
    _, tmp_dir = dry_run_result
    ckpt_path = tmp_dir / "test_checkpoint.pt"
    assert ckpt_path.is_file()

    config = load_config()
    model = DeepfakeDetector(config)
    ckpt = load_checkpoint(ckpt_path, model)
    assert ckpt["step"] == 2
    # loaded without raising -- also sanity-check a real parameter round-tripped
    loaded_param = next(model.parameters())
    assert torch.isfinite(loaded_param).all()
