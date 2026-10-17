"""Phase 6 checkpoint tests: loss functions.

Note: this proves the optimization math is wired correctly on dummy data,
not that the model learns anything real -- see BUILD.md Phase 6.
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.losses.contrastive import con_loss  # noqa: E402
from src.losses.total import total_loss  # noqa: E402
from src.models.model import DeepfakeDetector, load_config  # noqa: E402

B, F = 4, 2


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture
def dummy_batch(config):
    torch.manual_seed(0)
    img_size = config["spatial"]["img_size"]
    n_tokens = config["frequency"]["n_tokens"]
    dct_coeffs = config["frequency"]["dct_coeffs"]
    return {
        "rgb": torch.randn(B, F, 3, img_size, img_size),
        "dct": torch.randn(B, F, n_tokens, dct_coeffs),
        "labels": torch.tensor([0.0, 1.0, 0.0, 1.0]),
        "source_ids": ["s1", "s1", "s2", "s2"],
        "router_targets": ["clean_G0", "h264_G1", "h265_G2", "h264_G1"],
    }


def test_all_losses_finite_and_nonnegative(config, dummy_batch):
    model = DeepfakeDetector(config)
    out = model(dummy_batch["rgb"], dummy_batch["dct"])
    losses = total_loss(
        out,
        dummy_batch["labels"],
        dummy_batch["source_ids"],
        dummy_batch["router_targets"],
        config["router_clusters"],
        config["loss_weights"],
        config["contrastive"]["temperature"],
    )
    for name, value in losses.items():
        assert torch.isfinite(value), f"{name} is not finite: {value}"
        assert value.item() >= 0, f"{name} is negative: {value.item()}"


def test_con_loss_near_zero_when_same_source_well_separated():
    ids = ["s1", "s1", "s2", "s2"]
    z = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 1.0, 0, 0]])
    loss = con_loss(z, ids, temperature=0.07)
    assert loss.item() < 1e-4


def test_con_loss_large_when_same_source_forced_adversarial():
    ids = ["s1", "s1", "s2", "s2"]
    # same-source pair (0,1) forced orthogonal; cross-source pair (0,2) aligned
    z = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    loss = con_loss(z, ids, temperature=0.07)
    assert loss.item() > 5.0


def test_five_dummy_optimizer_steps_loss_decreases(config, dummy_batch):
    """Proves the optimization loop is mechanically correct -- not real
    learning. Uses BUILD.md's actual Phase 7 hyperparameters (lr=1e-4,
    wd=0.05): an arbitrarily-picked lr=1e-3 diverged on this dummy setup,
    1e-4 converges cleanly, and matching the real training config is more
    meaningful here anyway than tuning a throwaway test-only lr."""
    torch.manual_seed(0)
    model = DeepfakeDetector(config)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.05)

    totals = []
    for _ in range(5):
        opt.zero_grad()
        out = model(dummy_batch["rgb"], dummy_batch["dct"])
        losses = total_loss(
            out,
            dummy_batch["labels"],
            dummy_batch["source_ids"],
            dummy_batch["router_targets"],
            config["router_clusters"],
            config["loss_weights"],
            config["contrastive"]["temperature"],
        )
        losses["total"].backward()
        opt.step()
        totals.append(losses["total"].item())

    assert all(torch.isfinite(torch.tensor(totals)))
    assert totals[-1] < totals[0], f"loss did not trend downward: {totals}"
