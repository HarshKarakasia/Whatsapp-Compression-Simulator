"""Phase 5 checkpoint tests: architecture scaffold, forward/backward shapes."""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.models.frequency import FrequencyBranch  # noqa: E402
from src.models.fusion import FusionModule  # noqa: E402
from src.models.model import DeepfakeDetector, load_config  # noqa: E402
from src.models.spatial import SpatialBranch  # noqa: E402

B, F = 2, 4  # kept light on CPU/MPS, per BUILD.md


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def dummy_batch(config):
    torch.manual_seed(0)
    img_size = config["spatial"]["img_size"]
    n_tokens = config["frequency"]["n_tokens"]
    dct_coeffs = config["frequency"]["dct_coeffs"]
    rgb = torch.randn(B, F, 3, img_size, img_size)
    dct = torch.randn(B, F, n_tokens, dct_coeffs)
    return rgb, dct


def test_spatial_branch_output_shape(config):
    model = SpatialBranch(config["spatial"])
    x = torch.randn(B * F, 3, config["spatial"]["img_size"], config["spatial"]["img_size"])
    out = model(x)
    n_patches = (config["spatial"]["img_size"] // config["spatial"]["patch_size"]) ** 2
    assert out.shape == (B * F, n_patches, config["spatial"]["embed_dim"])


def test_frequency_branch_output_shape(config):
    model = FrequencyBranch(config["frequency"])
    x = torch.randn(B * F, config["frequency"]["n_tokens"], config["frequency"]["dct_coeffs"])
    out = model(x)
    assert out.shape == (B * F, config["frequency"]["n_tokens"], config["frequency"]["embed_dim"])


def test_fusion_output_shape(config):
    model = FusionModule(config["fusion"])
    embed_dim = config["fusion"]["embed_dim"]
    n_patches = (config["spatial"]["img_size"] // config["spatial"]["patch_size"]) ** 2
    rgb_tokens = torch.randn(B * F, n_patches, embed_dim)
    dct_tokens = torch.randn(B * F, config["frequency"]["n_tokens"], embed_dim)
    z = model(rgb_tokens, dct_tokens)
    assert z.shape == (B * F, embed_dim)


def test_full_model_forward_shapes(config, dummy_batch):
    model = DeepfakeDetector(config)
    rgb, dct = dummy_batch
    out = model(rgb, dct)
    assert out["logits"].shape == (B, 1)
    assert out["gate_weights"].shape == (B, config["num_experts"])
    assert out["z_video"].shape == (B, config["spatial"]["embed_dim"])


def test_exactly_top_k_experts_nonzero(config, dummy_batch):
    model = DeepfakeDetector(config)
    rgb, dct = dummy_batch
    out = model(rgb, dct)
    nonzero_counts = (out["gate_weights"] > 0).sum(dim=-1)
    assert torch.all(nonzero_counts == config["top_k"])
    # selected weights sum to 1 per sample
    assert torch.allclose(out["gate_weights"].sum(dim=-1), torch.ones(B), atol=1e-5)


def test_backward_pass_no_nan_grads(config, dummy_batch):
    model = DeepfakeDetector(config)
    rgb, dct = dummy_batch
    out = model(rgb, dct)
    loss = out["logits"].sum()
    loss.backward()

    checked = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} has no grad"
            assert not torch.isnan(p.grad).any(), f"{name} has NaN grad"
            checked += 1
    assert checked > 0


def test_architecture_summary_report_exists():
    path = REPO_ROOT / "reports" / "architecture_summary.txt"
    assert path.is_file()
    text = path.read_text()
    assert "TOTAL" in text
    assert "spatial" in text and "frequency" in text and "router" in text and "experts" in text
