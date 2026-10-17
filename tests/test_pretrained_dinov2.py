"""Verifies SpatialBranch.load_pretrained_dinov2() against timm's own
forward pass. Needs timm + the real ~330MB DINOv2 checkpoint (network on
first run, cached by huggingface_hub after) -- skipped unless that cache
already exists, so the routine suite stays fast and network-independent.
Run explicitly after a fresh `timm.create_model('vit_base_patch14_dinov2.
lvd142m', pretrained=True)` call to exercise it for real.

Note: if this ever needs to run on a fresh machine, huggingface_hub's Xet
transfer backend hung indefinitely in this sandbox (plain HTTPS worked
fine) -- set HF_HUB_DISABLE_XET=1 if a download stalls. See DECISIONS.md.
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--timm--vit_base_patch14_dinov2.lvd142m"

pytestmark = pytest.mark.skipif(not HF_CACHE.is_dir(), reason="pretrained DINOv2 weights not cached locally")


def test_pretrained_load_matches_timm_forward_pass_exactly():
    import timm

    from src.models.model import load_config
    from src.models.spatial import SpatialBranch

    config = load_config()
    sb = SpatialBranch(config["spatial"], pretrained=True)
    sb.eval()

    src = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, img_size=config["spatial"]["img_size"], num_classes=0)
    src.eval()

    torch.manual_seed(0)
    img = torch.randn(1, 3, config["spatial"]["img_size"], config["spatial"]["img_size"])
    with torch.no_grad():
        out_custom = sb(img)
        out_timm = src.forward_features(img)

    assert out_custom.shape == (1, sb.n_patches, sb.embed_dim)
    max_diff = (out_custom - out_timm[:, 1:, :]).abs().max().item()
    assert max_diff < 1e-5, f"pretrained-loaded forward pass diverges from timm by {max_diff}"


def test_pretrained_backbone_still_frozen_except_lora():
    from src.models.model import load_config
    from src.models.spatial import SpatialBranch

    config = load_config()
    sb = SpatialBranch(config["spatial"], pretrained=True)

    for name, p in sb.named_parameters():
        if "lora_" in name:
            assert p.requires_grad, f"{name} should be trainable"
        else:
            assert not p.requires_grad, f"{name} should be frozen"
