"""Full forward pass: rgb + dct -> p_fake logit, end to end.

rgb [B,F,3,H,W] and dct [B,F,n_tokens,dct_coeffs] go through the spatial
and frequency branches per-frame, fuse via cross-attention to a per-frame
z, mean-pool over frames to a per-video z, then route to gate-weighted
experts.
"""
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.models.experts import Experts
from src.models.frequency import FrequencyBranch
from src.models.fusion import FusionModule
from src.models.router import Router
from src.models.spatial import SpatialBranch

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | str | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "model.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


class DeepfakeDetector(nn.Module):
    def __init__(self, config: dict | None = None, pretrained_spatial: bool = False):
        super().__init__()
        config = config or load_config()
        self.config = config
        embed_dim = config["spatial"]["embed_dim"]

        self.spatial = SpatialBranch(config["spatial"], pretrained=pretrained_spatial)
        self.frequency = FrequencyBranch(config["frequency"])
        self.fusion = FusionModule(config["fusion"])
        self.router = Router(embed_dim, config["num_experts"], config["router"]["hidden_dim"], config["top_k"])
        self.experts = Experts(embed_dim, config["num_experts"], config["experts"]["hidden_dim"])

    def forward(self, rgb: torch.Tensor, dct: torch.Tensor) -> dict:
        b, f = rgb.shape[0], rgb.shape[1]

        rgb_flat = rgb.reshape(b * f, *rgb.shape[2:])
        dct_flat = dct.reshape(b * f, *dct.shape[2:])

        rgb_tokens = self.spatial(rgb_flat)  # [B*F, n_patches, D]
        dct_tokens = self.frequency(dct_flat)  # [B*F, n_tokens, D]
        z_frame = self.fusion(rgb_tokens, dct_tokens)  # [B*F, D]

        z_video = z_frame.reshape(b, f, -1).mean(dim=1)  # [B, D]

        router_out = self.router(z_video)
        logits = self.experts(z_video, router_out["gate_weights"])  # [B, 1]

        return {
            "logits": logits,
            "gate_weights": router_out["gate_weights"],
            "router_probs": router_out["probs"],
            "router_logits": router_out["logits"],
            "z_video": z_video,
        }
