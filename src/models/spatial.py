"""DINOv2 ViT-B/14-shaped spatial branch with LoRA (rank 8, q/v) adapters.

Phase 5 scaffold: builds the architecture at ViT-B/14's exact dimensions
with randomly-initialized weights (see DECISIONS.md for why -- the ~330MB+
pretrained DINOv2 checkpoint download was deliberately deferred to Phase 7's
actual Kaggle training script, where it's the only place it matters for
this ~60%-depth project). Base weights are frozen; only the LoRA A/B
matrices are trainable, matching how LoRA fine-tuning is meant to be wired
regardless of what the base weights are initialized to.
"""
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank adapter:
    out = base(x) + (alpha/rank) * B(A(x)). B starts at zero so the
    adapter is a no-op until trained."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_features, out_features = base.in_features, base.out_features
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))


class Attention(nn.Module):
    """Multi-head self-attention with separate q/k/v projections (not a
    fused qkv matrix) so LoRA can target q and v specifically."""

    def __init__(self, embed_dim: int, num_heads: int, lora_rank: int, lora_alpha: float, lora_targets: list[str]):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        q_proj = nn.Linear(embed_dim, embed_dim)
        k_proj = nn.Linear(embed_dim, embed_dim)
        v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = LoRALinear(q_proj, lora_rank, lora_alpha) if "q_proj" in lora_targets else q_proj
        self.k_proj = LoRALinear(k_proj, lora_rank, lora_alpha) if "k_proj" in lora_targets else k_proj
        self.v_proj = LoRALinear(v_proj, lora_rank, lora_alpha) if "v_proj" in lora_targets else v_proj
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(b, n, d)
        return self.out_proj(attn_out)


class ViTBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, lora_rank: int, lora_alpha: float, lora_targets: list[str]):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, lora_rank, lora_alpha, lora_targets)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SpatialBranch(nn.Module):
    """ViT-B/14-shaped encoder. forward(images [B,3,H,W]) -> patch tokens
    [B, n_patches, embed_dim] (CLS token dropped after encoding -- fusion
    cross-attends over the patch grid, not a single pooled vector)."""

    def __init__(self, config: dict, pretrained: bool = False):
        super().__init__()
        embed_dim = config["embed_dim"]
        patch_size = config["patch_size"]
        img_size = config["img_size"]
        depth = config["depth"]
        num_heads = config["num_heads"]
        mlp_ratio = config["mlp_ratio"]
        lora_rank = config["lora_rank"]
        lora_alpha = config["lora_alpha"]
        lora_targets = config["lora_targets"]

        if pretrained:
            raise NotImplementedError(
                "pretrained DINOv2 weight loading is deferred to Phase 7's training "
                "script -- see DECISIONS.md. This scaffold only supports pretrained=False."
            )
        logger.warning(
            "SpatialBranch: using a randomly-initialized ViT-B/14-shaped backbone, "
            "NOT pretrained DINOv2 weights. Real pretrained weights must be loaded "
            "before this is used for actual training (Phase 7)."
        )

        assert img_size % patch_size == 0
        self.n_patches_side = img_size // patch_size
        self.n_patches = self.n_patches_side**2

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for p in (self.cls_token, self.pos_embed):
            p.requires_grad = False  # part of the frozen "pretrained" backbone

        self.blocks = nn.ModuleList(
            [ViTBlock(embed_dim, num_heads, mlp_ratio, lora_rank, lora_alpha, lora_targets) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        for p in self.patch_embed.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b = images.shape[0]
        x = self.patch_embed(images).flatten(2).transpose(1, 2)  # [B, n_patches, embed_dim]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 1:, :]  # drop CLS, return patch tokens only
