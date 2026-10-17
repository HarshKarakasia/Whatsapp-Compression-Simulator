"""DINOv2 ViT-B/14-shaped spatial branch with LoRA (rank 8, q/v) adapters.

Supports loading real pretrained DINOv2 weights (`pretrained=True`) via
timm's `vit_base_patch14_dinov2.lvd142m`, remapped from its fused-QKV
attention into this module's separate q/k/v projections (see
load_pretrained_dinov2() and DECISIONS.md's Phase "frontend session"
entry for why the split exists and how the remap works). Defaults to
`pretrained=False` (random init) so importing this module never requires
timm or a network call unless real weights are actually requested.
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


class LayerScale(nn.Module):
    """Per-channel learned scale applied to each residual branch's output
    before the residual add -- part of DINOv2's real architecture (not
    standard ViT). Must be present for frozen pretrained weights to behave
    as actually pretrained: formula and placement confirmed against
    timm's own LayerScale/Block source, not guessed."""

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


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
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, lora_rank: int, lora_alpha: float, lora_targets: list[str], layer_norm_eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.attn = Attention(embed_dim, num_heads, lora_rank, lora_alpha, lora_targets)
        self.ls1 = LayerScale(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, embed_dim))
        self.ls2 = LayerScale(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
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
        layer_norm_eps = config.get("layer_norm_eps", 1e-6)

        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size

        assert img_size % patch_size == 0
        self.n_patches_side = img_size // patch_size
        self.n_patches = self.n_patches_side**2

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                ViTBlock(embed_dim, num_heads, mlp_ratio, lora_rank, lora_alpha, lora_targets, layer_norm_eps)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

        # Freeze everything except the LoRA adapters. Earlier versions of
        # this froze submodules piecemeal (patch_embed, final norm,
        # cls_token/pos_embed, ls1/ls2 individually) and missed norm1,
        # norm2, k_proj, out_proj, and the MLP -- ~71M params that were
        # silently trainable instead of frozen, undetected until a test
        # actually checked requires_grad on every named parameter. A
        # blanket pass keyed on "lora_" in the name can't miss a submodule
        # the way a manual per-module list can.
        for name, p in self.named_parameters():
            if "lora_" not in name:
                p.requires_grad = False

        if pretrained:
            self.load_pretrained_dinov2()
        else:
            logger.warning(
                "SpatialBranch: using a randomly-initialized ViT-B/14-shaped backbone, "
                "NOT pretrained DINOv2 weights. Pass pretrained=True to load real weights."
            )

    @torch.no_grad()
    def load_pretrained_dinov2(self, model_name: str = "vit_base_patch14_dinov2.lvd142m") -> None:
        """Loads real DINOv2 weights via timm and copies them into this
        module's frozen parameters, splitting timm's fused qkv.weight
        [3*D, D] into this module's separate q_proj/k_proj/v_proj.base
        weights. timm's attention concatenates [q; k; v] along dim 0
        (confirmed against timm's VisionTransformer.Attention source, not
        assumed) -- rows [0:D]=q, [D:2D]=k, [2D:3D]=v.

        Requires timm (local import -- not a hard dependency of this
        module for the random-init / architecture-only use case) and
        network access to download the ~330MB checkpoint on first call
        (cached by timm/huggingface_hub after that).
        """
        import timm

        logger.warning(f"SpatialBranch: loading real pretrained weights ({model_name}) via timm.")
        src = timm.create_model(model_name, pretrained=True, img_size=self.img_size, num_classes=0)
        sd = src.state_dict()
        d = self.embed_dim

        self.cls_token.data.copy_(sd["cls_token"])
        self.pos_embed.data.copy_(sd["pos_embed"])
        self.patch_embed.weight.data.copy_(sd["patch_embed.proj.weight"])
        self.patch_embed.bias.data.copy_(sd["patch_embed.proj.bias"])
        self.norm.weight.data.copy_(sd["norm.weight"])
        self.norm.bias.data.copy_(sd["norm.bias"])

        for i, block in enumerate(self.blocks):
            prefix = f"blocks.{i}."
            block.norm1.weight.data.copy_(sd[prefix + "norm1.weight"])
            block.norm1.bias.data.copy_(sd[prefix + "norm1.bias"])
            block.norm2.weight.data.copy_(sd[prefix + "norm2.weight"])
            block.norm2.bias.data.copy_(sd[prefix + "norm2.bias"])
            block.ls1.gamma.data.copy_(sd[prefix + "ls1.gamma"])
            block.ls2.gamma.data.copy_(sd[prefix + "ls2.gamma"])

            qkv_w = sd[prefix + "attn.qkv.weight"]  # [3D, D]
            qkv_b = sd[prefix + "attn.qkv.bias"]  # [3D]
            q_w, k_w, v_w = qkv_w[:d], qkv_w[d : 2 * d], qkv_w[2 * d :]
            q_b, k_b, v_b = qkv_b[:d], qkv_b[d : 2 * d], qkv_b[2 * d :]

            q_base = block.attn.q_proj.base if isinstance(block.attn.q_proj, LoRALinear) else block.attn.q_proj
            k_base = block.attn.k_proj.base if isinstance(block.attn.k_proj, LoRALinear) else block.attn.k_proj
            v_base = block.attn.v_proj.base if isinstance(block.attn.v_proj, LoRALinear) else block.attn.v_proj
            q_base.weight.data.copy_(q_w)
            q_base.bias.data.copy_(q_b)
            k_base.weight.data.copy_(k_w)
            k_base.bias.data.copy_(k_b)
            v_base.weight.data.copy_(v_w)
            v_base.bias.data.copy_(v_b)

            block.attn.out_proj.weight.data.copy_(sd[prefix + "attn.proj.weight"])
            block.attn.out_proj.bias.data.copy_(sd[prefix + "attn.proj.bias"])

            block.mlp[0].weight.data.copy_(sd[prefix + "mlp.fc1.weight"])
            block.mlp[0].bias.data.copy_(sd[prefix + "mlp.fc1.bias"])
            block.mlp[2].weight.data.copy_(sd[prefix + "mlp.fc2.weight"])
            block.mlp[2].bias.data.copy_(sd[prefix + "mlp.fc2.bias"])

        del src

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b = images.shape[0]
        x = self.patch_embed(images).flatten(2).transpose(1, 2)  # [B, n_patches, embed_dim]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 1:, :]  # drop CLS, return patch tokens only
