"""Generates reports/architecture_summary.txt: per-module parameter counts
for the Phase 5 scaffold (shape-correctness proof, not a trained model)."""
from pathlib import Path

import torch.nn as nn

from src.models.model import DeepfakeDetector, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def _count(module: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    config = load_config()
    model = DeepfakeDetector(config)

    lines = [
        "Phase 5 architecture parameter summary",
        "(scaffold -- random init, not trained; see DECISIONS.md)",
        "",
        f"{'module':<12} {'total params':>14} {'trainable params':>18}",
        "-" * 46,
    ]
    submodules = {
        "spatial": model.spatial,
        "frequency": model.frequency,
        "fusion": model.fusion,
        "router": model.router,
        "experts": model.experts,
    }
    for name, module in submodules.items():
        total, trainable = _count(module)
        lines.append(f"{name:<12} {total:>14,} {trainable:>18,}")

    total, trainable = _count(model)
    lines.append("-" * 46)
    lines.append(f"{'TOTAL':<12} {total:>14,} {trainable:>18,}")
    lines.append("")
    lines.append(f"num_experts (K) = {config['num_experts']}, top_k = {config['top_k']}")
    lines.append(
        "spatial branch is frozen except LoRA adapters (rank="
        f"{config['spatial']['lora_rank']}, targets={config['spatial']['lora_targets']})"
    )

    out_path = REPO_ROOT / "reports" / "architecture_summary.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
