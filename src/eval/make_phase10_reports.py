"""Phase 10: assembles report-ready loss-curve charts from real generated
data (Phase 6's fixed-batch convergence proof, Phase 7's dry-run log).
Neither is a real training curve -- both are captioned for exactly what
they demonstrate, not what they might be mistaken for."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.losses.total import total_loss
from src.models.model import DeepfakeDetector, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports"


def make_phase6_convergence_chart(out_path: Path) -> list[float]:
    """Re-runs Phase 6's fixed-batch, 5-step AdamW proof (same dummy data
    and hyperparameters as tests/test_phase6.py) and saves the curve."""
    config = load_config()
    torch.manual_seed(0)
    model = DeepfakeDetector(config)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.05)

    B, F = 4, 2
    img_size, n_tokens, dct_coeffs = config["spatial"]["img_size"], config["frequency"]["n_tokens"], config["frequency"]["dct_coeffs"]
    rgb = torch.randn(B, F, 3, img_size, img_size)
    dct = torch.randn(B, F, n_tokens, dct_coeffs)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    source_ids = ["s1", "s1", "s2", "s2"]
    router_targets = ["clean_G0", "h264_G1", "h265_G2", "h264_G1"]

    totals = []
    for _ in range(5):
        opt.zero_grad()
        out = model(rgb, dct)
        losses = total_loss(out, labels, source_ids, router_targets, config["router_clusters"], config["loss_weights"], config["contrastive"]["temperature"])
        losses["total"].backward()
        opt.step()
        totals.append(losses["total"].item())

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(range(len(totals)), totals, marker="o", color="tab:blue")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("total loss")
    ax.set_title(
        "Phase 6: fixed-batch dummy-data convergence proof\n"
        "(proves optimization math is wired correctly -- NOT real learning)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return totals


def make_phase7_dry_run_chart(log_path: Path, out_path: Path) -> None:
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    steps = [int(r["step"]) for r in rows]
    totals = [float(r["total"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(steps, totals, marker="o", color="tab:orange")
    ax.set_xlabel("step")
    ax.set_ylabel("total loss")
    ax.set_title(
        "Phase 7: local CPU dry-run log\n"
        "(different random batch each step -- proves the loop runs, NOT a convergence curve)",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    totals = make_phase6_convergence_chart(OUT_DIR / "phase6_dummy_convergence.png")
    print(f"wrote {OUT_DIR / 'phase6_dummy_convergence.png'} (totals: {totals})")

    dry_run_log = OUT_DIR / "train_log.csv"
    if dry_run_log.is_file():
        make_phase7_dry_run_chart(dry_run_log, OUT_DIR / "phase7_dry_run_log.png")
        print(f"wrote {OUT_DIR / 'phase7_dry_run_log.png'}")


if __name__ == "__main__":
    main()
