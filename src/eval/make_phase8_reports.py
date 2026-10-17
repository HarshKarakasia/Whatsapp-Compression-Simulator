"""Generates the Phase 8 benchmark table + degradation chart.

No real trained checkpoint exists yet (the Phase 7 dry-run checkpoint has
RANDOM weights -- see DECISIONS.md; real training happens on Kaggle and
hasn't been run by the user yet). Per BUILD.md's own explicit allowance
("this phase can still be built and tested against dummy prediction
arrays... clearly-labeled-synthetic sanity check if not [a checkpoint
exists]"), this generates SYNTHETIC predictions with a believable
depth-degradation pattern (MoE degrading more gracefully than the
ablations) purely to prove the benchmark harness works mechanically.

THESE ARE NOT REAL RESULTS. Every output is watermarked as synthetic.
Re-run this (or a real eval script) against actual model predictions
once data_cache/checkpoints/checkpoint.pt exists from a real Kaggle run.
"""
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.eval.benchmark import VARIANTS, comparison_table_by_depth, comparison_table_seen_unseen
from src.eval.slicing import depth_of

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports"

SEEN_ROUTER_TARGETS = {"clean_G0", "h264_G1", "h264_G2", "h265_G1", "h265_G2"}
ALL_ROUTER_TARGETS = (
    ["clean_G0"]
    + [f"{codec}_G{d}" for codec in ["h264", "h265", "vp9", "av1"] for d in (1, 2, 3)]
)

# depth-dependent "skill" per variant, purely illustrative: MoE degrades
# more gracefully than the ablations as depth increases
VARIANT_SKILL_BY_DEPTH = {
    "moe": {0: 0.95, 1: 0.90, 2: 0.85, 3: 0.78},
    "single_head": {0: 0.93, 1: 0.83, 2: 0.70, 3: 0.58},
    "no_con": {0: 0.90, 1: 0.78, 2: 0.62, 3: 0.52},
}


def make_synthetic_predictions(n_per_target: int = 20, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for router_target in ALL_ROUTER_TARGETS:
        depth = depth_of(router_target)
        for _ in range(n_per_target):
            label = rng.randint(0, 1)
            row = {"router_target": router_target, "label": label}
            for variant in VARIANTS:
                skill = VARIANT_SKILL_BY_DEPTH[variant][depth]
                # skill = P(prediction correctly separated); noisier as skill drops
                center = 0.5 + (skill - 0.5) * (1 if label == 1 else -1)
                pred = min(1.0, max(0.0, rng.gauss(center, 0.15)))
                row[f"pred_{variant}"] = pred
            rows.append(row)
    return rows


def make_degradation_chart(table: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    depths = [0, 1, 2, 3]
    for variant in VARIANTS:
        aucs = []
        for d in depths:
            match = [r for r in table if r["variant"] == variant and r["slice"] == "depth" and r["key"] == f"G{d}"]
            aucs.append(match[0]["auc"] if match else float("nan"))
        ax.plot(depths, aucs, marker="o", label=variant)

    ax.set_xlabel("transformation depth (G0-G3)")
    ax.set_ylabel("AUC")
    ax.set_xticks(depths)
    ax.set_ylim(0.4, 1.0)
    ax.set_title("SYNTHETIC sanity check -- NOT real results\n(AUC vs depth, MoE vs ablations)", fontsize=10, color="firebrick")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_table_txt(table: list[dict], out_path: Path) -> None:
    lines = [
        "SYNTHETIC SANITY CHECK -- NOT REAL RESULTS",
        "(no trained checkpoint exists yet; see DECISIONS.md Phase 8 entry)",
        "",
        f"{'variant':<14}{'slice':<13}{'key':<8}{'n':>5}{'auc':>8}{'acc':>8}{'ece':>8}",
    ]
    for r in table:
        auc_s = f"{r['auc']:.3f}" if r["auc"] == r["auc"] else "nan"  # nan check
        lines.append(f"{r['variant']:<14}{r['slice']:<13}{r['key']:<8}{r['n']:>5}{auc_s:>8}{r['acc']:>8.3f}{r['ece']:>8.3f}")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = make_synthetic_predictions()

    depth_table = comparison_table_by_depth(rows)
    seen_unseen_table = comparison_table_seen_unseen(rows, SEEN_ROUTER_TARGETS)

    write_table_txt(depth_table + seen_unseen_table, OUT_DIR / "SYNTHETIC_benchmark_table.txt")
    make_degradation_chart(depth_table, OUT_DIR / "SYNTHETIC_degradation_chart.png")

    print(f"wrote {OUT_DIR / 'SYNTHETIC_benchmark_table.txt'}")
    print(f"wrote {OUT_DIR / 'SYNTHETIC_degradation_chart.png'}")


if __name__ == "__main__":
    main()
