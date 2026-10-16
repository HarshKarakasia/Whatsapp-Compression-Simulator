"""Generates the Phase 4 visual-proof artifacts: an annotated grid of
aligned face crops across a few real files, and a DCT energy heatmap for
one of them. Saved to reports/phase4_samples/."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.preprocessing.pipeline import preprocess_video

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "reports" / "phase4_samples"

SAMPLE_FILES = [
    "data_cache/raw/ffpp/original/000.mp4",
    "data_cache/raw/ffpp/deepfakes/000_003.mp4",
    "data_cache/raw/celebdf/id0_0000.mp4",
]
FRAMES_PER_ROW = 8


def make_face_grid(results: dict[str, dict], out_path: Path) -> None:
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, FRAMES_PER_ROW, figsize=(FRAMES_PER_ROW * 1.5, n_rows * 1.7))
    frame_idx = np.linspace(0, 15, FRAMES_PER_ROW).astype(int)

    for row, (name, out) in enumerate(results.items()):
        rgb = out["rgb"]  # [16,3,224,224] in [0,1]
        for col, fi in enumerate(frame_idx):
            ax = axes[row, col]
            img = rgb[fi].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
            if row == 0:
                ax.set_title(f"frame {fi}", fontsize=7)

    fig.suptitle("Phase 4: aligned face crops (scale=1.3, size=224)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_dct_heatmap(dct: torch.Tensor, block: int, grid_side: int, out_path: Path, title: str) -> None:
    # mean absolute coefficient magnitude per block, averaged over all 16 frames
    energy = dct.abs().mean(dim=(0, 2)).numpy()  # [784]
    heatmap = energy.reshape(grid_side, grid_side)

    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(heatmap, cmap="magma")
    ax.set_title(f"Phase 4: DCT block energy -- {title}", fontsize=10)
    ax.set_xlabel(f"{grid_side}x{grid_side} blocks of {block}x{block} px")
    fig.colorbar(im, ax=ax, label="mean |DCT coeff|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {Path(f).name: preprocess_video(f) for f in SAMPLE_FILES}

    make_face_grid(results, OUT_DIR / "face_crop_grid.png")

    first_name, first_out = next(iter(results.items()))
    grid_side = int(round(first_out["dct"].shape[1] ** 0.5))
    make_dct_heatmap(first_out["dct"], block=8, grid_side=grid_side, out_path=OUT_DIR / "dct_heatmap.png", title=first_name)

    print(f"wrote {OUT_DIR / 'face_crop_grid.png'}")
    print(f"wrote {OUT_DIR / 'dct_heatmap.png'}")


if __name__ == "__main__":
    main()
