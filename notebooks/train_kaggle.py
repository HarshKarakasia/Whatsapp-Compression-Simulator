"""Real training on Kaggle (free T4x2, 30 hrs/week). Claude Code cannot
execute this itself -- upload and run it manually, per BUILD.md Phase 7.

=== Setup (one-time) ===
1. Zip this repo's `src/`, `configs/`, `data_cache/models/`,
   `data_cache/raw/`, and `data_cache/manifest/` directories together
   (NOT `data_cache/chains/` or `data_cache/synthetic/` -- not needed for
   training, just adds upload size). ~73MB total.
2. On kaggle.com: New Dataset -> upload that zip -> note its slug, e.g.
   "your-username/deepfake-detection-fyp".
3. New Notebook -> Add Input -> attach that dataset.
4. Notebook settings -> Accelerator -> GPU T4 x2.
5. Upload this file as the notebook's code (or paste its contents into
   a cell), fix REPO_ROOT below to match your dataset's mount path, run.
6. When done, download `/kaggle/working/checkpoint.pt` and
   `/kaggle/working/train_log.csv`.
7. Place checkpoint.pt in this repo at `data_cache/checkpoints/checkpoint.pt`
   (the default Phase 8/9 look for) before continuing to those phases.

=== What this runs ===
The exact same src/train/train.py used for the local CPU dry-run, just
with device="cuda", the real preprocessing config (n_frames=16, not the
dry-run's reduced value), a larger batch, and periodic checkpointing
(Kaggle sessions have a ~9-12hr execution limit -- see checkpoint_every
and resume_from below if a run needs to continue across sessions).

=== Experiment tracking ===
Logs to a local CSV only, no W&B or other tracking service wired in --
see DECISIONS.md's Phase 7 entry for why (avoids needing a new
credential; ask Claude Code to add W&B support later if you want it).
"""
import sys
from pathlib import Path

# --- fix this to your uploaded dataset's mount path ---
REPO_ROOT = Path("/kaggle/input/deepfake-detection-fyp")
sys.path.insert(0, str(REPO_ROOT))

import subprocess  # noqa: E402

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mediapipe"], check=True)

import torch  # noqa: E402

from src.train.train import train  # noqa: E402

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected -- check the notebook's Accelerator setting.")

    logs = train(
        n_steps=200,  # tune based on observed step time x 30hr/week budget
        sources_per_batch=4,
        views_per_source=4,
        n_frames_override=None,  # use the real configs/preprocessing.yaml value (16)
        device=device,
        checkpoint_path="/kaggle/working/checkpoint.pt",
        log_path="/kaggle/working/train_log.csv",
        save_optimizer_state=True,
        checkpoint_every=50,
        # resume_from="/kaggle/working/checkpoint.pt",  # uncomment to continue a prior run
    )
    print(f"done: {len(logs)} steps. Download /kaggle/working/checkpoint.pt and train_log.csv.")
