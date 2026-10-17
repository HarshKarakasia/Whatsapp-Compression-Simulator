"""Training loop (Phase 7). This is a LOCAL DRY-RUN ONLY -- it proves the
loop doesn't crash and the optimization math is wired correctly, not that
the model learns anything real. Real training needs Kaggle/Colab (M2 Mac:
OOM risk, MPS kernel gaps on this architecture) -- see
notebooks/train_kaggle.py and DECISIONS.md's Phase 7 entry.

Logs to a local CSV only (no W&B / experiment-tracking credential) -- see
DECISIONS.md for why.
"""
import csv
import time
from pathlib import Path

import torch

from src.losses.total import total_loss
from src.models.model import DeepfakeDetector
from src.models.model import load_config as load_model_config
from src.preprocessing.pipeline import load_config as load_preprocessing_config
from src.preprocessing.pipeline import preprocess_video
from src.train.dataset import SourceGroupedBatcher, load_real_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_batch_tensors(batch_rows: list[dict], preprocessing_config: dict) -> dict:
    """Note: the full model is slow on CPU at real scale (measured ~140s/step
    at the production n_frames=16, B=4 -- 12-layer ViT-B/14 + 4-layer/784-
    token frequency transformer + cross-attention fusion, all on B*F=64
    sequences). The dry-run caller should pass a preprocessing_config with a
    reduced n_frames (see train()'s default), matching Phase 5/6's own
    precedent of using small dummy shapes to keep CPU tests cheap -- this
    is a local mechanical check, not a realistic-scale run. See DECISIONS.md."""
    rgb_list, dct_list = [], []
    for row in batch_rows:
        out = preprocess_video(REPO_ROOT / row["file"], preprocessing_config)
        rgb_list.append(out["rgb"])
        dct_list.append(out["dct"])
    return {
        "rgb": torch.stack(rgb_list),  # [B,F,3,H,W]
        "dct": torch.stack(dct_list),  # [B,F,n_tokens,dct_coeffs]
        "labels": torch.tensor([1.0 if r["label"] == "fake" else 0.0 for r in batch_rows]),
        "source_ids": [r["source_id"] for r in batch_rows],
        "router_targets": [r["router_target"] for r in batch_rows],
    }


def save_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, model_config: dict, save_optimizer_state: bool = True
) -> None:
    """save_optimizer_state=False skips AdamW's moment buffers (2x the
    trainable-param count), which otherwise dominate checkpoint size --
    e.g. ~1.3GB vs ~450MB for this architecture. Off by default for the
    dry run, since there's nothing meaningful to resume; on for real
    training checkpoints where resuming matters."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"model_state_dict": model.state_dict(), "step": step, "config": model_config}
    if save_optimizer_state:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def train(
    n_steps: int,
    sources_per_batch: int = 2,
    views_per_source: int = 2,
    checkpoint_path: Path | str | None = None,
    log_path: Path | str | None = None,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    seed: int = 0,
    n_frames_override: int | None = 4,
    save_optimizer_state: bool = False,
    device: str = "cpu",
    checkpoint_every: int | None = None,
    resume_from: Path | str | None = None,
    manifest_rows: list[dict] | None = None,
) -> list[dict]:
    """n_frames_override: local CPU dry-run is too slow at the production
    n_frames (16) -- measured ~140s/step full-scale vs ~35s/step at 4
    frames. Pass None to use configs/preprocessing.yaml's real value (what
    Kaggle training should do, where device="cuda" makes full scale fast).

    checkpoint_every: if set, also save every N steps (not just at the
    end) -- matters on Kaggle, whose sessions have a ~9-12hr execution
    limit; resume_from picks a prior checkpoint back up via `load_checkpoint`.
    manifest_rows: pass explicitly (e.g. a bigger real-training manifest on
    Kaggle); defaults to this repo's Phase 3 sample via load_real_manifest().
    """
    model_config = load_model_config()
    preprocessing_config = load_preprocessing_config()
    if n_frames_override is not None:
        preprocessing_config = {**preprocessing_config, "n_frames": n_frames_override}
    rows = manifest_rows if manifest_rows is not None else load_real_manifest()

    torch.manual_seed(seed)
    model = DeepfakeDetector(model_config).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    start_step = 0
    if resume_from is not None:
        ckpt = load_checkpoint(Path(resume_from), model, opt)
        start_step = ckpt["step"]

    batcher = SourceGroupedBatcher(rows, sources_per_batch, views_per_source, seed=seed)

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else REPO_ROOT / "data_cache" / "checkpoints" / "dry_run_checkpoint.pt"
    log_path = Path(log_path) if log_path else REPO_ROOT / "reports" / "train_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_rows = []
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "total", "cls", "route", "con", "bal", "elapsed_s"])

        for step in range(start_step, start_step + n_steps):
            t0 = time.time()
            batch_rows = batcher.sample_batch()
            batch = build_batch_tensors(batch_rows, preprocessing_config)
            batch["rgb"] = batch["rgb"].to(device)
            batch["dct"] = batch["dct"].to(device)
            batch["labels"] = batch["labels"].to(device)

            opt.zero_grad()
            out = model(batch["rgb"], batch["dct"])
            losses = total_loss(
                out,
                batch["labels"],
                batch["source_ids"],
                batch["router_targets"],
                model_config["router_clusters"],
                model_config["loss_weights"],
                model_config["contrastive"]["temperature"],
            )
            losses["total"].backward()
            opt.step()

            elapsed = time.time() - t0
            row = {k: v.item() for k, v in losses.items()}
            row["step"] = step
            row["elapsed_s"] = elapsed
            log_rows.append(row)
            writer.writerow([step, row["total"], row["cls"], row["route"], row["con"], row["bal"], elapsed])
            f.flush()

            if checkpoint_every and (step + 1) % checkpoint_every == 0:
                save_checkpoint(checkpoint_path, model, opt, step + 1, model_config, save_optimizer_state=save_optimizer_state)

    save_checkpoint(checkpoint_path, model, opt, start_step + n_steps, model_config, save_optimizer_state=save_optimizer_state)
    return log_rows


if __name__ == "__main__":
    logs = train(n_steps=5)
    for row in logs:
        print(f"step {row['step']}: total={row['total']:.4f} ({row['elapsed_s']:.1f}s)")
    print(f"dry run complete, {len(logs)} steps logged")
