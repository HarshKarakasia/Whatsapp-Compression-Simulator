# Final Summary — Robust Deepfake Detection Under Unknown Social-Network Transformation Chains

Status snapshot as of Phase 10. This document pulls every phase's artifacts into one place and fills BUILD.md's slide-placeholder sections honestly — some are real/verified, some are architecture-only, some are explicitly untrained demos. None are presented as more than they are.

**One-line status: Phases 0-10 of BUILD.md are implemented and locally verified (50 tests passing). No real model training has happened yet** — that step requires the user to run `notebooks/train_kaggle.py` on Kaggle (GPU), which Claude Code cannot execute itself. Everything downstream of training (Phase 8's benchmark, Phase 9's demo) currently runs against synthetic data or random weights, clearly labeled as such everywhere it appears.

---

## Report assets collected

| Asset | Path | What it actually shows |
|---|---|---|
| Face-crop grid | `reports/phase4_samples/face_crop_grid.png` | Real aligned face crops from 3 real files (incl. a real/deepfake pair of the same identity) |
| DCT heatmap | `reports/phase4_samples/dct_heatmap.png` | Real per-block DCT energy on a real face crop — face-shaped low-energy silhouette |
| Architecture parameter summary | `reports/architecture_summary.txt` | Real per-module parameter counts (118.6M total, 103.7M trainable) of the Phase 5 scaffold |
| Dummy-data convergence curve | `reports/phase6_dummy_convergence.png` | Real 5-step AdamW run on a **fixed dummy batch** — proves the optimization math, not real learning |
| Local dry-run loss log (chart) | `reports/phase7_dry_run_log.png` | Real 5-step dry run on **real preprocessed videos**, different random batch each step — proves the training loop doesn't crash, not a convergence curve |
| Local dry-run loss log (raw) | `reports/train_log.csv` | Same run, raw CSV |
| Benchmark table | `reports/SYNTHETIC_benchmark_table.txt` | **Synthetic** predictions (watermarked) — proves the metrics/slicing/comparison harness works, not real model performance |
| Degradation chart | `reports/SYNTHETIC_degradation_chart.png` | **Synthetic** predictions (watermarked) — illustrates the intended MoE-vs-ablation shape, not a measured result |

No HTML architecture diagram or PPT file was found anywhere in this project directory (checked explicitly, not assumed) — both are referenced in BUILD.md as already existing outside Claude Code's working scope. The sections below are written to be pasted directly into such a deck once available.

---

## Modules / Components

| Module | Files | Role |
|---|---|---|
| Transform pipeline | `src/pipeline/transform.py`, `probe.py` | Multi-codec, multi-generation ffmpeg chain; actual (not requested) ffprobe metadata |
| Data | `src/data/split.py`, `src/data/download_sample.py`, `src/data/build_real_manifest.py` | Leakage-aware source_id splitter; real FF++/Celeb-DF sample acquisition |
| Preprocessing | `src/preprocessing/frames.py`, `align.py`, `dct.py`, `pipeline.py` | Frame sampling, face alignment (mediapipe), luma block-DCT |
| Models | `src/models/spatial.py`, `frequency.py`, `fusion.py`, `router.py`, `experts.py`, `model.py` | DINOv2-shaped ViT+LoRA, frequency transformer, cross-attention fusion, soft top-2 MoE router, expert heads |
| Losses | `src/losses/classification.py`, `routing.py`, `contrastive.py`, `balance.py`, `total.py` | L_cls, L_route, L_con, L_bal, weighted combiner |
| Training | `src/train/dataset.py`, `train.py` | Source-grouped batch sampler, training loop, checkpointing |
| Evaluation | `src/eval/metrics.py`, `slicing.py`, `benchmark.py` | AUC/ACC/ECE, depth + seen/unseen slicing, MoE-vs-ablation comparison |
| Serving | `src/api/main.py`, `ui/app.py` | FastAPI `/predict`, Streamlit demo UI |

## Algorithms / Models

- **Spatial branch**: ViT-B/14-shaped encoder (embed=768, depth=12, heads=12) with rank-8 LoRA adapters on Q and V projections only (custom separate q/k/v attention, not a fused QKV matrix, specifically so LoRA can target exactly those two).
- **Frequency branch**: 4-layer, 8-head transformer over 784 luma 8×8-block DCT tokens (64 coefficients each).
- **Fusion**: cross-attention, RGB patch tokens as queries, DCT tokens as keys/values, mean-pooled to z ∈ R^768 per frame, then mean-pooled over frames to one z per video.
- **Router**: gating MLP 768→256→K (K=5 for this project's scale — see Appendix A), soft top-2 selection, renormalized.
- **Experts**: K small heads (768→256→1), gate-weighted combination.
- **Losses**: L_cls (BCE), L_route (CE against an ffprobe-derived router_target cluster), L_con (source-aware InfoNCE, τ=0.07 — same source_id = positive pair), L_bal (Switch Transformer-style load balancing).

## Implementation Details

- K=5 experts (not the full K=13 design) — Appendix A's reasoning: ~80-120 source videos spread over 13 clusters starves the router's balancing loss at this project's scale. `configs/model.yaml`'s `router_clusters` maps clean_G0 + a reduced 2-codec×2-depth set (h264_G1/G2, h265_G1/G2) to the 5 experts explicitly; anything else falls back to a default cluster.
- AdamW, lr=1e-4, wd=0.05 — BUILD.md's specified hyperparameters, also what the Phase 6 dummy-convergence proof and Phase 7 dry-run actually use.
- Face detector: mediapipe (not facenet-pytorch — its packaging is broken on current pip/setuptools, see DECISIONS.md).
- Spatial branch is currently **randomly initialized**, not loaded from pretrained DINOv2 weights — deliberately deferred to the real Kaggle training script (the ~330MB+ download crosses this project's own 50MB auto-download threshold; asked the user, confirmed random-init-for-now). Shape-identical to real ViT-B/14, so swapping in real weights is a drop-in once that happens.
- WhatsApp-hop transforms wrap a teammate's existing simulator (`HarshKarakasia/Whatsapp-Compression-Simulator`) rather than reimplementing it, per an evidence-based Phase 0 evaluation.

## Experimental Setup

- **Phase 0-2 (synthetic)**: a locally-generated 3-second test clip, fully synthetic, used to prove the transform chain (84 files, 4 codecs × 3 generations) and metadata/leakage-split pipeline mechanics without touching real data.
- **Phase 3 (real)**: 45 real videos (25 FF++ c23, 20 Celeb-DF v2), a user-approved micro-run sample — 5 real identities × their real clip + fakes, each sharing one source_id. Full multi-generation transform chain smoke-tested on 1 real video (not all 45 — ~2hr/~4GB at full scale, a cost tradeoff surfaced to and approved by the user); the full 45-video sample is probed and leakage-split at the clean/G0 level.
- **Phase 4 (real)**: preprocessing verified on 3 real files spanning real + fake FF++ and real Celeb-DF.
- **Phase 5-6 (dummy data by design)**: architecture and loss correctness proven on small dummy tensors, per BUILD.md's own test design (keeps CPU tests cheap; shape/gradient correctness doesn't need real data).
- **Phase 7 (real data, random weights)**: training loop dry-run on real preprocessed videos from the Phase 3 sample, 5 steps, random model initialization — proves the loop doesn't crash, not that anything was learned.
- **Phase 8 (synthetic, pending real checkpoint)**: benchmark harness tested against synthetic predictions since no real checkpoint exists.
- **Phase 9 (untrained demo)**: live FastAPI + Streamlit demo, explicitly labeled "untrained model — architecture demo only" in every response, since no real checkpoint exists.

## Results and Analysis

**No real model-performance results exist yet.** Everything "verified" in this project so far is *mechanical* correctness: pipeline runs without error, shapes are correct at every stage, gradients are finite, the optimization loop decreases loss on a fixed dummy batch, the benchmark harness computes correct metrics on known inputs. None of this is evidence the model can detect deepfakes — that requires real training, which requires the user to run `notebooks/train_kaggle.py` on Kaggle.

What the Phase 6 dummy-convergence chart and Phase 8 synthetic benchmark *do* show: the training math and evaluation math are both wired correctly, so a real Kaggle run has a mechanically sound pipeline to produce real results into.

## Comparison with Existing/Baseline

Not yet measurable — no real training has been run for any variant (MoE, single-head ablation, no-L_con ablation). The comparison *harness* (`src/eval/benchmark.py`) is fully implemented and tested: it's ready to consume real per-variant predictions the moment real checkpoints exist for each. Running all three ablations would require 3× real Kaggle training runs, which is future work beyond this project's current scope.

**Scaffolded, not yet run** — flagging explicitly per BUILD.md's own instruction, so this submission doesn't overclaim.

---

## Honest final status by phase

| Phase | Status |
|---|---|
| 0 — Repo scaffold | Done, real |
| 1 — Transform pipeline | Done, real (synthetic + 1 real video, full scale) |
| 2 — Metadata + split | Done, real (synthetic) |
| 3 — Real dataset | Done, real (45 real videos, user-approved budget) |
| 4 — Preprocessing | Done, real (3 real files) |
| 5 — Architecture | Done, architecture-only (random weights, shape/gradient-verified) |
| 6 — Losses | Done, architecture-only (dummy data, math-verified) |
| 7 — Training loop | Done, local dry-run only (real data, random weights) — **real training needs user action on Kaggle** |
| 8 — Evaluation | Done, synthetic-only (harness verified, no real predictions) |
| 9 — Demo | Done, untrained-demo-only (real pipeline, random weights) |
| 10 — Report packaging | This document |

**What's needed to move any of this from "scaffolded" to "real": the user runs `notebooks/train_kaggle.py` on Kaggle, downloads the resulting checkpoint, and places it at `data_cache/checkpoints/checkpoint.pt`.** At that point Phase 8's benchmark and Phase 9's demo both switch to real results automatically — no code changes needed.
