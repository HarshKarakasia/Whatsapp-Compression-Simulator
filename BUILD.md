# BUILD.md — Robust Deepfake Detection (MoE) — Implementation Spec for Claude Code

Project: Robust Deepfake Detection Under Unknown Social-Network Transformation Chains
Team: SupaCodas (Amey Kulkarni, Ananya Patel, Harsh Karakasia, Rishabh Mody) — DJSCE AIML, Guide: Dr. Nilesh Rathod
Target: ~60% implementation depth, built as independently-presentable slices, local-first (M2 Mac, no CUDA), cloud only where unavoidable.

---

## 0. How to run this with Claude Code (read this first — saves tokens)

- **One phase per session/prompt.** Give Claude Code only the relevant `### Phase N` section, not the whole file. Say: *"Implement Phase N of BUILD.md. Run the Phase N tests. Stop after — do not start Phase N+1."*
- **Read `PROGRESS.md` first** in every new session before doing anything — it says what's done and what's next. Update it at the end of every phase.
- **Commit after every phase** with message `phase N: <summary>`. Small commits = cheap diffs = cheap context on resume.
- **Never let a phase cascade into the next automatically.** Each phase ends with a hard stop and a visible test result.
- If a session runs low on context mid-phase, stop, update `PROGRESS.md` with exactly what's left in that phase, commit, and resume in a fresh session.

---

## 1. Non-negotiable rules (apply to every phase)

1. **No fake-data-presented-as-real.** Synthetic/dummy data used for testing a module must be clearly labeled (filenames, logs, `PROGRESS.md` notes) as synthetic. Never report synthetic-data numbers as if they were real FaceForensics++/Celeb-DF results.
2. **Stop and ask before:**
   - downloading any dataset (Kaggle, HuggingFace model weights beyond what's cached, anything >50MB)
   - needing any credential or env var (Kaggle API key, HF token, W&B key, etc.) — name exactly which one and why
   - any step that implies cloud/notebook execution (Kaggle, Colab) — Claude Code cannot run these itself; it should produce a ready-to-run script/notebook and wait for the user to run it and report back
   - any step estimated to take >10 min of local compute or generate >1GB of files
3. **Every phase ends with a passing test before moving on.** No "looks right" — run it, show the output.
4. **Every phase must leave something demoable**, even if no later phase ever gets built.
5. **Config over hardcoding.** Codec list, depth, resolution recipes, number of MoE experts (K), loss weights — all in `configs/*.yaml`, never hardcoded in source.

---

## 2. Repo layout (create in Phase 0)

```
deepfake-detection-fyp/
├── BUILD.md
├── PROGRESS.md
├── DECISIONS.md
├── configs/
│   ├── transform.yaml        # codecs, depths, resolution/bitrate/fps/gop recipes
│   ├── model.yaml            # K (experts), dims, LoRA rank, loss weights
│   └── paths.yaml            # local data paths (gitignored values)
├── external/
│   └── whatsapp-compression-simulator/   # Harsh's repo, if integrated (Phase 0)
├── src/
│   ├── pipeline/              # ffmpeg transform chain + ffprobe metadata
│   ├── data/                  # manifest, leakage-aware splitter
│   ├── preprocessing/         # frame extraction, face align, DCT
│   ├── models/                # dinov2 branch, freq transformer, fusion, router, experts
│   ├── losses/
│   ├── train/
│   ├── eval/
│   └── api/                   # FastAPI backend
├── ui/                        # Streamlit app
├── tests/                     # one test module per phase
├── data_cache/                 # gitignored — synthetic + real videos live here
└── reports/                    # figures/tables for the written report
```

---

### Phase 0 — Repo scaffold + evaluate Harsh's WhatsApp simulator

**Goal:** Working repo skeleton; a factual decision on whether to integrate `HarshKarakasia/Whatsapp-Compression-Simulator`.

**Tasks:**
- Create the directory layout above, `git init`, `.gitignore` for `data_cache/`, model weights, `.env`.
- Generate a synthetic test clip (no download): `ffmpeg -f lavfi -i testsrc=duration=3:size=320x240:rate=15 -f lavfi -i sine=frequency=1000:duration=3 -shortest data_cache/synthetic/test_input.mp4`
- Clone `https://github.com/HarshKarakasia/Whatsapp-Compression-Simulator` into `external/`.
- Read its code/README. Determine:
  - Does it take a video in, video out, and is it scriptable (no GUI dependency)?
  - Does it **expose or log** the actual ffmpeg parameters used (codec, resolution, bitrate, CRF, GOP) per hop, or is it a black box that only returns a file?
  - Run it on `test_input.mp4`. Does it succeed without manual intervention?
- Write the decision to `DECISIONS.md`:
  - **If it runs and exposes parameters:** integrate it as the "WhatsApp hop" implementation inside `src/pipeline/` (wrap, don't fork-and-edit blindly).
  - **If it runs but is a black box:** wrap it, then bolt your own `ffprobe` measurement step on its output (Phase 2 already requires this anyway).
  - **If it doesn't run cleanly:** note the exact failure, keep it as a reference for its resize/bitrate *recipe values* only, and implement the hop yourself in Phase 1.

**Checkpoint:** none — everything in this phase is local and offline.

**Tests:**
- `pytest tests/test_phase0.py` asserts: repo dirs exist, `test_input.mp4` exists and is a valid video (`ffprobe` returns a stream), `DECISIONS.md` contains a non-empty "Phase 0 decision" section.

**Deliverable:** a repo that runs, plus a written, evidence-based call on reusing a teammate's code — itself a presentable piece of engineering judgement for the report.

---

### Phase 1 — FFmpeg multi-codec, multi-generation transformation pipeline

**Goal:** `src/pipeline/transform.py` that takes one source video and produces the full G1→G2→G3 chain tree across H.264/H.265/VP9/AV1.

**Tasks:**
- `configs/transform.yaml`: codecs (`h264,h265,vp9,av1` → `libx264,libx265,libvpx-vp9,libsvtav1`), per-hop recipes (resolution/fps/bitrate/gop for hop 1/2/3, matching the spec: 720p/30fps/2.4Mbps → 540p/30fps/1.1Mbps → 360p/25fps/0.4Mbps).
- `single_hop(input_path, codec, recipe) -> output_path`: one ffmpeg re-encode call.
- `build_chain_tree(source_path, depth=3) -> list[output_paths]`: recursively applies all 4 codecs at each depth, producing the 4 + 16 + 64 = 84 files-per-source structure (configurable depth/codec-count for smaller runs).
- Integrate Harsh's module per the Phase 0 decision for at least one hop type.

**Checkpoint:** none — runs entirely on the synthetic clip from Phase 0.

**Tests:**
- Run the full chain on `test_input.mp4`. Assert exactly 84 output files exist (or the configured count), every file plays (`ffprobe` succeeds on each), no two files are byte-identical.
- Spot-check: assert hop-3 files are smaller than hop-1 files for the same lineage (sanity on bitrate decay).

**Deliverable:** a folder of 85 real, playable video variants from one source clip — visually demoable as "before/after" pairs even with zero ML built yet.

---

### Phase 2 — FFprobe metadata + leakage-aware splitter

**Goal:** `metadata.jsonl` per transformed file with actual (not requested) stream properties, plus a `source_id`-level train/val/test splitter that provably leaks nothing.

**Tasks:**
- `src/pipeline/probe.py`: run `ffprobe -show_streams` on each file from Phase 1, extract actual resolution/fps/bitrate/GOP, assign `router_target = last_codec + "_G" + depth` (plus a `clean_G0` class).
- `src/data/split.py`: group by `source_id`, split 70/15/15, assert no `source_id` appears in two partitions. For FF++, real clip and all its fakes share one `source_id`.
- Write everything to `data_cache/manifest/metadata.jsonl`.

**Checkpoint:** none.

**Tests:**
- Run on Phase 1's synthetic chain (assign it a fake `source_id`). Assert `metadata.jsonl` has 85 rows, every `router_target` is well-formed, and the splitter raises if a `source_id` is forced into two splits.

**Deliverable:** a clean metadata table + a leakage-check report (0 leaked ids) — a concrete data-integrity artifact for the report's methodology section.

---

### Phase 3 — Real dataset acquisition

**Goal:** Pull real FF++ (c23) and Celeb-DF v2 source videos.

**STOP AND ASK before any part of this phase runs.** Specifically confirm with the user:
1. Kaggle API credentials: does `~/.kaggle/kaggle.json` (or `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars) already exist? If not, the user must create a Kaggle API token and provide it — Claude Code must not proceed without it.
2. **Size budget**: how many source videos to pull (recommend 80–120 total across both datasets for a micro-run — confirm this number with the user, don't assume it).
3. Estimated download size/time for that budget — state it and get a go-ahead.
4. Confirm it's OK to write that much to `data_cache/raw/` (check available disk first).

**Tasks (only after explicit go-ahead):**
- `kaggle datasets download` for `xdxd003/ff-c23` and `reubensuju/celeb-df-v2`, limited to the agreed sample count.
- Run the Phase 1 pipeline + Phase 2 metadata/split on the real sample.

**Tests:**
- Assert the downloaded count matches the agreed budget, assert the Phase 1/2 pipeline runs on real files without errors.

**Deliverable:** a real, leakage-checked, multi-generation dataset manifest — the first point where everything downstream is "real" rather than synthetic.

---

### Phase 4 — Preprocessing: frames, face align, DCT

**Goal:** `src/preprocessing/` turns a video into model-ready tensors.

**Tasks:**
- `extract_frames(video_path, n=16) -> frames`: uniform sampling.
- `align_crop(frames, scale=1.3, size=224)`: face detect + align (use a lightweight, pip-installable detector — e.g. `mediapipe` or `facenet-pytorch`'s MTCNN; do not assume a specific one is pre-approved, pick one and note it in `DECISIONS.md`).
- `block_dct(frames, block=8)`: luma-channel 8×8 block DCT → 784 tokens × 64 coeffs.

**Checkpoint:** none if using the already-downloaded data; if the chosen face detector needs its own model weights download, treat that as a small (<50MB) download and just log it — no need to stop, but note it in `PROGRESS.md`.

**Tests:**
- Run on 2–3 files from Phase 1/3. Assert output shapes: RGB `[16,3,224,224]`, DCT `[16,784,64]`. Save one annotated face-crop grid and one DCT heatmap image to `reports/phase4_samples/` for visual proof.

**Deliverable:** a visual grid of aligned face crops + DCT heatmaps — concrete, presentable even in isolation.

---

### Phase 5 — Model architecture scaffold (forward pass only, no training)

**Goal:** Every architectural piece from the spec, verified via shape-correct forward/backward passes on dummy tensors — not trained.

**Tasks:**
- `src/models/spatial.py`: DINOv2 ViT-B/14 branch with LoRA (rank 8, q/v) adapters.
  - **If downloading pretrained DINOv2 weights fails** (network restrictions in this environment), fall back to a randomly-initialized transformer with identical output shapes for testing, and clearly log this. Real pretrained weights must be loaded when running locally or on Kaggle with full network access.
- `src/models/frequency.py`: 4-layer, 8-head transformer over DCT tokens.
- `src/models/fusion.py`: cross-attention, RGB queries / DCT keys-values, pooled to `z ∈ R^768`.
- `src/models/router.py`: gating MLP 768→256→K (K from `configs/model.yaml`, default **5 for the micro-run**, see Appendix for why), soft top-2.
- `src/models/experts.py`: K small heads, 768→256→1, gate-weighted combination.
- `configs/model.yaml`: set `num_experts: 5` (reduced) with a comment explaining the full `K=13` design is documented but not validated at this dataset scale.

**Checkpoint:** only the DINOv2-weights-download fallback above; otherwise none.

**Tests:**
- Feed a dummy batch (`batch=2, frames=4` to keep it light on CPU/MPS) through the full forward pass end-to-end. Assert: output shapes correct at every stage, exactly top-2 experts have non-zero gate weight per sample, a single `.backward()` call succeeds with no NaN gradients.
- Print and save parameter counts per module to `reports/architecture_summary.txt`.

**Deliverable:** a verified, shape-correct architecture with a parameter-count summary and the HTML architecture diagram (already built) as the visual explainer — presentable as "the model is implemented and provably wired correctly" even with zero training done.

---

### Phase 6 — Loss functions

**Goal:** `src/losses/` implementing `L_cls` (BCE), `L_route` (CE vs ffprobe-derived target), `L_con` (source-aware InfoNCE, τ=0.07), `L_bal` (Switch-style balancing), combined per the spec's weights.

**Tasks:**
- Each loss as its own testable function; `total_loss()` combining them with configurable weights from `configs/model.yaml`.

**Checkpoint:** none.

**Tests:**
- On dummy batches: each loss is a finite, non-negative scalar; `L_con` is near-zero when same-source views are forced identical and large when they're forced orthogonal (sanity check on the contrastive logic, not real learning).
- Run 5 dummy optimizer steps (synthetic data, synthetic labels) and assert total loss trends downward — proves the training math is wired correctly, **not** that the model learns anything real.

**Deliverable:** a loss module with a printed 5-step dummy loss curve — proof the optimization loop is mechanically correct.

---

### Phase 7 — Training loop (local dry-run) → cloud training

**Goal:** A real training script, dry-run locally on synthetic/small data, then handed off for actual training.

**Tasks (local, no stop needed):**
- `src/train/train.py`: batch sampler (sources_per_batch × views_per_source), AdamW (lr 1e-4, wd 0.05), checkpointing, logging to a local CSV/JSON (no W&B unless the user wants it — ask, don't assume).
- Dry-run on the Phase 1/4 synthetic or small real sample for a handful of steps/epochs on CPU/MPS, purely to prove the loop doesn't crash.

**STOP AND ASK before real training:**
1. This cannot run on the M2 Mac (established earlier in this project — OOM risk, MPS kernel gaps). Real training needs Kaggle (free T4×2, 30 hrs/week) or Colab.
2. Confirm: should Claude Code export a ready-to-run Kaggle notebook / script (`notebooks/train_kaggle.ipynb` or `.py`) for the user to upload and run manually?
3. Ask whether any experiment-tracking credential (e.g. W&B API key) should be wired in, or keep it to local file logging only.
4. After the user runs it on Kaggle and downloads the checkpoint, ask where to place it locally (e.g. `data_cache/checkpoints/`) before continuing to Phase 8/9.

**Tests:**
- Local dry-run: assert training completes N steps without error, loss is logged, a checkpoint file is written and reloadable.

**Deliverable:** a training script proven mechanically correct locally, plus (once the user runs it) a real checkpoint and real loss curve from Kaggle — the first point with genuine learned weights.

---

### Phase 8 — Evaluation / robustness benchmark

**Goal:** `src/eval/` producing AUC/ACC/ECE sliced by depth (G0–G3) and seen-vs-unseen chain, plus the MoE-vs-single-head-ablation comparison.

**Tasks:**
- Metrics module (AUC, ACC, calibration/ECE).
- Slicing logic keyed off `metadata.jsonl`'s depth + the held-out-chain list from the spec.
- Comparison runner: MoE (top-2) vs single-head ablation vs no-`L_con` ablation.
- Degradation-curve chart generator (matplotlib) matching the "AUC vs depth" shape from the architecture doc.

**Checkpoint:** none if a checkpoint already exists from Phase 7; otherwise this phase can still be built and tested against dummy prediction arrays.

**Tests:**
- Run metrics/slicing on a synthetic predictions array first (no model needed) — assert correct bucket counts and metric math.
- If a real checkpoint exists, run end-to-end on the held-out real test split and save the comparison table + degradation chart to `reports/`.

**Deliverable:** a benchmark table and chart — real if a checkpoint exists, clearly-labeled-synthetic sanity check if not. Either way, presentable.

---

### Phase 9 — FastAPI backend + Streamlit UI

**Goal:** A working local demo: upload a video, see the routing decision and verdict.

**Tasks:**
- `src/api/main.py` (FastAPI): `/predict` endpoint — preprocess → forward pass → return `{p_fake, predicted_history, gate_weights, saliency}`. If no trained checkpoint exists yet, run with randomly-initialized weights and **clearly label the response as "untrained model — architecture demo only."**
- `ui/app.py` (Streamlit): upload box, calls the FastAPI endpoint, displays verdict, router gate weights (bar chart), predicted vs. actual processing history, per-frame score if available.
- `configs/paths.yaml` / `.env.example` for the API URL — document any env var Claude Code introduces here and ask before assuming defaults matter.

**Checkpoint:** ask whether the demo should point at a real trained checkpoint (if Phase 7 produced one) or run in explicit "untrained architecture demo" mode — don't silently pick one.

**Tests:**
- `uvicorn` starts without error; `curl`/`httpx` test hits `/predict` with the Phase 1 synthetic clip and gets a well-formed JSON response.
- Streamlit app loads and successfully calls the API end-to-end on the synthetic clip.

**Deliverable:** a live, clickable demo — the single most presentable artifact for a viva, independent of whether real training ever finished.

---

### Phase 10 — Report assets & packaging

**Goal:** Pull every phase's artifacts into report-ready form.

**Tasks:**
- Collect `reports/` figures (face/DCT grid, architecture summary, loss curves, degradation chart, comparison table) into a single `reports/final_summary.md`.
- Fill in the "Modules/Components," "Algorithms/Models," "Implementation Details," "Experimental Setup," "Results and Analysis," and "Comparison with Existing/Baseline" slide placeholders from the PPT using whatever phases actually completed — explicitly marking any slide as "scaffolded, not yet run" where applicable.
- Update `DECISIONS.md` and `PROGRESS.md` with an honest final status: which phases are real-data-verified, which are architecture-only, which are untrained-demo-only.

**Checkpoint:** none.

**Tests:** manual review only — no code test applies.

**Deliverable:** report content that accurately reflects what's real vs. scaffolded, so the submission never overclaims.

---

## Appendix A — Why K=5 (not 13) for the micro-run

With ~80–120 source videos split across train/val/test, spreading them over the full 13 codec×depth clusters leaves several clusters with too few examples for the router's load-balancing loss to behave well. Default `configs/model.yaml` to a reduced cluster set (e.g. 2 codecs × 2 depths + clean = 5) for anything actually trained at this scale, while keeping the full `K=13` architecture documented as the designed target. Change `num_experts` in the config, not in code, if the real dataset budget ends up larger.

## Appendix B — `PROGRESS.md` starter template

```markdown
# Progress

## Status
- Last completed phase: (none yet)
- Current phase: 0
- Blocked on: (nothing)

## Phase log
- Phase 0: not started
- Phase 1: not started
...
```

## Appendix C — `DECISIONS.md` starter template

```markdown
# Decisions

## Phase 0 — Harsh's WhatsApp Compression Simulator
- Tested on: (date)
- Result: (runs clean / runs but black-box / fails — with exact error)
- Decision: (integrated as-is / wrapped with ffprobe / reimplemented, using it only as a recipe reference)
```
