# Progress

## Status
- Last completed phase: 4
- Current phase: 5 (not started)
- Blocked on: nothing

## Notes
- ffmpeg: Homebrew fell back to a from-source build in this sandbox (bottle host unreachable) — would have taken 15-20+ min. Switched to a dedicated conda-forge env (`fyp-ffmpeg`) instead, which has the full codec set (libx264/libx265/libvpx-vp9/libsvtav1) needed for Phase 1. Pipeline code must resolve ffmpeg/ffprobe via `configs/paths.yaml` (`ffmpeg_binary`/`ffprobe_binary`), not `$PATH` — see DECISIONS.md.
- Harsh's WhatsApp-Compression-Simulator: evaluated, runs clean, exposes real ffmpeg params via JSON sidecar. Wrapped as the h264-hop implementation in `src/pipeline/transform.py` (see Phase 1 decision in DECISIONS.md) rather than reimplemented.
- `libsvtav1` needed a per-codec rate-control exception (`-maxrate`/`-bufsize` unsupported outside CRF mode) — see Phase 1 decision in DECISIONS.md. Fixed via config, not a hardcoded special case in code.
- All data used so far is synthetic (`data_cache/synthetic/test_input.mp4`, generated via ffmpeg testsrc/sine, no download; chain outputs in `data_cache/chains/`, 84 files, 7.4MB). No real FF++/Celeb-DF data touched yet — that's Phase 3, gated on explicit user go-ahead.

## Phase log
- Phase 0: done — `pytest tests/test_phase0.py` 3/3 passed
- Phase 1: done — `src/pipeline/transform.py` (`single_hop`, `build_chain_tree`), `configs/transform.yaml`; `pytest tests/test_phase1.py` 4/4 passed; 84 playable, non-duplicate files generated from the Phase 0 synthetic clip in ~19s
- Phase 2: done — `src/pipeline/probe.py` (actual ffprobe metadata + router_target), `src/data/split.py` (leakage-aware source_id splitter); `data_cache/manifest/metadata.jsonl` has 85 rows (1 clean_G0 + 84 transforms), all from one synthetic source_id; `pytest tests/test_phase2.py` 6/6 passed
- Phase 3: done — real FF++ c23 + Celeb-DF v2 sample (45 videos, 72.1MB) downloaded per user-approved budget; `configs/dataset_sample.yaml` records exact composition; `src/data/download_sample.py`, `src/data/build_real_manifest.py`; `data_cache/manifest/metadata_real.jsonl` (45 real rows, leakage-free split, kept separate from Phase 2's synthetic manifest); full transform chain smoke-tested on 1 real video (84/84 files, no errors) rather than all 45 (user-approved cost tradeoff — see DECISIONS.md); `pytest tests/test_phase3.py` 5/5 passed. Full suite: 18/18 passed.
- Phase 4: done — `src/preprocessing/` (`frames.py`, `align.py`, `dct.py`, `pipeline.py`); face detection via mediapipe's FaceDetector task (facenet-pytorch's packaging is broken upstream — see DECISIONS.md), model weights auto-downloaded (229KB, logged per BUILD.md's <50MB checkpoint note, no stop needed); verified RGB `[16,3,224,224]` and DCT `[16,784,64]` shapes on 3 real files (real + fake FF++, real Celeb-DF); visual proof in `reports/phase4_samples/` (face-crop grid, DCT heatmap); `pytest tests/test_phase4.py` 5/5 passed. Full suite: 23/23 passed.

## Known data-cache contents (gitignored, for reference)
- `data_cache/raw/ffpp/`, `data_cache/raw/celebdf/`: the 45 real downloaded videos + `sample_manifest.json`
- `data_cache/chains/`: synthetic chain (84 files, `test_input__*`) + one real smoke-test chain (84 files, `000__*`)
- `data_cache/manifest/metadata.jsonl` (synthetic, Phase 2) and `metadata_real.jsonl` (real, Phase 3) — kept deliberately separate
- Phase 1: not started
- Phase 2: not started
- Phase 3: not started
- Phase 4: not started
- Phase 5: not started
- Phase 6: not started
- Phase 7: not started
- Phase 8: not started
- Phase 9: not started
- Phase 10: not started
