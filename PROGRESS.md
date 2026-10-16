# Progress

## Status
- Last completed phase: 0
- Current phase: 1 (not started)
- Blocked on: nothing

## Notes
- ffmpeg: Homebrew fell back to a from-source build in this sandbox (bottle host unreachable) — would have taken 15-20+ min. Switched to a dedicated conda-forge env (`fyp-ffmpeg`) instead, which has the full codec set (libx264/libx265/libvpx-vp9/libsvtav1) needed for Phase 1. Pipeline code must resolve ffmpeg/ffprobe via `configs/paths.yaml` (`ffmpeg_binary`/`ffprobe_binary`), not `$PATH` — see DECISIONS.md.
- Harsh's WhatsApp-Compression-Simulator: evaluated, runs clean, exposes real ffmpeg params via JSON sidecar. Will be wrapped as one named recipe inside `src/pipeline/` for the "WhatsApp hop" in Phase 1 — see DECISIONS.md.
- All data used so far is synthetic (`data_cache/synthetic/test_input.mp4`, generated via ffmpeg testsrc/sine, no download). No real FF++/Celeb-DF data touched yet — that's Phase 3, gated on explicit user go-ahead.

## Phase log
- Phase 0: done — `pytest tests/test_phase0.py` 3/3 passed
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
