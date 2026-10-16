# WhatsApp Video Compression Research Pipeline

A lightweight, CPU-based Python research toolkit designed to investigate and simulate how messaging platforms like WhatsApp transform video files when shared through standard video sharing (not as uncompressed documents).

This project provides empirical calibration tools for researchers studying **robust deepfake detection under unknown social-network transformation chains**.

---

## 1. Architecture Overview

The system consists of two primary pipelines:

1. **Pipeline 1 (`compress.py`)**: An FFmpeg-based WhatsApp compression simulator that accepts local videos, applies configurable transformation baselines (H.264/AVC, 720p scaling, aspect ratio preservation, even dimensions, AAC audio), and outputs compressed files alongside reproducible JSON sidecars in `output/simulated/`.
2. **Pipeline 2 (`compare.py`)**: A metadata extraction and quantitative comparison engine using FFprobe. It extracts structured container, video stream, and audio stream properties, saving raw FFprobe JSON and consolidated CSV files. It performs 3-way quantitative comparisons across:
   $$\text{Original Video} \longleftrightarrow \text{Simulated Video} \longleftrightarrow \text{Actual WhatsApp Video}$$

```text
Whatsapp-Compression-Simulator/
├── input/
│   └── Put original videos here
├── output/
│   ├── simulated/
│   │   └── Compressed videos + JSON sidecars
│   └── whatsapp/
│       └── Received videos copied here manually
├── results/
│   ├── metadata.csv
│   ├── comparisons.csv
│   ├── experiment_log.csv
│   └── metadata_json/
│       └── <category>_<stem>_metadata.json
├── compress.py
├── compare.py
├── requirements.txt
└── README.md
```

---

## 2. Prerequisites & Installation

### Requirements
- **OS**: Windows 10/11 (or Linux / macOS)
- **Python**: 3.10+
- **FFmpeg & FFprobe**: Installed locally (no GPU required; runs entirely on CPU)

### Installing FFmpeg on Windows
If FFmpeg is not already installed, install it using Windows Package Manager (`winget`):
```powershell
winget install Gyan.FFmpeg.Essentials
```
*Note: Both `compress.py` and `compare.py` automatically detect FFmpeg and FFprobe in standard Windows WinGet directories, registry paths, and system PATH.*

### Python Environment Setup
From within the repository root:
```powershell
pip install -r requirements.txt
```
*(Standard Python libraries are used throughout; `pandas` is included for robust CSV handling).*

---

## 3. Pipeline 1: Compression Simulator (`compress.py`)

Simulates social-network compression while preserving aspect ratios, enforcing even dimensions, handling videos with or without audio tracks, and logging reproducible execution metadata.

### Research Baseline Settings (Provisional)
WhatsApp transcode behavior varies depending on input dimensions, standard vs. HD toggle, and client platform. The default profile represents a **provisional standard-quality baseline**:
- **Video Codec**: H.264 / AVC (`libx264`)
- **Pixel Format**: `yuv420p` (compatible with 8-bit mobile decoders)
- **CRF**: `28` (configurable range: 24–32)
- **Preset**: `medium`
- **Max Resolution**: `1280`x`720` (downscales larger videos while preserving aspect ratio; never upscales)
- **Container**: MP4 with `+faststart` (web/stream-optimized)
- **Audio**: AAC at `128k` (or passthrough `copy`), automatically disabled (`-an`) if the source has no audio track
- **Frame Rate**: Original FPS preserved by default (FPS changes are never applied silently)

### Usage Examples

#### Single Video Compression
```powershell
python compress.py input/sample.mp4
```

#### Custom Calibration Parameters
```powershell
python compress.py input/sample.mp4 --crf 30 --max-width 848 --max-height 480 --preset faster
```

#### Optional Frame Rate Target
```powershell
python compress.py input/sample.mp4 --fps 30
```

#### Batch Processing
Compress all videos placed in `input/`:
```powershell
python compress.py --batch
```

#### Dry Run Preview
Inspect the generated FFmpeg command without executing:
```powershell
python compress.py input/sample.mp4 --dry-run
```

### Reproducibility & Sidecar JSON
For every video processed, a JSON sidecar is written to `output/simulated/<stem>_simulated.json`. It records:
- The exact FFmpeg command-line string and argument list
- Execution duration and timestamps (UTC)
- Encoder parameters (CRF, preset, pixel format, max resolution)
- Input and output byte counts and size reduction percentages
- Clear attribution tagging identifying the file as a **simulated result**

---

## 4. Pipeline 2: Metadata Extraction & Comparison (`compare.py`)

Extracts technical stream metadata and performs quantitative delta analyses across video triplets.

### Mode A: Scan & Extract Metadata
Scans `input/`, `output/simulated/`, and `output/whatsapp/`. Saves structured metadata to `results/metadata.csv` and raw FFprobe JSON trees to `results/metadata_json/`.

```powershell
python compare.py scan
```

#### Extracted Metadata Fields
- **Container**: Filename, full path, file size (bytes and MB), container format name, duration (seconds), overall bitrate (kbps).
- **Video Stream**: Codec name (`h264`, `hevc`, etc.), codec tag (`avc1`), width, height, resolution string, average and nominal FPS, video bitrate (kbps), pixel format (`yuv420p`), profile, level, frame count.
- **Audio Stream**: Presence flag (`has_audio`), audio codec (`aac`), audio bitrate (kbps), sample rate (Hz), channels.

### Mode B: Triplet Comparison
Compares the original source against the simulated output and the actual received WhatsApp output.

#### Compare by Explicit Paths
```powershell
python compare.py compare --original input/sample.mp4 --simulated output/simulated/sample_simulated.mp4 --whatsapp output/whatsapp/sample.mp4
```

#### Compare by Sample ID
Automatically pairs matching files across `input/`, `output/simulated/`, and `output/whatsapp/`:
```powershell
python compare.py compare --sample-id sample
```

#### Batch Comparison
Automatically evaluates all matched pairs/triplets in the workspace:
```powershell
python compare.py compare --batch
```

*Note: If the actual WhatsApp video has not yet been copied into `output/whatsapp/`, the comparison script safely computes Original vs. Simulator metrics and marks the WhatsApp ground-truth comparison as `[Pending]` without failing.*

### Comparison Calculations
1. **File Size**: Absolute byte reduction and percentage reduction.
2. **Resolution & Geometry**: Width/height changes, pixel count reduction %, and resolution match boolean.
3. **Duration**: Absolute duration delta (seconds) and percentage change.
4. **Frame Rate**: Exact FPS difference (`comp_fps - orig_fps`).
5. **Bitrate**: Overall bitrate reduction (kbps) and percentage reduction.
6. **Codecs & Formats**: Codec match flags for video, audio, and pixel format.
7. **Direct Calibration Delta**: Simulated output vs. actual WhatsApp output (byte delta, bitrate delta, resolution match, FPS delta).

Results are appended/updated in `results/comparisons.csv` and rendered as a formatted report in the terminal.

---

## 5. WhatsApp Ground-Truth Collection & Calibration Workflow

To calibrate the FFmpeg simulator against actual WhatsApp compression:

1. **Place Original Video**: Place an uncompressed or high-bitrate original video in `input/` (e.g. `input/sample.mp4`).
2. **Run Simulator**:
   ```powershell
   python compress.py input/sample.mp4
   ```
3. **Send via WhatsApp**:
   - Send `input/sample.mp4` to a test device using **standard WhatsApp video sharing** (select from gallery/camera roll as a standard video).
   - **Do NOT** send as a Document/File (which skips transcoding).
   - Select Standard Quality (or HD if calibrating the HD profile).
4. **Retrieve Received Video**:
   - On the receiving device, save the received video.
   - Copy the received video file into `output/whatsapp/` (e.g., `output/whatsapp/sample.mp4`).
   - **Never edit, trim, or re-encode** the received video.
5. **Log Ground-Truth Parameters**:
   Record experimental conditions:
   ```powershell
   python compare.py log-experiment --sample-id sample --whatsapp-version "2.24.15.78" --send-device "iPhone 13 iOS 17.5" --receive-device "Pixel 7 Android 14" --send-method "Standard Normal Video" --notes "Indoor lighting, 5G network"
   ```
   *(Saved to `results/experiment_log.csv`).*
6. **Execute Comparison**:
   ```powershell
   python compare.py compare --sample-id sample
   ```
7. **Calibrate Simulator Parameters**:
   - Inspect the `SIMULATOR CALIBRATION DELTA` section in the terminal report.
   - If WhatsApp's output resolution is smaller (e.g. 848x480), adjust `--max-width` and `--max-height`.
   - If WhatsApp's bitrate or file size is lower, adjust `--crf` (increase CRF to lower bitrate) or specify `--video-bitrate`.
   - Re-run `compress.py` with calibrated arguments and re-compare until the simulation closely mirrors the empirical distribution.
8. **Repeat Across Test Corpus**:
   Repeat across videos with diverse properties:
   - High motion (sports, fast pans) vs. low motion (talking heads, interviews)
   - High detail (textures, nature) vs. flat lighting
   - Different aspect ratios (16:9 landscape, 9:16 portrait, 1:1 square)
   - Different native frame rates (24, 30, 60 fps)

---

## 6. Research Disclaimers & Safety Principles

> [!WARNING]
> **Important Scientific Notice**
> 1. Matching codec names (`h264`) or container formats (`mp4`) does **not** prove identical encoder settings, quantization matrices, motion vector algorithms, or equivalent visual/forensic quality.
> 2. Settings in `compress.py` are **provisional baseline models** designed for empirical calibration. WhatsApp's internal transcoding pipelines are proprietary and subject to periodic server-side and client-side updates.
> 3. This repository does **not** use unofficial WhatsApp APIs, automated scraping, or reverse-engineering of app binaries. All ground truth is gathered via manual user sharing.
> 4. Source video files and received ground-truth videos are treated as read-only and will never be overwritten or deleted by these scripts.

---

## 7. Quick Start Verification

Run the end-to-end verification pipeline:
```powershell
# 1. Compress a test video
python compress.py input/sample.mp4

# 2. Extract metadata
python compare.py scan

# 3. View comparison report
python compare.py compare --sample-id sample
```
