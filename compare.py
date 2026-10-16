#!/usr/bin/env python3
"""
WhatsApp Video Metadata Extraction & Comparison Tool (Research Pipeline 2)

Extracts structured technical metadata using FFprobe and performs quantitative
comparisons across video triplets:
  [Original Video] <---> [Simulator Video] <---> [Actual WhatsApp Video]

RESEARCH DISCLAIMER:
Matching codec names or resolutions do NOT prove identical encoder settings,
GOP structure, rate-control behavior, or equivalent visual quality. This tool
measures observable stream properties for empirical calibration.
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Supported video formats
SUPPORTED_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".m4v", ".flv", ".wmv", ".ts", ".m2ts"
}


def find_ffprobe_binary() -> Optional[str]:
    """
    Locate ffprobe executable. Checks system PATH, Windows registry,
    and WinGet installation folders.
    """
    path = shutil.which("ffprobe")
    if path:
        return str(Path(path).resolve())

    if sys.platform == "win32":
        try:
            import winreg
            for hkey in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    subpath = r"Environment" if hkey == winreg.HKEY_CURRENT_USER else r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
                    with winreg.OpenKey(hkey, subpath) as key:
                        val, _ = winreg.QueryValueEx(key, "Path")
                        for p in val.split(";"):
                            candidate = Path(p.strip()) / "ffprobe.exe"
                            if candidate.is_file():
                                os.environ["PATH"] = p.strip() + os.pathsep + os.environ.get("PATH", "")
                                return str(candidate.resolve())
                except OSError:
                    continue
        except ImportError:
            pass

        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            import glob
            pattern = os.path.join(local_app_data, "Microsoft", "WinGet", "Packages", "**", "ffprobe.exe")
            matches = glob.glob(pattern, recursive=True)
            if matches:
                bin_dir = str(Path(matches[0]).parent)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                return str(Path(matches[0]).resolve())

        candidates = [
            Path(r"C:\Program Files\ffmpeg\bin\ffprobe.exe"),
            Path(r"C:\ffmpeg\bin\ffprobe.exe"),
            Path(r"C:\ProgramData\chocolatey\bin\ffprobe.exe"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                bin_dir = str(candidate.parent)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                return str(candidate.resolve())

    return None


def get_ffprobe() -> str:
    """Ensure ffprobe is available, or exit with a clear error."""
    ffprobe_path = find_ffprobe_binary()
    if not ffprobe_path:
        sys.stderr.write(
            "ERROR: 'ffprobe' executable not found in PATH or standard installation directories.\n"
            "Please ensure FFmpeg/FFprobe is installed.\n"
        )
        sys.exit(1)
    return ffprobe_path


def parse_fraction(val: Optional[str]) -> Optional[float]:
    """Safely parse FFprobe fractional strings like '30000/1001' or '30/1'."""
    if not val or val in ("0/0", "N/A"):
        return None
    try:
        if "/" in val:
            num, den = val.split("/", 1)
            f_den = float(den)
            return round(float(num) / f_den, 3) if f_den != 0 else None
        return round(float(val), 3)
    except (ValueError, TypeError):
        return None


def safe_int(val: Any) -> Optional[int]:
    """Safely convert value to integer or return None."""
    if val is None or val == "" or val == "N/A":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return None


def safe_float(val: Any, decimals: int = 3) -> Optional[float]:
    """Safely convert value to float or return None."""
    if val is None or val == "" or val == "N/A":
        return None
    try:
        return round(float(val), decimals)
    except (ValueError, TypeError):
        return None


def run_ffprobe_json(ffprobe_path: str, video_path: Path) -> Dict[str, Any]:
    """
    Execute ffprobe to extract raw structured JSON (-show_streams, -show_format).
    """
    cmd = [
        ffprobe_path,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path)
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(proc.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFprobe execution failed for {video_path}: {e.stderr.strip()}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to decode JSON from FFprobe output: {e}")


def extract_video_metadata(ffprobe_path: str, video_path: Path, category: str = "unknown") -> Dict[str, Any]:
    """
    Extract container, video stream, and audio stream metadata from a video file.
    Returns normalized dictionary with all required fields (nulls for missing fields).
    """
    video_path = video_path.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    raw_data = run_ffprobe_json(ffprobe_path, video_path)

    fmt = raw_data.get("format", {})
    streams = raw_data.get("streams", [])

    # Find first video stream
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), {})

    # Find first audio stream
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

    file_size_bytes = video_path.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 3)

    # Container information
    container_format = fmt.get("format_name")
    duration_sec = safe_float(fmt.get("duration"))
    overall_bitrate_bps = safe_int(fmt.get("bit_rate"))
    overall_bitrate_kbps = round(overall_bitrate_bps / 1000, 2) if overall_bitrate_bps else None

    # Video stream information
    video_codec = v_stream.get("codec_name")
    codec_tag = v_stream.get("codec_tag_string") or v_stream.get("codec_tag")
    width = safe_int(v_stream.get("width"))
    height = safe_int(v_stream.get("height"))
    resolution = f"{width}x{height}" if (width and height) else None
    avg_fps = parse_fraction(v_stream.get("avg_frame_rate"))
    nominal_fps = parse_fraction(v_stream.get("r_frame_rate"))
    v_bitrate_bps = safe_int(v_stream.get("bit_rate"))
    if not v_bitrate_bps:
        # Check stream tags if available
        tags = v_stream.get("tags", {})
        bps_tag = tags.get("BPS") or tags.get("BPS-eng")
        if bps_tag:
            v_bitrate_bps = safe_int(bps_tag)
    v_bitrate_kbps = round(v_bitrate_bps / 1000, 2) if v_bitrate_bps else None
    pixel_format = v_stream.get("pix_fmt")
    profile = v_stream.get("profile")
    level = v_stream.get("level")
    nb_frames = safe_int(v_stream.get("nb_frames"))

    # Audio stream information
    has_audio = bool(a_stream)
    audio_codec = a_stream.get("codec_name") if has_audio else None
    a_bitrate_bps = safe_int(a_stream.get("bit_rate")) if has_audio else None
    if has_audio and not a_bitrate_bps:
        tags = a_stream.get("tags", {})
        bps_tag = tags.get("BPS") or tags.get("BPS-eng")
        if bps_tag:
            a_bitrate_bps = safe_int(bps_tag)
    a_bitrate_kbps = round(a_bitrate_bps / 1000, 2) if a_bitrate_bps else None
    sample_rate = safe_int(a_stream.get("sample_rate")) if has_audio else None
    channels = safe_int(a_stream.get("channels")) if has_audio else None

    return {
        "category": category,                      # 'original', 'simulated', 'whatsapp', or custom
        "sample_id": video_path.stem.replace("_simulated", "").replace("_whatsapp", ""),
        "filename": video_path.name,
        "file_path": str(video_path),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "container_format": container_format,
        "duration_seconds": duration_sec,
        "overall_bitrate_bps": overall_bitrate_bps,
        "overall_bitrate_kbps": overall_bitrate_kbps,
        # Video stream
        "video_codec": video_codec,
        "codec_tag": codec_tag,
        "width": width,
        "height": height,
        "resolution": resolution,
        "avg_frame_rate": avg_fps,
        "nominal_frame_rate": nominal_fps,
        "video_bitrate_bps": v_bitrate_bps,
        "video_bitrate_kbps": v_bitrate_kbps,
        "pixel_format": pixel_format,
        "profile": profile,
        "level": level,
        "nb_frames": nb_frames,
        # Audio stream
        "has_audio": has_audio,
        "audio_codec": audio_codec,
        "audio_bitrate_bps": a_bitrate_bps,
        "audio_bitrate_kbps": a_bitrate_kbps,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        # Raw ffprobe cache
        "_raw_data": raw_data,
    }


def save_metadata_json(raw_data: Dict[str, Any], output_path: Path):
    """Write raw FFprobe JSON to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)


def save_csv(rows: List[Dict[str, Any]], csv_path: Path):
    """Save rows to CSV using pandas if available, falling back to standard csv."""
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter out private keys (like _raw_data)
    clean_rows = []
    for r in rows:
        clean_rows.append({k: v for k, v in r.items() if not k.startswith("_")})

    try:
        import pandas as pd
        df = pd.DataFrame(clean_rows)
        df.to_csv(csv_path, index=False, encoding="utf-8")
    except ImportError:
        fieldnames = list(clean_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clean_rows)


def scan_directories(
    input_dir: Path,
    simulated_dir: Path,
    whatsapp_dir: Path,
    results_dir: Path,
    ffprobe_path: str
) -> List[Dict[str, Any]]:
    """
    Mode A: Scan video files from input/, output/simulated/, and output/whatsapp/.
    Extracts metadata, saves raw FFprobe JSON to results/metadata_json/,
    and writes results/metadata.csv.
    """
    json_dir = results_dir / "metadata_json"
    json_dir.mkdir(parents=True, exist_ok=True)

    scan_targets = [
        ("original", input_dir),
        ("simulated", simulated_dir),
        ("whatsapp", whatsapp_dir),
    ]

    all_metadata: List[Dict[str, Any]] = []

    print("=" * 80)
    print("MODE A: METADATA EXTRACTION SCAN")
    print("=" * 80)

    for category, directory in scan_targets:
        if not directory.is_dir():
            print(f"[INFO] Directory '{directory}' does not exist yet (skipping {category}).")
            continue

        video_files = [
            f for f in sorted(directory.iterdir())
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        print(f"Scanning [{category.upper()}]: '{directory}' -> found {len(video_files)} video(s)")

        for vid in video_files:
            try:
                meta = extract_video_metadata(ffprobe_path, vid, category=category)
                all_metadata.append(meta)

                # Save raw FFprobe JSON
                json_filename = f"{category}_{vid.stem}_metadata.json"
                save_metadata_json(meta["_raw_data"], json_dir / json_filename)
                print(f"  + Extracted: {vid.name} ({meta['resolution']}, {meta['file_size_mb']} MB, codec={meta['video_codec']})")
            except Exception as e:
                print(f"  - [ERROR] Failed to extract {vid.name}: {e}", file=sys.stderr)

    # Save to CSV
    csv_path = results_dir / "metadata.csv"
    save_csv(all_metadata, csv_path)

    print("-" * 80)
    print(f"Extraction complete: {len(all_metadata)} file(s) processed.")
    print(f"Metadata CSV saved: {csv_path}")
    print(f"FFprobe JSON saved: {json_dir}")
    print("=" * 80)

    return all_metadata


def compute_comparison_metrics(
    orig: Dict[str, Any],
    comp: Dict[str, Any],
    prefix: str
) -> Dict[str, Any]:
    """
    Compute delta and percentage reduction metrics between an original video
    and a compressed / modified video.
    """
    results: Dict[str, Any] = {}

    orig_size = orig.get("file_size_bytes")
    comp_size = comp.get("file_size_bytes")
    if orig_size is not None and comp_size is not None:
        diff_bytes = orig_size - comp_size
        pct = round((diff_bytes / orig_size) * 100, 2) if orig_size > 0 else 0.0
        results[f"{prefix}_size_reduction_bytes"] = diff_bytes
        results[f"{prefix}_size_reduction_pct"] = pct
    else:
        results[f"{prefix}_size_reduction_bytes"] = None
        results[f"{prefix}_size_reduction_pct"] = None

    # Resolution & Pixels
    orig_w, orig_h = orig.get("width"), orig.get("height")
    comp_w, comp_h = comp.get("width"), comp.get("height")
    if orig_w is not None and orig_h is not None and comp_w is not None and comp_h is not None:
        results[f"{prefix}_width_change"] = comp_w - orig_w
        results[f"{prefix}_height_change"] = comp_h - orig_h
        results[f"{prefix}_resolution_match"] = (orig_w == comp_w and orig_h == comp_h)
        orig_pixels = orig_w * orig_h
        comp_pixels = comp_w * comp_h
        diff_pixels = orig_pixels - comp_pixels
        pct_pixels = round((diff_pixels / orig_pixels) * 100, 2) if orig_pixels > 0 else 0.0
        results[f"{prefix}_pixel_count_reduction_pct"] = pct_pixels
    else:
        results[f"{prefix}_width_change"] = None
        results[f"{prefix}_height_change"] = None
        results[f"{prefix}_resolution_match"] = None
        results[f"{prefix}_pixel_count_reduction_pct"] = None

    # Duration
    orig_dur = orig.get("duration_seconds")
    comp_dur = comp.get("duration_seconds")
    if orig_dur is not None and comp_dur is not None:
        diff_dur = round(comp_dur - orig_dur, 3)
        pct_dur = round((diff_dur / orig_dur) * 100, 2) if orig_dur > 0 else 0.0
        results[f"{prefix}_duration_diff_sec"] = diff_dur
        results[f"{prefix}_duration_change_pct"] = pct_dur
    else:
        results[f"{prefix}_duration_diff_sec"] = None
        results[f"{prefix}_duration_change_pct"] = None

    # FPS
    orig_fps = orig.get("avg_frame_rate")
    comp_fps = comp.get("avg_frame_rate")
    if orig_fps is not None and comp_fps is not None:
        results[f"{prefix}_fps_diff"] = round(comp_fps - orig_fps, 3)
    else:
        results[f"{prefix}_fps_diff"] = None

    # Bitrate
    orig_br = orig.get("overall_bitrate_kbps")
    comp_br = comp.get("overall_bitrate_kbps")
    if orig_br is not None and comp_br is not None:
        diff_br = round(orig_br - comp_br, 2)
        pct_br = round((diff_br / orig_br) * 100, 2) if orig_br > 0 else 0.0
        results[f"{prefix}_bitrate_reduction_kbps"] = diff_br
        results[f"{prefix}_bitrate_reduction_pct"] = pct_br
    else:
        results[f"{prefix}_bitrate_reduction_kbps"] = None
        results[f"{prefix}_bitrate_reduction_pct"] = None

    # Codecs and Pixel Format
    results[f"{prefix}_video_codec_match"] = (
        orig.get("video_codec") == comp.get("video_codec")
        if (orig.get("video_codec") and comp.get("video_codec")) else None
    )
    results[f"{prefix}_pixel_format_match"] = (
        orig.get("pixel_format") == comp.get("pixel_format")
        if (orig.get("pixel_format") and comp.get("pixel_format")) else None
    )
    results[f"{prefix}_audio_codec_match"] = (
        orig.get("audio_codec") == comp.get("audio_codec")
        if (orig.get("has_audio") and comp.get("has_audio")) else None
    )

    return results


def compare_triplet(
    orig_meta: Dict[str, Any],
    sim_meta: Dict[str, Any],
    wa_meta: Optional[Dict[str, Any]],
    sample_id: str
) -> Dict[str, Any]:
    """
    Perform complete 3-way quantitative comparison:
      1. Original vs Simulator
      2. Original vs Actual WhatsApp (if available)
      3. Simulator vs Actual WhatsApp (if available)
    """
    row: Dict[str, Any] = {
        "sample_id": sample_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        # Original summary
        "orig_file": orig_meta.get("filename"),
        "orig_size_mb": orig_meta.get("file_size_mb"),
        "orig_resolution": orig_meta.get("resolution"),
        "orig_vcodec": orig_meta.get("video_codec"),
        "orig_fps": orig_meta.get("avg_frame_rate"),
        "orig_bitrate_kbps": orig_meta.get("overall_bitrate_kbps"),
        # Simulated summary
        "sim_file": sim_meta.get("filename"),
        "sim_size_mb": sim_meta.get("file_size_mb"),
        "sim_resolution": sim_meta.get("resolution"),
        "sim_vcodec": sim_meta.get("video_codec"),
        "sim_fps": sim_meta.get("avg_frame_rate"),
        "sim_bitrate_kbps": sim_meta.get("overall_bitrate_kbps"),
    }

    # 1. Original vs Simulated
    orig_sim_metrics = compute_comparison_metrics(orig_meta, sim_meta, "orig_vs_sim")
    row.update(orig_sim_metrics)

    # WhatsApp information
    if wa_meta:
        row["wa_status"] = "available"
        row["wa_file"] = wa_meta.get("filename")
        row["wa_size_mb"] = wa_meta.get("file_size_mb")
        row["wa_resolution"] = wa_meta.get("resolution")
        row["wa_vcodec"] = wa_meta.get("video_codec")
        row["wa_fps"] = wa_meta.get("avg_frame_rate")
        row["wa_bitrate_kbps"] = wa_meta.get("overall_bitrate_kbps")

        # 2. Original vs WhatsApp
        orig_wa_metrics = compute_comparison_metrics(orig_meta, wa_meta, "orig_vs_wa")
        row.update(orig_wa_metrics)

        # 3. Direct Simulated vs WhatsApp Comparison
        sim_size = sim_meta.get("file_size_bytes")
        wa_size = wa_meta.get("file_size_bytes")
        if sim_size is not None and wa_size is not None:
            size_diff = sim_size - wa_size
            pct = round((size_diff / wa_size) * 100, 2) if wa_size > 0 else 0.0
            row["sim_vs_wa_size_diff_bytes"] = size_diff
            row["sim_vs_wa_size_diff_pct"] = pct
        else:
            row["sim_vs_wa_size_diff_bytes"] = None
            row["sim_vs_wa_size_diff_pct"] = None

        sim_br = sim_meta.get("overall_bitrate_kbps")
        wa_br = wa_meta.get("overall_bitrate_kbps")
        row["sim_vs_wa_bitrate_diff_kbps"] = round(sim_br - wa_br, 2) if (sim_br and wa_br) else None

        row["sim_vs_wa_resolution_match"] = (sim_meta.get("resolution") == wa_meta.get("resolution"))
        row["sim_vs_wa_vcodec_match"] = (sim_meta.get("video_codec") == wa_meta.get("video_codec"))
        row["sim_vs_wa_pixfmt_match"] = (sim_meta.get("pixel_format") == wa_meta.get("pixel_format"))

        sim_fps = sim_meta.get("avg_frame_rate")
        wa_fps = wa_meta.get("avg_frame_rate")
        row["sim_vs_wa_fps_diff"] = round(sim_fps - wa_fps, 3) if (sim_fps and wa_fps) else None

    else:
        row["wa_status"] = "pending_collection"
        row["wa_file"] = None
        row["wa_size_mb"] = None
        row["wa_resolution"] = None
        row["wa_vcodec"] = None
        row["wa_fps"] = None
        row["wa_bitrate_kbps"] = None
        # Nullify WhatsApp comparisons
        for key in [
            "orig_vs_wa_size_reduction_bytes", "orig_vs_wa_size_reduction_pct",
            "orig_vs_wa_width_change", "orig_vs_wa_height_change",
            "orig_vs_wa_resolution_match", "orig_vs_wa_pixel_count_reduction_pct",
            "orig_vs_wa_duration_diff_sec", "orig_vs_wa_duration_change_pct",
            "orig_vs_wa_fps_diff", "orig_vs_wa_bitrate_reduction_kbps",
            "orig_vs_wa_bitrate_reduction_pct", "orig_vs_wa_video_codec_match",
            "orig_vs_wa_pixel_format_match", "orig_vs_wa_audio_codec_match",
            "sim_vs_wa_size_diff_bytes", "sim_vs_wa_size_diff_pct",
            "sim_vs_wa_bitrate_diff_kbps", "sim_vs_wa_resolution_match",
            "sim_vs_wa_vcodec_match", "sim_vs_wa_pixfmt_match", "sim_vs_wa_fps_diff"
        ]:
            row[key] = None

    row["research_disclaimer"] = (
        "Matching codec names or resolutions do NOT prove identical encoder settings "
        "or equivalent visual quality."
    )

    return row


def print_comparison_terminal(row: Dict[str, Any]):
    """Pretty-print comparison results to the terminal."""
    sid = row["sample_id"]
    print("=" * 80)
    print(f"TRIPLET COMPARISON REPORT: Sample ID '{sid}'")
    print("=" * 80)
    print(f"{'Attribute':<24} | {'Original':<16} | {'Simulated':<16} | {'WhatsApp Actual':<16}")
    print("-" * 80)

    wa_status = row.get("wa_status")
    wa_avail = (wa_status == "available")

    file_wa = str(row.get('wa_file') or '[Pending]')
    res_wa = str(row.get('wa_resolution') or '[Pending]')
    codec_wa = str(row.get('wa_vcodec') or '[Pending]')
    fps_wa = f"{row['wa_fps']} fps" if wa_avail and row.get('wa_fps') else '[Pending]'
    br_wa = f"{row['wa_bitrate_kbps']} kbps" if wa_avail and row.get('wa_bitrate_kbps') else '[Pending]'
    size_wa = f"{row['wa_size_mb']} MB" if wa_avail and row.get('wa_size_mb') else '[Pending]'

    orig_fps_str = f"{row.get('orig_fps')} fps" if row.get('orig_fps') is not None else "N/A"
    sim_fps_str = f"{row.get('sim_fps')} fps" if row.get('sim_fps') is not None else "N/A"
    orig_br_str = f"{row.get('orig_bitrate_kbps')} kbps" if row.get('orig_bitrate_kbps') is not None else "N/A"
    sim_br_str = f"{row.get('sim_bitrate_kbps')} kbps" if row.get('sim_bitrate_kbps') is not None else "N/A"
    orig_sz_str = f"{row.get('orig_size_mb')} MB" if row.get('orig_size_mb') is not None else "N/A"
    sim_sz_str = f"{row.get('sim_size_mb')} MB" if row.get('sim_size_mb') is not None else "N/A"

    print(f"{'Filename':<24} | {str(row.get('orig_file')):<16} | {str(row.get('sim_file')):<16} | {file_wa:<16}")
    print(f"{'Resolution':<24} | {str(row.get('orig_resolution')):<16} | {str(row.get('sim_resolution')):<16} | {res_wa:<16}")
    print(f"{'Video Codec':<24} | {str(row.get('orig_vcodec')):<16} | {str(row.get('sim_vcodec')):<16} | {codec_wa:<16}")
    print(f"{'Avg Frame Rate':<24} | {orig_fps_str:<16} | {sim_fps_str:<16} | {fps_wa:<16}")
    print(f"{'Overall Bitrate':<24} | {orig_br_str:<16} | {sim_br_str:<16} | {br_wa:<16}")
    print(f"{'File Size':<24} | {orig_sz_str:<16} | {sim_sz_str:<16} | {size_wa:<16}")
    print("-" * 80)

    print("QUANTITATIVE REDUCTIONS / DELTAS:")
    print(f"  * Orig vs Simulated Size Reduction : {row.get('orig_vs_sim_size_reduction_pct')}% ({row.get('orig_vs_sim_size_reduction_bytes'):,} bytes)")
    print(f"  * Orig vs Simulated Bitrate Reduct : {row.get('orig_vs_sim_bitrate_reduction_pct')}% ({row.get('orig_vs_sim_bitrate_reduction_kbps')} kbps)")
    print(f"  * Orig vs Simulated Resolution Mat : {row.get('orig_vs_sim_resolution_match')}")

    if wa_avail:
        print(f"  * Orig vs WhatsApp Size Reduction  : {row.get('orig_vs_wa_size_reduction_pct')}% ({row.get('orig_vs_wa_size_reduction_bytes'):,} bytes)")
        print(f"  * Orig vs WhatsApp Bitrate Reduct  : {row.get('orig_vs_wa_bitrate_reduction_pct')}% ({row.get('orig_vs_wa_bitrate_reduction_kbps')} kbps)")
        print(f"  * Orig vs WhatsApp Resolution Mat  : {row.get('orig_vs_wa_resolution_match')}")
        print("-" * 80)
        print("SIMULATOR CALIBRATION DELTA (Simulated vs Actual WhatsApp):")
        print(f"  * Size Delta (Sim - WA)            : {row.get('sim_vs_wa_size_diff_bytes'):,} bytes ({row.get('sim_vs_wa_size_diff_pct')}%)")
        print(f"  * Bitrate Delta (Sim - WA)         : {row.get('sim_vs_wa_bitrate_diff_kbps')} kbps")
        print(f"  * Resolution Match                 : {row.get('sim_vs_wa_resolution_match')}")
        print(f"  * Video Codec Match                : {row.get('sim_vs_wa_vcodec_match')}")
        print(f"  * FPS Delta                        : {row.get('sim_vs_wa_fps_diff')}")
    else:
        print("  * Actual WhatsApp comparison       : PENDING (copy received video to output/whatsapp/)")

    print("-" * 80)
    print("NOTE: " + row["research_disclaimer"])
    print("=" * 80 + "\n")


def find_matching_file(directory: Path, stem: str, suffix_tag: str = "") -> Optional[Path]:
    """Find a video file in directory matching the stem or stem + suffix_tag."""
    if not directory.is_dir():
        return None
    candidates = [stem]
    if suffix_tag:
        candidates.append(f"{stem}{suffix_tag}")
    for ext in SUPPORTED_EXTENSIONS:
        for c in candidates:
            p = directory / f"{c}{ext}"
            if p.is_file():
                return p
    return None


def run_comparisons(
    original_path: Optional[Path],
    simulated_path: Optional[Path],
    whatsapp_path: Optional[Path],
    sample_id: Optional[str],
    batch: bool,
    input_dir: Path,
    simulated_dir: Path,
    whatsapp_dir: Path,
    results_dir: Path,
    ffprobe_path: str
):
    """Execute Mode B comparison on explicit paths, sample ID, or batch."""
    comparisons_csv = results_dir / "comparisons.csv"
    existing_rows: List[Dict[str, Any]] = []

    # Read existing comparisons if CSV exists to avoid losing previous comparisons
    if comparisons_csv.is_file():
        try:
            import pandas as pd
            existing_df = pd.read_csv(comparisons_csv)
            existing_rows = existing_df.to_dict(orient="records")
        except Exception:
            pass

    triplets_to_compare: List[Tuple[Path, Path, Optional[Path], str]] = []

    if original_path and simulated_path:
        sid = sample_id or original_path.stem
        triplets_to_compare.append((original_path, simulated_path, whatsapp_path, sid))

    elif sample_id:
        orig = find_matching_file(input_dir, sample_id)
        sim = find_matching_file(simulated_dir, sample_id, suffix_tag="_simulated")
        wa = find_matching_file(whatsapp_dir, sample_id, suffix_tag="_whatsapp") or find_matching_file(whatsapp_dir, sample_id)

        if not orig:
            sys.stderr.write(f"ERROR: Could not find original video for sample ID '{sample_id}' in '{input_dir}'\n")
            sys.exit(1)
        if not sim:
            sys.stderr.write(f"ERROR: Could not find simulated video for sample ID '{sample_id}' in '{simulated_dir}'\n")
            sys.exit(1)
        triplets_to_compare.append((orig, sim, wa, sample_id))

    elif batch:
        # Auto-match every original video with its simulated and whatsapp counterparts
        if not input_dir.is_dir():
            print(f"[WARNING] Input directory '{input_dir}' does not exist.")
            return

        for vid in sorted(input_dir.iterdir()):
            if vid.is_file() and vid.suffix.lower() in SUPPORTED_EXTENSIONS:
                sid = vid.stem
                sim = find_matching_file(simulated_dir, sid, suffix_tag="_simulated")
                wa = find_matching_file(whatsapp_dir, sid, suffix_tag="_whatsapp") or find_matching_file(whatsapp_dir, sid)
                if sim:
                    triplets_to_compare.append((vid, sim, wa, sid))
                else:
                    print(f"[INFO] Skipping '{vid.name}': No simulated video found yet in '{simulated_dir}'.")

        if not triplets_to_compare:
            print("[INFO] No complete triplets or pairs found to compare in batch mode.")
            return
    else:
        sys.stderr.write("ERROR: Please specify --original and --simulated, or --sample-id, or --batch.\n")
        sys.exit(1)

    new_results: List[Dict[str, Any]] = []

    for orig_f, sim_f, wa_f, sid in triplets_to_compare:
        try:
            print(f"[INFO] Comparing sample '{sid}'...")
            orig_meta = extract_video_metadata(ffprobe_path, orig_f, category="original")
            sim_meta = extract_video_metadata(ffprobe_path, sim_f, category="simulated")
            wa_meta = extract_video_metadata(ffprobe_path, wa_f, category="whatsapp") if (wa_f and wa_f.is_file()) else None

            row = compare_triplet(orig_meta, sim_meta, wa_meta, sample_id=sid)
            print_comparison_terminal(row)
            new_results.append(row)
        except Exception as e:
            print(f"[ERROR] Failed to compare sample '{sid}': {e}", file=sys.stderr)

    if new_results:
        # Merge by sample_id
        merged_dict = {r["sample_id"]: r for r in existing_rows if "sample_id" in r}
        for r in new_results:
            merged_dict[r["sample_id"]] = r
        final_list = list(merged_dict.values())

        save_csv(final_list, comparisons_csv)
        print(f"[SUCCESS] Updated comparison table saved to: {comparisons_csv}")


def log_ground_truth_experiment(
    results_dir: Path,
    sample_id: str,
    wa_version: str,
    send_device: str,
    receive_device: str,
    send_method: str,
    notes: str = ""
):
    """
    Record WhatsApp version, device info, send method, and date in an experiment log CSV.
    """
    log_csv = results_dir / "experiment_log.csv"
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample_id": sample_id,
        "whatsapp_version": wa_version,
        "sending_device": send_device,
        "receiving_device": receive_device,
        "send_method": send_method,
        "notes": notes,
    }

    existing = []
    if log_csv.is_file():
        try:
            import pandas as pd
            existing = pd.read_csv(log_csv).to_dict(orient="records")
        except Exception:
            pass

    existing.append(row)
    save_csv(existing, log_csv)
    print("=" * 70)
    print(f"[SUCCESS] Ground-truth experiment logged for sample '{sample_id}'")
    print(f"Log saved to: {log_csv}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="WhatsApp Video Metadata Extraction & Comparison Tool (Research Pipeline 2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Operational mode")

    # Mode A: scan
    scan_parser = subparsers.add_parser("scan", help="Scan folders and extract metadata to CSV and JSON.")
    scan_parser.add_argument("--input-dir", type=str, default="input", help="Directory of original videos.")
    scan_parser.add_argument("--simulated-dir", type=str, default="output/simulated", help="Directory of simulated videos.")
    scan_parser.add_argument("--whatsapp-dir", type=str, default="output/whatsapp", help="Directory of WhatsApp videos.")
    scan_parser.add_argument("--results-dir", type=str, default="results", help="Directory for output results.")

    # Mode B: compare
    comp_parser = subparsers.add_parser("compare", help="Compare original, simulated, and WhatsApp videos.")
    comp_parser.add_argument("--original", type=str, help="Path to original video.")
    comp_parser.add_argument("--simulated", type=str, help="Path to simulated compressed video.")
    comp_parser.add_argument("--whatsapp", type=str, help="Path to actual WhatsApp-transferred video.")
    comp_parser.add_argument("--sample-id", type=str, help="Sample identifier to auto-match files.")
    comp_parser.add_argument("--batch", action="store_true", help="Compare all matched video sets automatically.")
    comp_parser.add_argument("--input-dir", type=str, default="input", help="Directory of original videos.")
    comp_parser.add_argument("--simulated-dir", type=str, default="output/simulated", help="Directory of simulated videos.")
    comp_parser.add_argument("--whatsapp-dir", type=str, default="output/whatsapp", help="Directory of WhatsApp videos.")
    comp_parser.add_argument("--results-dir", type=str, default="results", help="Directory for output results.")

    # Experiment log helper
    log_parser = subparsers.add_parser("log-experiment", help="Log ground-truth WhatsApp transfer experiment details.")
    log_parser.add_argument("--sample-id", type=str, required=True, help="Sample ID tested.")
    log_parser.add_argument("--whatsapp-version", type=str, required=True, help="WhatsApp app version (e.g., 2.24.15.78).")
    log_parser.add_argument("--send-device", type=str, required=True, help="Sending device model & OS (e.g., iPhone 13 iOS 17.5).")
    log_parser.add_argument("--receive-device", type=str, required=True, help="Receiving device model & OS (e.g., Pixel 7 Android 14).")
    log_parser.add_argument("--send-method", type=str, default="Standard Normal Video (not document)", help="Video sharing method.")
    log_parser.add_argument("--notes", type=str, default="", help="Experimental notes.")
    log_parser.add_argument("--results-dir", type=str, default="results", help="Directory for results.")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    ffprobe_path = get_ffprobe()
    base_dir = Path(__file__).resolve().parent

    def resolve_dir(dir_str: str) -> Path:
        d = Path(dir_str)
        if not d.is_absolute() and not d.exists() and (base_dir / dir_str).exists():
            return (base_dir / dir_str).resolve()
        return d.resolve()

    if args.command == "scan":
        scan_directories(
            input_dir=resolve_dir(args.input_dir),
            simulated_dir=resolve_dir(args.simulated_dir),
            whatsapp_dir=resolve_dir(args.whatsapp_dir),
            results_dir=resolve_dir(args.results_dir),
            ffprobe_path=ffprobe_path,
        )

    elif args.command == "compare":
        orig_p = Path(args.original).resolve() if args.original else None
        sim_p = Path(args.simulated).resolve() if args.simulated else None
        wa_p = Path(args.whatsapp).resolve() if args.whatsapp else None

        run_comparisons(
            original_path=orig_p,
            simulated_path=sim_p,
            whatsapp_path=wa_p,
            sample_id=args.sample_id,
            batch=args.batch,
            input_dir=resolve_dir(args.input_dir),
            simulated_dir=resolve_dir(args.simulated_dir),
            whatsapp_dir=resolve_dir(args.whatsapp_dir),
            results_dir=resolve_dir(args.results_dir),
            ffprobe_path=ffprobe_path,
        )

    elif args.command == "log-experiment":
        log_ground_truth_experiment(
            results_dir=resolve_dir(args.results_dir),
            sample_id=args.sample_id,
            wa_version=args.whatsapp_version,
            send_device=args.send_device,
            receive_device=args.receive_device,
            send_method=args.send_method,
            notes=args.notes,
        )


if __name__ == "__main__":
    main()
