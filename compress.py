#!/usr/bin/env python3
"""
WhatsApp Video Compression Simulator (Research Pipeline 1)

This script simulates the video compression typically applied when videos are shared
through messaging platforms like WhatsApp (as normal videos, not as documents).

IMPORTANT RESEARCH NOTE:
Settings in this simulator are a configurable baseline profile intended for research
calibration and deepfake robustness studies. They are PROVISIONAL baseline parameters,
NOT claimed to be WhatsApp's official proprietary internal settings.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Supported video container formats
SUPPORTED_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".m4v", ".flv", ".wmv", ".ts", ".m2ts"
}

# Configurable Provisional Baseline Profile
# Real-world WhatsApp typically transcodes high-res videos to 720p (or 480p),
# re-encodes to H.264/AVC with yuv420p, targets CRF ~26-30, and encodes audio as AAC.
DEFAULT_CONFIG = {
    "crf": 28,                  # Constant Rate Factor (lower = higher quality, higher = more compression)
    "preset": "medium",         # x264 preset (ultrafast, faster, medium, slow, etc.)
    "max_width": 1280,          # Standard WhatsApp max width (720p landscape)
    "max_height": 720,          # Standard WhatsApp max height (720p landscape)
    "fps": None,                # Frame rate conversion (None = preserve original frame rate)
    "video_bitrate": None,      # Optional target video bitrate cap (e.g., '1200k')
    "max_bitrate": None,        # Optional max bitrate limit
    "bufsize": None,            # Optional buffer size for bitrate control
    "pix_fmt": "yuv420p",       # Standard 8-bit 4:2:0 YUV format required for WhatsApp compatibility
    "audio_codec": "aac",       # 'aac' for re-encoding, or 'copy' for passthrough
    "audio_bitrate": "128k",    # Audio bitrate for AAC
    "faststart": True,          # Enable MP4 +faststart for streaming / messaging optimization
}


def find_binary(binary_name: str) -> Optional[str]:
    """
    Locate an executable binary (e.g. ffmpeg, ffprobe).
    Checks system PATH first, then scans common Windows WinGet / local install paths,
    and inspects Windows registry environment if available.
    """
    # 1. Standard PATH check
    path = shutil.which(binary_name)
    if path:
        return str(Path(path).resolve())

    # 2. Windows registry check (handles fresh installs where terminal hasn't restarted)
    if sys.platform == "win32":
        try:
            import winreg
            for hkey in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    subpath = r"Environment" if hkey == winreg.HKEY_CURRENT_USER else r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
                    with winreg.OpenKey(hkey, subpath) as key:
                        val, _ = winreg.QueryValueEx(key, "Path")
                        for p in val.split(";"):
                            candidate = Path(p.strip()) / f"{binary_name}.exe"
                            if candidate.is_file():
                                # Add to current process PATH
                                os.environ["PATH"] = p.strip() + os.pathsep + os.environ.get("PATH", "")
                                return str(candidate.resolve())
                except OSError:
                    continue
        except ImportError:
            pass

        # 3. Common WinGet installation path in LocalAppData
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            pattern = os.path.join(local_app_data, "Microsoft", "WinGet", "Packages", "**", f"{binary_name}.exe")
            matches = glob.glob(pattern, recursive=True)
            if matches:
                bin_dir = str(Path(matches[0]).parent)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                return str(Path(matches[0]).resolve())

        # 4. Standard Program Files paths
        candidates = [
            Path(r"C:\Program Files\ffmpeg\bin") / f"{binary_name}.exe",
            Path(r"C:\ffmpeg\bin") / f"{binary_name}.exe",
            Path(r"C:\ProgramData\chocolatey\bin") / f"{binary_name}.exe",
        ]
        for candidate in candidates:
            if candidate.is_file():
                bin_dir = str(candidate.parent)
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                return str(candidate.resolve())

    return None


def verify_environment() -> Tuple[str, str]:
    """
    Verify that ffmpeg and ffprobe are available and report helpful errors if missing.
    """
    ffmpeg_path = find_binary("ffmpeg")
    ffprobe_path = find_binary("ffprobe")

    errors = []
    if not ffmpeg_path:
        errors.append(
            "ERROR: 'ffmpeg' executable could not be found.\n"
            "Please ensure FFmpeg is installed and added to your system PATH.\n"
            "On Windows, you can install it using:\n"
            "  winget install Gyan.FFmpeg.Essentials\n"
        )
    if not ffprobe_path:
        errors.append(
            "ERROR: 'ffprobe' executable could not be found.\n"
            "Please ensure FFprobe is installed alongside FFmpeg."
        )

    if errors:
        sys.stderr.write("\n".join(errors) + "\n")
        sys.exit(1)

    return ffmpeg_path, ffprobe_path


def probe_input_video(ffprobe_path: str, video_path: Path) -> Dict[str, Any]:
    """
    Use ffprobe to inspect container and stream information of an input video.
    Returns parsed dictionary or raises RuntimeError.
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
        raise RuntimeError(f"FFprobe failed to inspect {video_path}: {e.stderr.strip()}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"FFprobe returned invalid JSON for {video_path}: {e}")


def has_audio_stream(probe_data: Dict[str, Any]) -> bool:
    """Check if the probed video contains at least one audio stream."""
    streams = probe_data.get("streams", [])
    return any(s.get("codec_type") == "audio" for s in streams)


def build_filter_complex(max_width: Optional[int], max_height: Optional[int], fps: Optional[float]) -> Optional[str]:
    """
    Construct FFmpeg video filter chain preserving aspect ratio and enforcing
    even dimensions required by H.264 (yuv420p).
    """
    filters = []

    # Aspect-ratio preserving downscale:
    # - Only scale down if video dimensions exceed max_width or max_height.
    # - force_original_aspect_ratio=decrease ensures the video fits within the box.
    # - Enforce even dimensions with trunc(iw/2)*2:trunc(ih/2)*2.
    if max_width and max_height:
        scale_filter = (
            f"scale=w='min(iw,{max_width})':h='min(ih,{max_height})':"
            f"force_original_aspect_ratio=decrease,"
            f"scale=trunc(iw/2)*2:trunc(ih/2)*2"
        )
        filters.append(scale_filter)
    elif max_width:
        filters.append(f"scale=w='min(iw,{max_width})':h=-2")
    elif max_height:
        filters.append(f"scale=w=-2:h='min(ih,{max_height})'")

    # Optional frame rate conversion (only applied if explicitly requested)
    if fps is not None and fps > 0:
        filters.append(f"fps={fps}")

    return ",".join(filters) if filters else None


def compress_video(
    input_path: Path,
    output_dir: Path,
    ffmpeg_path: str,
    ffprobe_path: str,
    crf: int = DEFAULT_CONFIG["crf"],
    preset: str = DEFAULT_CONFIG["preset"],
    max_width: Optional[int] = DEFAULT_CONFIG["max_width"],
    max_height: Optional[int] = DEFAULT_CONFIG["max_height"],
    fps: Optional[float] = DEFAULT_CONFIG["fps"],
    video_bitrate: Optional[str] = DEFAULT_CONFIG["video_bitrate"],
    max_bitrate: Optional[str] = DEFAULT_CONFIG["max_bitrate"],
    bufsize: Optional[str] = DEFAULT_CONFIG["bufsize"],
    pix_fmt: str = DEFAULT_CONFIG["pix_fmt"],
    audio_codec: str = DEFAULT_CONFIG["audio_codec"],
    audio_bitrate: str = DEFAULT_CONFIG["audio_bitrate"],
    faststart: bool = DEFAULT_CONFIG["faststart"],
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Compress a single video file according to simulator parameters and write output
    alongside a JSON metadata sidecar.
    """
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Source video not found: {input_path}")

    # Validate file extension
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported video extension '{input_path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    # Prepare output directory
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Target output filename: preserve stem and append _simulated.mp4
    output_filename = f"{input_path.stem}_simulated.mp4"
    output_path = output_dir / output_filename

    # Safety check: Never overwrite or match source video
    if output_path.resolve() == input_path.resolve():
        raise ValueError("Output path cannot be identical to the source video path!")

    # Safety check: Never output directly to whatsapp received directory
    if "whatsapp" in output_path.parts:
        raise ValueError("Cannot write simulated output into the actual 'whatsapp/' directory!")

    # Inspect input video streams
    print(f"\n[INFO] Inspecting input video: {input_path.name}")
    probe_info = probe_input_video(ffprobe_path, input_path)
    audio_present = has_audio_stream(probe_info)

    # Build FFmpeg command arguments list
    cmd: List[str] = [
        ffmpeg_path,
        "-y",               # Overwrite existing simulated file if re-running simulator
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", str(preset),
        "-crf", str(crf),
        "-pix_fmt", str(pix_fmt),
    ]

    # Video filters (scaling, aspect ratio, optional fps)
    vf = build_filter_complex(max_width, max_height, fps)
    if vf:
        cmd.extend(["-vf", vf])

    # Optional bitrate capping
    if video_bitrate:
        cmd.extend(["-b:v", str(video_bitrate)])
    if max_bitrate:
        cmd.extend(["-maxrate", str(max_bitrate)])
        buf = bufsize or str(max_bitrate)
        cmd.extend(["-bufsize", str(buf)])

    # Faststart flag for MP4 container
    if faststart:
        cmd.extend(["-movflags", "+faststart"])

    # Audio handling: check if input actually has audio
    if audio_present:
        if audio_codec == "copy":
            cmd.extend(["-c:a", "copy"])
        else:
            cmd.extend(["-c:a", "aac", "-b:a", str(audio_bitrate)])
    else:
        # Gracefully handle audio-less videos without errors
        cmd.append("-an")

    # Final output path
    cmd.append(str(output_path))

    # Log command and parameters
    cmd_str = subprocess.list2cmdline(cmd)
    print("=" * 70)
    print("WHATSAPP COMPRESSION SIMULATOR (PROVISIONAL RESEARCH BASELINE)")
    print("=" * 70)
    print(f"Input Video   : {input_path}")
    print(f"Output Video  : {output_path}")
    print(f"Has Audio     : {'Yes' if audio_present else 'No'}")
    print(f"CRF           : {crf} | Preset: {preset} | PixFmt: {pix_fmt}")
    print(f"Max Res Box   : {max_width or 'orig'}x{max_height or 'orig'}")
    print(f"FPS Target    : {fps or 'Preserve original'}")
    print(f"Audio Config  : {audio_codec if audio_present else 'None (no audio track)'}")
    print(f"Command       : {cmd_str}")
    print("-" * 70)

    if dry_run:
        print("[DRY-RUN] Command not executed.")
        return {
            "status": "dry_run",
            "command": cmd_str,
            "input": str(input_path),
            "output": str(output_path)
        }

    # Execute FFmpeg
    start_time = time.time()
    status = "failed"
    error_msg = None

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        status = "success"
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip()
        print(f"[ERROR] FFmpeg compression failed:\n{error_msg}", file=sys.stderr)
        raise RuntimeError(f"FFmpeg execution failed: {error_msg}")

    elapsed_time = round(time.time() - start_time, 2)

    # Post-compression verification
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Encoding finished but output file is missing or empty: {output_path}")

    # Verify output readability with ffprobe
    try:
        out_probe = probe_input_video(ffprobe_path, output_path)
    except Exception as e:
        raise RuntimeError(f"Encoded output exists but cannot be parsed by FFprobe: {e}")

    orig_size = input_path.stat().st_size
    comp_size = output_path.stat().st_size
    ratio_pct = round((1.0 - (comp_size / orig_size)) * 100, 2) if orig_size > 0 else 0.0

    print(f"[SUCCESS] Encoding finished in {elapsed_time}s")
    print(f"Original Size : {orig_size:,} bytes ({orig_size / (1024*1024):.2f} MB)")
    print(f"Simulated Size: {comp_size:,} bytes ({comp_size / (1024*1024):.2f} MB)")
    print(f"Size Reduction: {ratio_pct}%")

    # Generate JSON sidecar for full scientific reproducibility
    sidecar_data = {
        "simulation_tag": "simulated_whatsapp_compression",
        "is_simulated": True,
        "provisional_baseline_disclaimer": (
            "Provisional baseline profile for research calibration against WhatsApp. "
            "Not an official WhatsApp specification."
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "input_file": str(input_path),
        "input_filename": input_path.name,
        "input_size_bytes": orig_size,
        "output_file": str(output_path),
        "output_filename": output_path.name,
        "output_size_bytes": comp_size,
        "size_reduction_pct": ratio_pct,
        "execution_time_seconds": elapsed_time,
        "parameters": {
            "encoder": "libx264",
            "crf": crf,
            "preset": preset,
            "max_width": max_width,
            "max_height": max_height,
            "fps": fps,
            "video_bitrate": video_bitrate,
            "max_bitrate": max_bitrate,
            "bufsize": bufsize,
            "pix_fmt": pix_fmt,
            "audio_present": audio_present,
            "audio_codec": audio_codec if audio_present else None,
            "audio_bitrate": audio_bitrate if audio_present else None,
            "faststart": faststart,
        },
        "ffmpeg_command_args": cmd,
        "ffmpeg_command_string": cmd_str,
        "error_message": error_msg,
    }

    sidecar_path = output_dir / f"{output_path.stem}.json"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_data, f, indent=2)

    print(f"Sidecar saved : {sidecar_path}")
    print("=" * 70)

    return sidecar_data


def find_supported_videos(directory: Path) -> List[Path]:
    """Find all supported video files in a given directory."""
    if not directory.is_dir():
        return []
    videos = []
    for item in sorted(directory.iterdir()):
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            videos.append(item)
    return videos


def main():
    parser = argparse.ArgumentParser(
        description="WhatsApp Video Compression Simulator (Research Pipeline 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input modes
    parser.add_argument(
        "input_video",
        nargs="?",
        type=str,
        help="Path to an individual video file to compress.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch process all supported video files in the input directory.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="input",
        help="Directory containing source videos for batch processing.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/simulated",
        help="Directory where compressed simulated videos will be saved.",
    )

    # Compression parameters
    parser.add_argument(
        "--crf",
        type=int,
        default=DEFAULT_CONFIG["crf"],
        help="Constant Rate Factor (x264). Typical WhatsApp baseline ~26-30.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_CONFIG["preset"],
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        help="FFmpeg x264 encoding preset.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=DEFAULT_CONFIG["max_width"],
        help="Maximum output video width (preserves aspect ratio).",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=DEFAULT_CONFIG["max_height"],
        help="Maximum output video height (preserves aspect ratio).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_CONFIG["fps"],
        help="Target frame rate (e.g. 30). If omitted, original frame rate is preserved.",
    )
    parser.add_argument(
        "--video-bitrate",
        type=str,
        default=DEFAULT_CONFIG["video_bitrate"],
        help="Optional target video bitrate cap (e.g., '1000k').",
    )
    parser.add_argument(
        "--max-bitrate",
        type=str,
        default=DEFAULT_CONFIG["max_bitrate"],
        help="Optional max bitrate limit for VBR (e.g., '1500k').",
    )
    parser.add_argument(
        "--pix-fmt",
        type=str,
        default=DEFAULT_CONFIG["pix_fmt"],
        help="Pixel format (yuv420p is standard for WhatsApp).",
    )
    parser.add_argument(
        "--audio-codec",
        type=str,
        default=DEFAULT_CONFIG["audio_codec"],
        choices=["aac", "copy"],
        help="Audio codec to use: 'aac' re-encodes, 'copy' passes through.",
    )
    parser.add_argument(
        "--audio-bitrate",
        type=str,
        default=DEFAULT_CONFIG["audio_bitrate"],
        help="Audio bitrate for AAC encoding (e.g. '128k', '96k', '64k').",
    )
    parser.add_argument(
        "--no-faststart",
        action="store_true",
        help="Disable MP4 faststart (+faststart).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the generated FFmpeg command without executing.",
    )

    args = parser.parse_args()

    # Discover and verify FFmpeg and FFprobe binaries
    ffmpeg_path, ffprobe_path = verify_environment()

    # Determine input videos
    input_paths: List[Path] = []
    base_dir = Path(__file__).resolve().parent

    if args.input_video:
        path = Path(args.input_video)
        if not path.is_absolute() and not path.exists():
            # Check relative to script directory as fallback
            alt_path = base_dir / args.input_video
            if alt_path.exists():
                path = alt_path
        if not path.is_file():
            sys.stderr.write(f"ERROR: Input file not found: {args.input_video}\n")
            sys.exit(1)
        input_paths.append(path)
    elif args.batch:
        input_dir = Path(args.input_dir)
        if not input_dir.is_absolute() and not input_dir.exists():
            alt_dir = base_dir / args.input_dir
            if alt_dir.exists():
                input_dir = alt_dir
        input_paths = find_supported_videos(input_dir)
        if not input_paths:
            print(f"[WARNING] No supported video files found in '{input_dir}'.")
            sys.exit(0)
        print(f"[INFO] Found {len(input_paths)} video(s) for batch compression in '{input_dir}'.")
    else:
        parser.print_help()
        print("\nERROR: Please specify a video file path or use --batch.")
        sys.exit(1)

    # Determine output directory
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute() and not output_dir.exists():
        # Place relative to current directory or base directory
        if not output_dir.parent.exists() and (base_dir / args.output_dir).parent.exists():
            output_dir = base_dir / args.output_dir

    success_count = 0
    failure_count = 0

    for vid in input_paths:
        try:
            compress_video(
                input_path=vid,
                output_dir=output_dir,
                ffmpeg_path=ffmpeg_path,
                ffprobe_path=ffprobe_path,
                crf=args.crf,
                preset=args.preset,
                max_width=args.max_width,
                max_height=args.max_height,
                fps=args.fps,
                video_bitrate=args.video_bitrate,
                max_bitrate=args.max_bitrate,
                pix_fmt=args.pix_fmt,
                audio_codec=args.audio_codec,
                audio_bitrate=args.audio_bitrate,
                faststart=not args.no_faststart,
                dry_run=args.dry_run,
            )
            success_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to process {vid.name}: {e}", file=sys.stderr)
            failure_count += 1

    print("\n" + "=" * 70)
    print(f"BATCH SUMMARY: {success_count} succeeded, {failure_count} failed.")
    print("=" * 70)

    if failure_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
