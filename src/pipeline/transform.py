"""FFmpeg multi-codec, multi-generation transformation chain (Phase 1).

Builds a tree of re-encoded variants of a source video: each "hop" branches
into every codec in configs/transform.yaml, so depth=3 with 4 codecs
produces 4 + 16 + 64 = 84 output files.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | str | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "transform.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_paths(path: Path | str | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "paths.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _scale_filter(max_width: int, max_height: int) -> str:
    # Never upscale, preserve aspect ratio, enforce even dimensions
    # (matches Harsh's WhatsApp simulator's own scaling convention).
    return (
        f"scale=w='min(iw,{max_width})':h='min(ih,{max_height})':"
        "force_original_aspect_ratio=decrease,"
        "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    )


def _run_ffmpeg_hop(
    input_path: Path, codec: str, recipe: dict, output_path: Path, config: dict, paths_cfg: dict
) -> None:
    encoder = config["codecs"][codec]
    speed_opts = config.get("encoder_speed_opts", {}).get(codec, [])
    bitrate = recipe["bitrate"]

    cmd = [
        paths_cfg["ffmpeg_binary"],
        "-y",
        "-i", str(input_path),
        "-c:v", encoder,
        "-b:v", bitrate,
    ]
    if config.get("supports_maxrate_bufsize", {}).get(codec, True):
        cmd += ["-maxrate", bitrate, "-bufsize", bitrate]
    cmd += [
        "-g", str(recipe["gop"]),
        "-r", str(recipe["fps"]),
        "-vf", _scale_filter(recipe["max_width"], recipe["max_height"]),
        "-pix_fmt", "yuv420p",
        *speed_opts,
        "-an",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _run_whatsapp_simulator(
    input_path: Path, recipe: dict, output_path: Path, wa_cfg: dict, paths_cfg: dict
) -> None:
    script = REPO_ROOT / wa_cfg["script"]
    with tempfile.TemporaryDirectory() as tmp_dir:
        cmd = [
            "python3", str(script), str(input_path),
            "--output-dir", tmp_dir,
            "--max-width", str(recipe["max_width"]),
            "--max-height", str(recipe["max_height"]),
            "--fps", str(recipe["fps"]),
            "--video-bitrate", recipe["bitrate"],
            "--max-bitrate", recipe["bitrate"],
        ]
        env = dict(os.environ)
        env["PATH"] = f"{Path(paths_cfg['ffmpeg_binary']).parent}:{env.get('PATH', '')}"
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)

        produced = list(Path(tmp_dir).glob("*_simulated.mp4"))
        if not produced:
            raise RuntimeError(f"whatsapp simulator produced no output for {input_path}")
        shutil.move(str(produced[0]), str(output_path))


def single_hop(
    input_path: Path, codec: str, recipe: dict, output_path: Path, config: dict | None = None
) -> Path:
    """Apply one re-encode hop. Returns output_path."""
    config = config or load_config()
    paths_cfg = load_paths()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wa_cfg = config.get("whatsapp_simulator", {})
    if wa_cfg.get("enabled") and codec == wa_cfg.get("codec"):
        _run_whatsapp_simulator(input_path, recipe, output_path, wa_cfg, paths_cfg)
    else:
        _run_ffmpeg_hop(input_path, codec, recipe, output_path, config, paths_cfg)

    return output_path


def build_chain_tree(source_path: Path, depth: int | None = None, config: dict | None = None) -> list[Path]:
    """Recursively apply all codecs at each depth. Returns all generated output paths."""
    config = config or load_config()
    depth = depth if depth is not None else config["depth"]
    codecs = list(config["codecs"].keys())
    hop_recipes = config["hop_recipes"]
    out_dir = REPO_ROOT / config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(source_path)
    stem = source_path.stem
    outputs: list[Path] = []

    def recurse(current_path: Path, current_depth: int, lineage: str) -> None:
        if current_depth > depth:
            return
        recipe = hop_recipes[current_depth]
        for codec_name in codecs:
            tag = f"{lineage}__{codec_name}_G{current_depth}"
            out_path = out_dir / f"{stem}{tag}.mp4"
            single_hop(current_path, codec_name, recipe, out_path, config=config)
            outputs.append(out_path)
            recurse(out_path, current_depth + 1, tag)

    recurse(source_path, 1, "")
    return outputs
