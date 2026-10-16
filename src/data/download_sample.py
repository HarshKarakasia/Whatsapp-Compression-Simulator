"""Phase 3: pull the exact, user-approved micro-run sample from Kaggle
(configs/dataset_sample.yaml) into data_cache/raw/, and write a small
sample manifest (local path, source_id, label, dataset) for Phase 1-2 to
consume. Tolerant of individual missing files -- logs and continues rather
than aborting the whole batch over one 404.
"""
import json
import sys
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_sample_config(path: Path | str | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "dataset_sample.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _download_one(api, dataset: str, remote_path: str, local_dir: Path) -> Path | None:
    """Download one file. Kaggle names the saved file after its signed
    download URL, not our requested path -- it sometimes comes back as
    '<name>.zip' (a real zip wrapping the single file) instead of '<name>'.
    We diff the directory before/after to find whatever actually landed,
    then unwrap it if it's a zip."""
    local_dir.mkdir(parents=True, exist_ok=True)
    local_name = remote_path.split("/")[-1]
    local_path = local_dir / local_name
    if local_path.is_file():
        return local_path  # already downloaded in a prior run

    before = set(local_dir.iterdir())
    try:
        api.dataset_download_file(dataset, remote_path, path=str(local_dir), force=False, quiet=True)
    except Exception as exc:  # Kaggle API raises plain Exception on 404s
        print(f"  [FAILED] {dataset}:{remote_path} -> {exc}", file=sys.stderr)
        return None

    new_files = set(local_dir.iterdir()) - before
    if not new_files:
        print(f"  [MISSING] {dataset}:{remote_path} -> no file produced", file=sys.stderr)
        return None

    produced = new_files.pop()
    if zipfile.is_zipfile(produced):
        with zipfile.ZipFile(produced) as zf:
            zf.extractall(local_dir)
        produced.unlink()
        if not local_path.is_file():
            print(f"  [MISSING] {dataset}:{remote_path} -> zip didn't contain {local_name}", file=sys.stderr)
            return None
        return local_path
    return produced


def download_sample(config: dict | None = None) -> list[dict]:
    from kaggle.api.kaggle_api_extended import KaggleApi

    config = config or load_sample_config()
    out_dir = REPO_ROOT / config["output_dir"]
    api = KaggleApi()
    api.authenticate()

    rows: list[dict] = []

    ffpp = config["ffpp"]
    for ident, spec in ffpp["identities"].items():
        source_id = f"ffpp_{ident}"
        real_path = _download_one(api, ffpp["dataset"], spec["real"], out_dir / "ffpp" / "original")
        if real_path:
            rows.append({
                "local_path": str(real_path.relative_to(REPO_ROOT)),
                "source_id": source_id, "label": "real", "dataset": "ffpp", "identity": ident,
            })
        # every method's fake file shares the same "<id>_<pair>.mp4" basename,
        # so each method needs its own subdirectory to avoid overwriting.
        for method, remote in spec["fakes"].items():
            fake_path = _download_one(api, ffpp["dataset"], remote, out_dir / "ffpp" / method)
            if fake_path:
                rows.append({
                    "local_path": str(fake_path.relative_to(REPO_ROOT)),
                    "source_id": source_id, "label": "fake", "dataset": "ffpp",
                    "identity": ident, "method": method,
                })

    celebdf = config["celebdf"]
    for ident, spec in celebdf["identities"].items():
        source_id = f"celebdf_{ident}"
        local_dir = out_dir / "celebdf"
        real_path = _download_one(api, celebdf["dataset"], spec["real"], local_dir)
        if real_path:
            rows.append({
                "local_path": str(real_path.relative_to(REPO_ROOT)),
                "source_id": source_id, "label": "real", "dataset": "celebdf", "identity": ident,
            })
        for remote in spec["fakes"]:
            fake_path = _download_one(api, celebdf["dataset"], remote, local_dir)
            if fake_path:
                rows.append({
                    "local_path": str(fake_path.relative_to(REPO_ROOT)),
                    "source_id": source_id, "label": "fake", "dataset": "celebdf", "identity": ident,
                })

    manifest_path = out_dir / "sample_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(rows, f, indent=2)

    return rows


if __name__ == "__main__":
    rows = download_sample()
    n_real = sum(1 for r in rows if r["label"] == "real")
    n_fake = sum(1 for r in rows if r["label"] == "fake")
    total_bytes = sum((REPO_ROOT / r["local_path"]).stat().st_size for r in rows)
    print(f"downloaded {len(rows)} files ({n_real} real, {n_fake} fake), "
          f"{total_bytes / 1e6:.1f} MB total")
