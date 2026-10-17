"""FastAPI backend (Phase 9): upload a video, get the model's verdict,
routing decision, and per-frame saliency.

No real trained checkpoint exists yet (see DECISIONS.md's Phase 9 entry)
-- this runs with the Phase 5 architecture's random initialization and
labels every response accordingly. If data_cache/checkpoints/checkpoint.pt
appears (the location notebooks/train_kaggle.py's instructions point to),
it's loaded automatically and the response reflects that instead.
"""
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from src.models.model import DeepfakeDetector
from src.models.model import load_config as load_model_config
from src.preprocessing.pipeline import load_config as load_preprocessing_config
from src.preprocessing.pipeline import preprocess_video
from src.train.train import load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = REPO_ROOT / "data_cache" / "checkpoints" / "checkpoint.pt"

_state: dict = {}


def _cluster_name_by_index(router_clusters: dict) -> dict[int, str]:
    inv: dict[int, str] = {}
    for name, idx in router_clusters.items():
        if name == "default":
            continue
        inv.setdefault(idx, name)
    return inv


def _load_model() -> None:
    model_config = load_model_config()
    model = DeepfakeDetector(model_config)

    if CHECKPOINT_PATH.is_file():
        load_checkpoint(CHECKPOINT_PATH, model)
        status = f"trained checkpoint loaded: {CHECKPOINT_PATH.relative_to(REPO_ROOT)}"
    else:
        status = "untrained model -- architecture demo only (no checkpoint at data_cache/checkpoints/checkpoint.pt)"

    model.eval()
    _state["model"] = model
    _state["model_config"] = model_config
    _state["preprocessing_config"] = load_preprocessing_config()
    _state["status"] = status
    _state["cluster_names"] = _cluster_name_by_index(model_config["router_clusters"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield
    _state.clear()


app = FastAPI(title="Deepfake Detection API (Phase 9 -- architecture scaffold)", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": _state.get("status", "not loaded")}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model = _state["model"]
    preprocessing_config = _state["preprocessing_config"]
    cluster_names = _state["cluster_names"]

    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        pre = preprocess_video(tmp_path, preprocessing_config)
        rgb = pre["rgb"].unsqueeze(0).clone().requires_grad_(True)  # [1,F,3,H,W]
        dct = pre["dct"].unsqueeze(0)  # [1,F,n_tokens,dct_coeffs]

        out = model(rgb, dct)
        p_fake = torch.sigmoid(out["logits"]).item()

        gate_weights = out["gate_weights"][0].tolist()
        top_expert = int(out["gate_weights"][0].argmax().item())
        predicted_history = cluster_names.get(top_expert, f"expert_{top_expert}")

        out["logits"].sum().backward()
        per_frame_saliency = rgb.grad.abs().mean(dim=(0, 2, 3, 4)).tolist()  # [F], vanilla-gradient saliency

        return JSONResponse(
            {
                "p_fake": p_fake,
                "predicted_history": predicted_history,
                "gate_weights": gate_weights,
                "saliency": per_frame_saliency,
                "model_status": _state["status"],
            }
        )
    finally:
        tmp_path.unlink(missing_ok=True)
