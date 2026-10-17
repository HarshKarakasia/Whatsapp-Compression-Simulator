"""Phase 9 checkpoint tests: FastAPI backend + Streamlit UI.

No real checkpoint exists yet, so every response is the explicit
"untrained model -- architecture demo only" mode -- see DECISIONS.md.
"""
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SYNTHETIC_CLIP = REPO_ROOT / "data_cache" / "synthetic" / "test_input.mp4"
TEST_PORT = 8799
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"


@pytest.fixture(scope="module")
def api_server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--port", str(TEST_PORT)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(60):
            try:
                httpx.get(f"{BASE_URL}/health", timeout=1).raise_for_status()
                break
            except (httpx.ConnectError, httpx.HTTPStatusError):
                time.sleep(0.5)
        else:
            proc.terminate()
            raise RuntimeError("uvicorn did not start in time: " + (proc.stdout.read().decode() if proc.stdout else ""))
        yield BASE_URL
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_uvicorn_starts_and_health_reports_untrained(api_server):
    resp = httpx.get(f"{api_server}/health", timeout=5)
    assert resp.status_code == 200
    assert "untrained model" in resp.json()["status"]


def test_predict_on_synthetic_clip_returns_well_formed_response(api_server):
    assert SYNTHETIC_CLIP.is_file()
    with open(SYNTHETIC_CLIP, "rb") as f:
        resp = httpx.post(
            f"{api_server}/predict",
            files={"file": ("test_input.mp4", f, "video/mp4")},
            timeout=60,
        )
    assert resp.status_code == 200
    data = resp.json()

    assert 0.0 <= data["p_fake"] <= 1.0
    assert isinstance(data["predicted_history"], str) and data["predicted_history"]
    assert isinstance(data["gate_weights"], list) and len(data["gate_weights"]) == 5
    assert sum(1 for w in data["gate_weights"] if w > 0) == 2  # top-2 routing
    assert isinstance(data["saliency"], list) and len(data["saliency"]) == 16
    assert "untrained model" in data["model_status"]


def test_streamlit_app_loads_and_calls_api_end_to_end(api_server, monkeypatch):
    monkeypatch.setenv("API_URL", api_server)
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(REPO_ROOT / "ui" / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception

    clip_bytes = SYNTHETIC_CLIP.read_bytes()
    at.get("file_uploader")[0].upload("test_input.mp4", clip_bytes, "video/mp4")
    at.run()
    assert not at.exception

    at.button[0].click().run()
    assert not at.exception

    body_text = "\n".join(m.value for m in at.markdown) + "\n".join(str(w.value) for w in at.warning)
    assert "untrained model" in body_text
    metric_labels = [m.label for m in at.metric]
    assert "P(fake)" in metric_labels
    assert "Predicted processing history" in metric_labels
