"""Streamlit UI (Phase 9): upload a video, see the verdict, routing
decision, gate weights, and per-frame saliency from the FastAPI backend.
"""
import os
import re
from pathlib import Path

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Deepfake Detection Demo", page_icon=":mag:")
st.title("Robust Deepfake Detection -- Demo")

with st.sidebar:
    st.caption(f"API: {API_URL}")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.info(health["status"])
    except Exception as exc:
        st.error(f"API unreachable: {exc}")

uploaded = st.file_uploader("Upload a video (.mp4)", type=["mp4"])

if uploaded is not None:
    st.video(uploaded)

    if st.button("Run detection"):
        with st.spinner("Running..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), "video/mp4")}
            result = None
            try:
                resp = requests.post(f"{API_URL}/predict", files=files, timeout=120)
                resp.raise_for_status()
                result = resp.json()
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        if result:
            st.warning(result["model_status"])

            col1, col2 = st.columns(2)
            col1.metric("P(fake)", f"{result['p_fake']:.3f}")
            col2.metric("Predicted processing history", result["predicted_history"])

            # Best-effort: this project's own Phase 1 chain files encode their
            # real lineage in the filename (e.g. "..._h264_G1_h265_G2.mp4").
            # Won't resolve for an arbitrary upload -- that's expected.
            stem = Path(uploaded.name).stem
            matches = re.findall(r"__([a-z0-9]+)_G(\d+)", stem)
            actual_history = f"{matches[-1][0]}_G{matches[-1][1]}" if matches else "unknown (not a lineage-named file from this project)"
            st.caption(f"Actual processing history (inferred from filename, best-effort): {actual_history}")

            st.subheader("Router gate weights")
            st.bar_chart({f"expert {i}": w for i, w in enumerate(result["gate_weights"])})

            st.subheader("Per-frame saliency (vanilla gradient)")
            st.line_chart(result["saliency"])
