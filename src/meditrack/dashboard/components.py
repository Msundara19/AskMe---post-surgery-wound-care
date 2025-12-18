from pathlib import Path
from typing import Dict
import streamlit as st
from PIL import Image

def show_header():
    st.title("🩹 MediTrack: Wound Healing Monitor")
    st.caption("Demo dashboard for wound analysis.")

def show_image_and_overlay(image_path: Path, overlay_path: Path | None):
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Original")
        st.image(Image.open(image_path), use_column_width=True)
    if overlay_path and overlay_path.exists():
        with cols[1]:
            st.subheader("Overlay")
            st.image(Image.open(overlay_path), use_column_width=True)

def show_metrics(metrics: Dict):
    st.subheader("Metrics")
    for k, v in metrics.items():
        st.write(f"**{k}**: {v}")
