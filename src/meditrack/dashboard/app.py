from pathlib import Path
import streamlit as st
from meditrack.config import CONFIG
from meditrack.pipeline.pathway_pipeline import process_single_image
from meditrack.dashboard.components import show_header, show_image_and_overlay, show_metrics
from meditrack.llm.analyzer import analyze_wound

def main():
    show_header()
    images = list(CONFIG.sample_wounds_dir.glob("*.png"))
    if not images:
        st.warning("No images found in data/sample_wounds")
        return

    selected = st.selectbox("Choose an image", images, format_func=lambda p: p.name)
    if st.button("Analyze"):
        result = process_single_image(selected)
        metrics = {"wound_area_mm2": result["wound_area_mm2"]}
        show_image_and_overlay(Path(result["image_path"]), Path(result["overlay_path"]))
        show_metrics(metrics)
        st.subheader("LLM Analysis")
        st.write(analyze_wound(metrics))

if __name__ == "__main__":
    main()
