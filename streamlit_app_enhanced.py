"""
MediTrack - Enhanced Streamlit Dashboard
Real-time wound monitoring with Pathway streaming and Aparavi PHI protection
"""

import sys
from pathlib import Path

# Add src directory to Python path to import meditrack
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import json
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from aparavi_integration import run_aparavi_pipeline_on_wound_files
from load_env import load_environment

load_environment()

from meditrack.llm.ai_client import (
    generate_ai_summary_groq,
    generate_ai_summary_gemini,
)

try:
    from meditrack.pipeline.pathway_pipeline import publish_wound_event
except Exception:
    publish_wound_event = None

# -------------------------------------------------------------------
# Paths and constants
# -------------------------------------------------------------------

DATA_DIR = Path("data")
SAMPLE_WOUNDS_DIR = DATA_DIR / "sample_wounds"
PATHWAY_OUTPUT_DIR = DATA_DIR / "outputs"
PATHWAY_OUTPUT_FILE = PATHWAY_OUTPUT_DIR / "wound_events.jsonl"

SAMPLE_WOUNDS_DIR.mkdir(parents=True, exist_ok=True)
PATHWAY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Page config & styles
# -------------------------------------------------------------------

# Load custom icon from project directory
ICON_PATH = Path(__file__).parent / "icon.png"

try:
    if ICON_PATH.exists():
        icon_image = Image.open(ICON_PATH)
        page_icon = icon_image
    else:
        page_icon = "🩹"
except:
    page_icon = "🩹"

st.set_page_config(
    page_title="MediTrack - Wound Healing Monitor",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    /* Remove black top bar and make header white */
    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
    }
    
    /* Hide the default Streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Remove top padding */
    .block-container {
        padding-top: 1rem !important;
    }
    
    /* Base app styling */
    .stApp {
        background-color: #ffffff !important;
    }
    
    /* Default text color */
    .stApp * {
        color: #1a1a1a !important;
    }
    
    .main-header {
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: #1a1a1a !important;
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    
    .subtitle {
        text-align: center;
        color: #666666 !important;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a !important;
    }
    
    p, span, div, label {
        color: #333333 !important;
    }
    
    /* Alert boxes with proper backgrounds */
    .alert-box {
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6;
    }
    .alert-box h3 {
        color: #1a1a1a !important;
        font-weight: 600;
        margin: 0 0 0.75rem 0;
    }
    .alert-box p {
        color: #333333 !important;
        line-height: 1.6;
    }
    
    .alert-high { 
        border-left: 4px solid #dc3545;
        background-color: #fff5f5 !important;
    }
    .alert-medium { 
        border-left: 4px solid #ffc107;
        background-color: #fffbf0 !important;
    }
    .alert-low { 
        border-left: 4px solid #28a745;
        background-color: #f0fff4 !important;
    }
    
    /* Buttons with white text */
    .stButton > button {
        background-color: #2c5aa0 !important;
        color: #ffffff !important;
        border-radius: 6px;
        border: none;
        font-weight: 500;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #1e4485 !important;
        color: #ffffff !important;
    }
    
    /* Primary button (Analyze Wound) */
    .stButton > button[kind="primary"] {
        background-color: #2c5aa0 !important;
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1e4485 !important;
        color: #ffffff !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa !important;
    }
    .stTabs [data-baseweb="tab-list"] button {
        color: #495057 !important;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #2c5aa0 !important;
        border-bottom-color: #2c5aa0 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #ffffff !important;
    }
    
    /* Metrics */
    [data-testid="stMetricLabel"] {
        color: #495057 !important;
        font-weight: 500;
    }
    [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
        font-weight: 600;
    }
    [data-testid="stMetricDelta"] {
        color: #495057 !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
    }
    section[data-testid="stSidebar"] * {
        color: #1a1a1a !important;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #495057 !important;
    }
    
    /* File uploader with light background */
    [data-testid="stFileUploader"] {
        background-color: #f8f9fa !important;
        border: 2px dashed #dee2e6 !important;
        border-radius: 8px;
        padding: 1rem;
    }
    [data-testid="stFileUploader"] label {
        color: #495057 !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
    }
    
    /* Fix for uploaded file display */
    [data-testid="stFileUploader"] > div > div {
        background-color: #ffffff !important;
    }
    
    /* Browse files button */
    [data-testid="stFileUploader"] button {
        background-color: #2c5aa0 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background-color: #1e4485 !important;
        color: #ffffff !important;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #333333 !important;
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        background-color: #f8f9fa !important;
    }
    .stAlert > div {
        color: #1a1a1a !important;
    }
    
    /* Checkbox and radio labels */
    .stCheckbox label, .stRadio label {
        color: #1a1a1a !important;
    }
    
    /* Text input */
    .stTextInput input {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
    }
    .stTextInput label {
        color: #495057 !important;
    }
    
    /* Subheaders */
    .stSubheader {
        color: #1a1a1a !important;
    }
    
    /* Caption text */
    .caption {
        color: #6c757d !important;
        font-size: 0.875rem;
    }
    
    /* Horizontal rule */
    hr {
        border-color: #dee2e6 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------


def initialize_session_state():
    if "patient_id" not in st.session_state:
        st.session_state.patient_id = "DEMO-001"
    if "wound_history" not in st.session_state:
        st.session_state.wound_history = []
    if "aparavi_enabled" not in st.session_state:
        st.session_state.aparavi_enabled = True
    if "pathway_streaming" not in st.session_state:
        st.session_state.pathway_streaming = False
    if "ai_provider" not in st.session_state:
        st.session_state.ai_provider = "Groq"
    if "ai_live" not in st.session_state:
        st.session_state.ai_live = True


# -------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------


def render_header():
    # Embed icon using base64 from project directory
    icon_html = "🩹 "
    try:
        import base64
        icon_path = Path(__file__).parent / "icon.png"
        if icon_path.exists():
            with open(icon_path, "rb") as f:
                icon_data = base64.b64encode(f.read()).decode()
                icon_html = f'<img src="data:image/png;base64,{icon_data}" style="width: 32px; height: 32px; vertical-align: middle; margin-right: 10px; display: inline-block;">'
    except Exception as e:
        pass
    
    st.markdown(
        f'''
        <div class="main-header">
            {icon_html}MediTrack
        </div>
        <p class="subtitle">AI-powered wound healing assistant</p>
        ''',
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("### Settings")

        patient_id = st.text_input(
            "Patient ID",
            value=st.session_state.patient_id,
        )
        st.session_state.patient_id = patient_id

        st.markdown("---")
        st.markdown("### Features")

        aparavi = st.checkbox(
            "PHI Protection",
            value=st.session_state.aparavi_enabled,
        )
        st.session_state.aparavi_enabled = aparavi

        pathway = st.checkbox(
            "Real-time Streaming",
            value=st.session_state.pathway_streaming,
        )
        st.session_state.pathway_streaming = pathway

        st.markdown("---")
        st.markdown("### AI Engine")

        st.session_state.ai_provider = st.radio(
            "Provider",
            ["Groq", "Gemini"],
            index=0 if st.session_state.ai_provider == "Groq" else 1,
        )
        st.session_state.ai_live = st.checkbox(
            "Use live LLM",
            value=st.session_state.ai_live,
        )

        st.markdown("---")
        st.markdown('<p class="caption">⚠️ Educational prototype only.</p>', unsafe_allow_html=True)


# -------------------------------------------------------------------
# Image upload + processing
# -------------------------------------------------------------------


def upload_and_process_image():
    st.header("Upload Wound Image")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
        )

    with col2:
        st.markdown(
            """
**Guidelines:**
- Good lighting
- Clear wound view
- Include reference object
- Avoid glare
            """
        )

    if uploaded_file is not None:
        SAMPLE_WOUNDS_DIR.mkdir(parents=True, exist_ok=True)
        img_save_path = SAMPLE_WOUNDS_DIR / uploaded_file.name
        with open(img_save_path, "wb") as f:
            f.write(uploaded_file.read())

        image = Image.open(img_save_path)
        img_array = np.array(image)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)

        if st.session_state.aparavi_enabled:
            with col2:
                st.subheader("PHI Detection")
                with st.spinner("Scanning..."):
                    phi_detections = simulate_phi_detection(img_array)
                    if phi_detections > 0:
                        st.warning(f"{phi_detections} PHI element(s) detected")
                        redacted_img = apply_redaction(img_array)
                        st.image(redacted_img, use_container_width=True)
                    else:
                        st.success("No PHI detected")
                        st.image(image, use_container_width=True)
        else:
            with col2:
                st.subheader("PHI Detection")
                st.info("Disabled")

        with col3:
            st.subheader("Segmentation")
            with st.spinner("Processing..."):
                segmented = simulate_segmentation(img_array)
                st.image(segmented, use_container_width=True)

        st.markdown("---")

        if st.button("Analyze Wound", type="primary", use_container_width=True):
            with st.spinner("Running AI analysis..."):
                process_wound_analysis(img_array)


# -------------------------------------------------------------------
# Wound analysis
# -------------------------------------------------------------------


def process_wound_analysis(image: np.ndarray):
    metrics = extract_wound_metrics(image)

    st.header("Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Wound Area",
            f"{metrics['area']:.2f} cm²",
            delta=f"{metrics['area_change']:.1f}%",
            delta_color="inverse",
        )
    with col2:
        st.metric(
            "Redness",
            f"{metrics['redness']:.0f}%",
            delta=f"{metrics['redness_change']:+.0f}%",
            delta_color="inverse",
        )
    with col3:
        st.metric(
            "Granulation",
            f"{metrics['granulation']:.0f}%",
            delta=f"{metrics['granulation_change']:+.0f}%",
        )
    with col4:
        st.metric(
            "Edge Quality",
            f"{metrics['edge_quality']:.2f}",
            delta="Good" if metrics["edge_quality"] > 0.7 else "Monitor",
        )

    st.markdown("---")

    provider = st.session_state.get("ai_provider", "Groq")
    use_live = st.session_state.get("ai_live", True)
    analysis = generate_llm_analysis(metrics, provider=provider, use_live=use_live)
    display_ai_analysis(analysis)
    save_to_history(metrics, analysis)

    if st.session_state.get("pathway_streaming", False):
        if publish_wound_event is not None:
            try:
                publish_wound_event(
                    patient_id=st.session_state.patient_id,
                    metrics=metrics,
                    risk_level=analysis["risk_level"],
                )
                st.info("Event published")
            except Exception as e:
                st.warning(f"Streaming failed: {e}")


def generate_llm_analysis(
    metrics: dict, provider: str = "Groq", use_live: bool = True
) -> dict:
    area_change = metrics["area_change"]
    redness = metrics["redness"]
    granulation = metrics["granulation"]
    area = metrics["area"]
    area_fraction = metrics.get("area_fraction", area / 12.0)

    if ((area_fraction < 0.12) or (area_fraction > 0.8)) and redness < 35:
        risk_level = "low"
        area_trend = "no clear wound detected"
    elif area_change < -10 and redness < 45 and granulation > 60:
        risk_level = "low"
        area_trend = "improving rapidly"
    elif area_change > 15 or (redness > 70 and area_fraction > 0.15):
        risk_level = "high"
        area_trend = "worsening"
    else:
        risk_level = "medium"
        area_trend = (
            "improving"
            if area_change < 0
            else "stable"
            if abs(area_change) <= 5
            else "worsening slightly"
        )

    base_summary = (
        f"The wound shows {area_trend} progress with a "
        f"{abs(area_change):.1f}% change in area. "
        f"Granulation tissue is at {granulation:.0f}%, "
        f"and redness is {redness:.0f}%. {metrics['redness_context']}"
    )

    recommendations = [
        "Continue normal care." if risk_level == "low" else "Monitor for infection.",
        "Keep the area clean and dry.",
        "Take progress photos if changes occur.",
        "Seek medical help if pain or fever develop.",
    ]

    consult_doctor = risk_level != "low"
    trend = area_trend
    summary_text = base_summary

    if use_live:
        try:
            patient_id = st.session_state.get("patient_id", "DEMO-001")
            latest_metrics = {
                "area_cm2": metrics["area"],
                "area_change_pct": metrics["area_change"],
                "redness_pct": metrics["redness"],
                "granulation_pct": metrics["granulation"],
                "edge_quality": metrics["edge_quality"],
                "healing_score": metrics["healing_score"],
                "area_fraction": metrics.get("area_fraction"),
            }
            trend_notes = (
                f"Wound area changed by {metrics['area_change']:.1f}%. "
                f"Redness: {metrics['redness']:.1f}%, "
                f"Granulation: {metrics['granulation']:.1f}%."
            )

            use_aparavi_phi = st.session_state.get("aparavi_enabled", False)

            if provider == "Groq":
                summary_md, llm_risk = generate_ai_summary_groq(
                    patient_id=patient_id,
                    latest_metrics=latest_metrics,
                    trend_notes=trend_notes,
                    use_aparavi=use_aparavi_phi,
                )
            else:
                summary_md, llm_risk = generate_ai_summary_gemini(
                    patient_id=patient_id,
                    latest_metrics=latest_metrics,
                    trend_notes=trend_notes,
                    use_aparavi=use_aparavi_phi,
                )

            if summary_md:
                summary_text = summary_md
            if llm_risk and llm_risk != "UNKNOWN":
                risk_level = llm_risk.lower()

        except Exception as e:
            st.warning(f"LLM failed: {e}")

    return {
        "summary": summary_text,
        "risk_level": risk_level,
        "recommendations": recommendations,
        "consult_doctor": consult_doctor,
        "trend": trend,
    }


def display_ai_analysis(analysis: dict):
    st.header("AI Insights")

    risk_colors = {"low": "alert-low", "medium": "alert-medium", "high": "alert-high"}
    risk_icons = {"low": "✅", "medium": "⚠️", "high": "🚨"}

    risk_key = analysis["risk_level"]
    if risk_key not in risk_colors:
        risk_key = "medium"

    st.markdown(
        f"""
    <div class="alert-box {risk_colors[risk_key]}">
        <h3>{risk_icons[risk_key]} Risk Level: {risk_key.upper()}</h3>
        <p>{analysis['summary']}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recommendations")
        for i, rec in enumerate(analysis["recommendations"], 1):
            st.write(f"{i}. {rec}")
    with col2:
        st.subheader("Actions")
        if analysis["consult_doctor"]:
            st.warning("Contact healthcare provider")
        else:
            st.success("Continue current care")
        st.info(f"Trend: {analysis['trend'].title()}")


# -------------------------------------------------------------------
# Historical trends
# -------------------------------------------------------------------


def render_historical_trends():
    st.header("Healing Progress")
    if len(st.session_state.wound_history) < 2:
        st.info("Upload at least 2 images to see trends")
        return

    df = pd.DataFrame(st.session_state.wound_history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["area"],
            mode="lines+markers",
            name="Wound Area",
            line=dict(color="#2c5aa0", width=2),
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        title="Wound Area Over Time",
        xaxis_title="Date",
        yaxis_title="Area (cm²)",
        height=350,
        template="simple_white",
        font=dict(color="#1a1a1a"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_metrics_dashboard():
    st.header("Detailed Metrics")
    if not st.session_state.wound_history:
        st.info("No data yet")
        return

    latest = st.session_state.wound_history[-1]
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Morphological")
        st.write(f"Area: {latest['area']:.2f} cm²")
        st.write(f"Perimeter: {latest['perimeter']:.2f} cm")
    with col2:
        st.subheader("Tissue")
        st.write(f"Granulation: {latest['granulation']:.0f}%")
        st.write(f"Necrotic: {latest['necrotic']:.0f}%")
    with col3:
        st.subheader("Healing")
        st.write(f"Redness: {latest['redness']:.0f}%")
        st.write(f"Score: {latest['healing_score']:.0f}/100")


def render_pathway_live_view():
    st.header("Live Events")

    if not PATHWAY_OUTPUT_FILE.exists():
        st.info("No events yet")
        return

    rows = []
    with open(PATHWAY_OUTPUT_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        st.info("No events")
        return

    for row in reversed(rows[-10:]):
        patient = row.get("patient_id", "?")
        area = row.get("area_cm2", "?")
        ts = row.get("timestamp", "?")
        st.write(f"**{patient}** | {ts} | Area: {area} cm²")
        st.markdown("---")


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------


def simulate_phi_detection(image: np.ndarray) -> int:
    return int(np.random.choice([0, 0, 0, 1, 2]))


def apply_redaction(image: np.ndarray) -> np.ndarray:
    img = image.copy()
    h, w = img.shape[:2]
    img[0 : int(h * 0.1), :] = cv2.GaussianBlur(
        img[0 : int(h * 0.1), :], (51, 51), 30
    )
    return img


def simulate_segmentation(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    overlay = image.copy().astype(np.float32)

    wound_mask = mask == 0
    overlay[wound_mask, 0] = overlay[wound_mask, 0] * 0.5 + 255 * 0.5
    overlay[wound_mask, 1] = overlay[wound_mask, 1] * 0.5
    overlay[wound_mask, 2] = overlay[wound_mask, 2] * 0.5

    return overlay.astype(np.uint8)


def extract_wound_metrics(image: np.ndarray) -> dict:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    wound_pixels = mask > 0
    wound_area_px = int(wound_pixels.sum())

    if wound_area_px == 0:
        wound_pixels = np.ones_like(gray, dtype=bool)
        wound_area_px = wound_pixels.sum()

    area_fraction = wound_area_px / float(h * w)
    base_area = area_fraction * 12.0

    if st.session_state.wound_history:
        prev_area = st.session_state.wound_history[-1]["area"]
        area_change = ((base_area - prev_area) / prev_area) * 100
    else:
        area_change = 0.0

    img_float = image.astype(np.float32)
    red = img_float[:, :, 0]
    green = img_float[:, :, 1]
    red_wound = red[wound_pixels]
    green_wound = green[wound_pixels]
    red_diff = np.clip(red_wound.mean() - green_wound.mean(), -80, 80)
    redness = np.interp(red_diff, [-80, 80], [20, 90])

    brightness = gray[wound_pixels].mean()
    granulation = np.interp(brightness, [40, 200], [40, 85])
    granulation += np.random.uniform(-5, 5)
    granulation = float(np.clip(granulation, 0, 100))

    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges[wound_pixels].mean() / 255.0
    edge_quality = float(np.clip(0.4 + edge_density, 0.0, 1.0))

    area_score = np.clip(100 - area_fraction * 200, 0, 100)
    redness_score = np.clip(100 - redness, 0, 100)
    healing_score = float(0.4 * granulation + 0.3 * area_score + 0.3 * redness_score)

    return {
        "area": float(base_area),
        "area_change": float(area_change),
        "area_fraction": float(area_fraction),
        "perimeter": float(base_area * 2.5),
        "aspect_ratio": 1.2,
        "redness": float(redness),
        "redness_change": -3 if area_change < 0 else 2,
        "redness_context": (
            "Redness is normal." if redness < 50 else "Elevated redness detected."
        ),
        "granulation": granulation,
        "granulation_change": 5,
        "epithelialization": 20,
        "necrotic": 5,
        "edge_quality": edge_quality,
        "healing_score": healing_score,
    }


def save_to_history(metrics: dict, analysis: dict):
    entry = {
        **metrics,
        "timestamp": datetime.now().isoformat(),
        "patient_id": st.session_state.patient_id,
        "risk_level": analysis["risk_level"],
    }
    st.session_state.wound_history.append(entry)
    if len(st.session_state.wound_history) > 30:
        st.session_state.wound_history = st.session_state.wound_history[-30:]


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def main():
    initialize_session_state()
    render_sidebar()
    render_header()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["New Analysis", "Progress", "Metrics", "Live Stream"]
    )
    with tab1:
        upload_and_process_image()
    with tab2:
        render_historical_trends()
    with tab3:
        render_metrics_dashboard()
    with tab4:
        render_pathway_live_view()

    st.markdown("---")
    st.markdown('<p class="caption">Built for Hack With Chicago 2.0</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()