"""
Lightweight Pathway pipeline for MediTrack - Wound Healing Monitor

- NO heavy LLM xPacks or docling/unstructured imports.
- Only uses core Pathway streaming.

Pipeline:
  * Reads Aparavi-enriched JSON files from:
        data/processed/aparavi_results/*.json
  * Treats them as a live stream of wound events
  * Writes them to:
        data/outputs/wound_events.jsonl

The Streamlit app:
  * Calls Groq / Gemini for AI explanations.
  * Reads this JSONL file in the "📡 Pathway Stream" tab.
"""

import os
from pathlib import Path
from typing import Any, Dict

import pathway as pw
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------
# Directories
# --------------------------------------------------------------------

BASE_DATA_DIR = Path("data")
PROCESSED_DATA_DIR = Path(
    os.getenv("PROCESSED_DATA_DIR", BASE_DATA_DIR / "processed")
)
APARAVI_RESULTS_DIR = PROCESSED_DATA_DIR / "aparavi_results"

OUTPUT_DIR = Path(os.getenv("PATHWAY_OUTPUT_DIR", BASE_DATA_DIR / "outputs"))
OUTPUT_FILE = OUTPUT_DIR / "wound_events.jsonl"

APARAVI_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Schema for Aparavi-enriched JSON
# Adjust field names if your Aparavi pipeline uses different keys.
# --------------------------------------------------------------------


class WoundDocSchema(pw.Schema):
    # IDs
    patient_id: str
    doc_id: str

    # Wound metadata (tune to your JSON)
    wound_stage: str
    redness_score: float
    area_cm2: float
    infection_risk_flag: bool

    # Timestamp of capture / analysis (string is fine)
    timestamp: str

    # Free-text description / OCR summary from Aparavi
    text_summary: str


# --------------------------------------------------------------------
# Public function: publish_wound_event
# (used by Streamlit app; still just a logging stub)
# --------------------------------------------------------------------


def publish_wound_event(
    patient_id: str,
    metrics: Dict[str, Any],
    risk_level: str,
) -> None:
    """
    Lightweight stub to keep Streamlit happy.

    The Streamlit app calls this when "Pathway Streaming" is enabled.
    For now, we just log to stdout.

    In a future version, this could push CV-derived metrics into
    a Kafka topic or socket, and a Pathway connector could read them.
    """
    print(
        "[Pathway publish_wound_event] "
        f"patient_id={patient_id}, risk_level={risk_level}, metrics={metrics}"
    )


# --------------------------------------------------------------------
# Pathway pipeline (no LLM inside Pathway)
# --------------------------------------------------------------------


def build_pipeline() -> pw.Table:
    """
    Build a simple Pathway streaming pipeline:

    1. Read Aparavi-enriched JSON files from APARAVI_RESULTS_DIR
       in streaming mode.
    2. Optionally do light transformations.
    3. Emit a live table that Streamlit can read.
    """

    # 1) Streaming read from directory
    docs = pw.io.fs.read(
        path=str(APARAVI_RESULTS_DIR),
        format="json",
        schema=WoundDocSchema,
        mode="streaming",  # new files are picked up incrementally
        object_pattern="*.json",
    )

    # 2) For now, we just pass through fields as-is.
    events = docs.select(
        patient_id=pw.this.patient_id,
        doc_id=pw.this.doc_id,
        wound_stage=pw.this.wound_stage,
        redness_score=pw.this.redness_score,
        area_cm2=pw.this.area_cm2,
        infection_risk_flag=pw.this.infection_risk_flag,
        timestamp=pw.this.timestamp,
        summary=pw.this.text_summary,  # used by the Streamlit "Pathway Stream" tab
    )

    return events


def run_pipeline() -> None:
    """
    Build the pipeline and write resulting table to JSONL
    so the Streamlit app can read `data/outputs/wound_events.jsonl`.
    """
    table = build_pipeline()

    pw.io.fs.write(
        table=table,
        filename=str(OUTPUT_FILE),
        format="json",
    )

    print(
        f"[Pathway] Streaming pipeline started.\n"
        f"Reading Aparavi JSON from: {APARAVI_RESULTS_DIR}\n"
        f"Writing live wound events to: {OUTPUT_FILE}\n"
        "Leave this process running while you use the Streamlit UI."
    )

    pw.run()


if __name__ == "__main__":
    run_pipeline()
