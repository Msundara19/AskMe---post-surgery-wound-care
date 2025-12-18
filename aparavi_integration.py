import os
import json
import shutil
from glob import glob
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------------------
# Optional Aparavi SDK import
# -------------------------------------------------------------------

try:
    # If you have the official Aparavi SDK installed locally,
    # this import will succeed. Otherwise we fall back to demo mode.
    from aparavi_dtc_sdk import AparaviClient  # type: ignore
except Exception:
    AparaviClient = None

APARAVI_BASE_URL = os.getenv("APARAVI_BASE_URL")
APARAVI_API_KEY = os.getenv("APARAVI_API_KEY")

BASE_DATA_DIR = Path("data")
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", BASE_DATA_DIR / "processed"))
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

APARAVI_RESULTS_DIR = PROCESSED_DATA_DIR / "aparavi_results"
APARAVI_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _get_client() -> Optional["AparaviClient"]:
    """Return an AparaviClient if SDK and creds are available, else None."""
    if AparaviClient is None:
        return None
    if not APARAVI_BASE_URL or not APARAVI_API_KEY:
        return None
    return AparaviClient(base_url=APARAVI_BASE_URL, api_key=APARAVI_API_KEY)


# -------------------------------------------------------------------
# Public API used by Streamlit app
# -------------------------------------------------------------------


def run_aparavi_pipeline_on_wound_files(
    file_glob: str,
    pipeline_config: str = "aparavi_pipeline_config.json",
    output_subdir: str = "aparavi_results",
) -> Path:
    """
    Send wound-related files to Aparavi for PHI/PII detection & enrichment.

    In demo/offline mode (no SDK or API key), we simply COPY the files into
    data/processed/aparavi_results so that Pathway still sees "new data" and
    updates the stream.
    """
    out_dir = PROCESSED_DATA_DIR / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    client = _get_client()
    if client is None:
        # Demo / offline mode: mirror input files so Pathway can react
        for f in glob(file_glob):
            src = Path(f)
            dst = out_dir / src.name
            shutil.copy2(src, dst)

        print(
            "[Aparavi] SDK not available or credentials missing – running in "
            f"LOCAL DEMO MODE. Copied files into: {out_dir}"
        )
        return out_dir

    # Real Aparavi call (for when SDK & creds are available)
    result = client.execute_pipeline_workflow(
        pipeline=pipeline_config,
        file_glob=file_glob,
    )
    print("[Aparavi] Pipeline result:", result)
    print("[Aparavi] Ensure pipeline writes structured JSON into", out_dir)
    return out_dir


def write_local_event_for_pathway(
    patient_id: str,
    metrics: dict,
    analysis: dict,
    doc_id: Optional[str] = None,
    output_dir: Path = APARAVI_RESULTS_DIR,
) -> Path:
    """
    Create an 'Aparavi-shaped' JSON event from the CV metrics + LLM analysis
    and write it into data/processed/aparavi_results for Pathway to pick up.

    This simulates the output of an Aparavi enrichment pipeline so the full
    chain works even when we are offline or in demo mode.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if doc_id is None:
        doc_id = f"{patient_id}_{datetime.utcnow().isoformat().replace(':', '-')}"
    now_ts = datetime.utcnow().isoformat()

    trend = analysis.get("trend", "stable")
    risk = analysis.get("risk_level", "medium")

    if risk == "high":
        wound_stage = "critical"
    elif risk == "medium":
        wound_stage = "intermediate"
    else:
        wound_stage = "improving"

    event = {
        "patient_id": patient_id,
        "doc_id": doc_id,
        "wound_stage": wound_stage,
        "redness_score": float(metrics["redness"]),
        "area_cm2": float(metrics["area"]),
        "infection_risk_flag": bool(risk == "high"),
        "timestamp": now_ts,
        "text_summary": analysis["summary"],
    }

    out_path = output_dir / f"{doc_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(event, f, ensure_ascii=False, indent=2)

    print(f"[Aparavi-local] Wrote event for Pathway: {out_path}")
    return out_path


def get_aparavi_enriched_docs(
    aparavi_output_dir: Path = APARAVI_RESULTS_DIR,
    pattern: str = "*.json",
) -> List[Path]:
    """Helper to list Aparavi-like JSON artifacts (for debugging / inspection)."""
    return sorted(aparavi_output_dir.glob(pattern))
