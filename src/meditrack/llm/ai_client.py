"""
LLM client utilities for MediTrack.

Providers:
- Groq (Llama 3.x)  -> generate_ai_summary_groq
- Gemini            -> generate_ai_summary_gemini

Both return: (summary_markdown: str, risk_level: str)

Aparavi:
- Optional PHI masking of the LLM prompt text when use_aparavi=True.
"""

import os
from typing import Dict, Any, Tuple

from dotenv import load_dotenv
import requests  # used for Aparavi HTTP call


# ---------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------


def _load_env_if_needed() -> None:
    """Idempotent env loader – safe to call multiple times."""
    # Render secret file (if mounted)
    secret_path = "/etc/secrets/imp.env"
    if os.path.exists(secret_path):
        load_dotenv(secret_path)
    else:
        # Local dev: imp.env in project root, then default .env
        if os.path.exists("imp.env"):
            load_dotenv("imp.env")
        else:
            load_dotenv()  # default .env


# ---------------------------------------------------------------------
# Aparavi PHI filter helpers
# ---------------------------------------------------------------------


def get_aparavi_key() -> str | None:
    _load_env_if_needed()
    return os.getenv("APARAVI_API_KEY")


def _apply_aparavi_phi_filter(text: str) -> str:
    """
    Send text to Aparavi for PHI/PII masking.

    IMPORTANT:
    - The URL, payload shape, and response format here are placeholders.
    - Replace `APARAVI_API_URL` or the default URL and JSON schema
      with whatever your actual Aparavi API / SDK expects.

    If anything fails, this function returns the original text.
    """
    api_key = get_aparavi_key()
    if not api_key:
        return text

    # Configure the real URL via env var. This is a placeholder.
    url = os.getenv("APARAVI_API_URL", "https://YOUR_APARAVI_ENDPOINT_HERE")

    payload = {
        "text": text,  # <- change this field name if Aparavi expects a different payload
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        if resp.ok:
            data = resp.json()
            # Adjust this according to Aparavi response schema
            return data.get("filtered_text", data.get("text", text))
        else:
            print(f"[Aparavi] HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"[Aparavi] PHI filter failed: {e}")

    return text


# ---------------------------------------------------------------------
# Groq client (DEFAULT)
# ---------------------------------------------------------------------

try:
    from groq import Groq  # type: ignore
except ImportError:
    Groq = None  # type: ignore


def _get_groq_client() -> "Groq":
    _load_env_if_needed()
    if Groq is None:
        raise RuntimeError(
            "groq package is not installed. Add 'groq' to requirements.txt."
        )
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in the environment.")
    return Groq(api_key=api_key)


def generate_ai_summary_groq(
    patient_id: str,
    latest_metrics: Dict[str, Any],
    trend_notes: str,
    use_aparavi: bool = False,
) -> Tuple[str, str]:
    """
    Generate wound-healing summary using Groq (Llama 3.x).

    Parameters:
        patient_id: ID string (can be PHI, which Aparavi will mask if enabled)
        latest_metrics: dict of numeric metrics
        trend_notes: human-readable description
        use_aparavi: if True, send the prompt text through Aparavi PHI filter

    Returns:
        summary_md: Markdown string
        risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'UNKNOWN'
    """
    client = _get_groq_client()

    system_prompt = (
        "You are an assistant for educational wound-healing monitoring. "
        "You explain wound status in simple, non-alarming language and NEVER "
        "give definitive medical diagnoses. You always remind users to consult "
        "a healthcare professional for any concerns."
    )

    user_prompt = f"""
Patient ID: {patient_id}

Latest computed wound metrics (JSON-like):
{latest_metrics}

Trend notes:
{trend_notes}

Tasks:
1. Briefly summarize how the wound seems to be healing.
2. List 3–5 bullet points in plain language.
3. Provide a qualitative risk level: one of LOW, MEDIUM, or HIGH concern.
4. Add a final line reminding them to contact a clinician for any worries.

Respond in Markdown. At the very end, add a line starting with:
RISK_LEVEL:
and then the risk level in ALL CAPS. Example: RISK_LEVEL: MEDIUM
"""

    if use_aparavi:
        user_prompt = _apply_aparavi_phi_filter(user_prompt)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # works on Groq today
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content.strip()

    # Extract risk level from final line
    risk_level = "UNKNOWN"
    lines = content.splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx].strip()
        if line.upper().startswith("RISK_LEVEL:"):
            risk_level = line.split(":", 1)[1].strip().upper()
            # drop that line from Markdown
            content = "\n".join(lines[:idx]).strip()
            break

    return content, risk_level


# ---------------------------------------------------------------------
# Gemini client (optional secondary provider)
# ---------------------------------------------------------------------

try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    genai = None  # type: ignore


def _configure_gemini():
    _load_env_if_needed()
    if genai is None:
        raise RuntimeError(
            "google-generativeai package is not installed. "
            "Add 'google-generativeai' to requirements.txt."
        )
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
    genai.configure(api_key=api_key)


def generate_ai_summary_gemini(
    patient_id: str,
    latest_metrics: Dict[str, Any],
    trend_notes: str,
    use_aparavi: bool = False,
) -> Tuple[str, str]:
    """
    Generate wound-healing summary using Google Gemini.

    Returns:
        summary_md: Markdown string
        risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'UNKNOWN'
    """
    _configure_gemini()

    system_prompt = (
        "You are an assistant for educational wound-healing monitoring. "
        "Explain wound status simply, avoid strong diagnostic language, and "
        "always advise consulting clinicians for decisions."
    )

    prompt = f"""{system_prompt}

Patient ID: {patient_id}

Latest computed wound metrics:
{latest_metrics}

Trend notes:
{trend_notes}

Tasks:
1. Briefly summarize healing status.
2. Give 3–5 clear bullet points.
3. Provide a final line in the format: RISK_LEVEL: LOW/MEDIUM/HIGH
"""

    if use_aparavi:
        prompt = _apply_aparavi_phi_filter(prompt)

    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    content = resp.text.strip()

    # Extract risk level
    risk_level = "UNKNOWN"
    lines = content.splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx].strip()
        if line.upper().startswith("RISK_LEVEL:"):
            risk_level = line.split(":", 1)[1].strip().upper()
            content = "\n".join(lines[:idx]).strip()
            break

    return content, risk_level


# ---------------------------------------------------------------------
# Pathway key helper (for integration in pathway_pipeline.py)
# ---------------------------------------------------------------------


def get_pathway_license_key() -> str | None:
    _load_env_if_needed()
    return os.getenv("PATHWAY_LICENSE_KEY")
