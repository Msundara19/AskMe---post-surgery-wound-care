from typing import Dict

def analyze_wound(metrics: Dict) -> str:
    area = metrics.get("wound_area_mm2", None)
    msg = "Wound analysis summary:\n"
    if area is not None:
        msg += f"- Estimated wound area: {area:.2f} mm²\n"
    msg += "\n(This is a placeholder. Plug in OpenAI/Gemini here.)"
    return msg
