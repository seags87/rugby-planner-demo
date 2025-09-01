from __future__ import annotations

import os
import logging
from dotenv import load_dotenv

# Ensure environment variables from a local .env are available when running in Studio/API
try:
    load_dotenv()
except Exception:
    pass

log = logging.getLogger(__name__)
from datetime import date
from typing import Dict
from langsmith import traceable


@traceable(name="nutrition.tips", run_type="tool")
def nutrition_tips(event_type: str, when: date, weather: Dict[str, object]) -> str:
    """Return short, practical nutrition and gear tips.

    event_type: "match" | "training" | "general"
    weather: dict with keys: summary, temp_c, precip, wind_kph
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log.warning("[nutrition] Missing OPENAI_API_KEY; returning fallback message.")
        return "Nutrition guidance currently unavailable (no OpenAI API key configured)."

    try:
        # Import the LLM client (dependency should be installed via requirements/langgraph.json)
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatOpenAI(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.3,
            timeout=15,
        )
        system_prompt = (
            "You are a rugby performance nutrition assistant. Provide SHORT, practical tips. "
            "Always include hydration, carbs/protein timing, and specific weather-appropriate gear. "
            "Keep to 6–10 bullets max; no long paragraphs."
        )
        desc = (
            f"Context:\n"
            f"- Event type: {event_type}\n"
            f"- Date: {when.isoformat()}\n"
            f"- Weather: {weather.get('summary')} | {weather.get('temp_c')}°C | rain {int(100*float(weather.get('precip',0)))}% | wind {weather.get('wind_kph')} kph\n\n"
            "Output a concise bullet list only. Example style:\n"
            "- Hydration...\n- Pre-...\n- Gear...\n- Post-...\n"
        )
        msg = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=desc)])
        content = getattr(msg, "content", None) or ""
        if not content.strip():
            return "Nutrition guidance currently unavailable (empty response from model)."
        return content
    except Exception as e:
        log.exception("[nutrition] LLM call failed")
        return f"Nutrition guidance currently unavailable (LLM request failed: {type(e).__name__}: {e})"
