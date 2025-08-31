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
from typing import Optional


def generate_recovery_plan(query: str, *, model: Optional[str] = None) -> str:
    """Generate a concise, practical recovery plan using an LLM if available.

    Returns an "unavailable" message when no API key/provider is configured or when the LLM call fails.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log.warning("[recovery] Missing OPENAI_API_KEY; returning fallback message.")
        return "Recovery guidance currently unavailable (no OpenAI API key configured)."

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatOpenAI(
            api_key=api_key,
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.3,
            timeout=15,
        )
        system_prompt = (
            "You are a rugby physio assistant. Create a short, practical recovery plan. "
            "Assume non-emergency soft-tissue injury unless the user states otherwise. "
            "Always include: phased timeline, pain-free progression gates, and a clear referral disclaimer. "
            "Keep it concise (8–12 bullets)."
        )
        user = (
            f"User context: {query}\n\n"
            "Output format:\n"
            "- Phase 1 (Days X–Y): ...\n"
            "- Phase 2 (Days X–Y): ...\n"
            "- Phase 3 (Week X): ...\n"
            "- Return-to-play criteria: ...\n"
            "- Disclaimer: ...\n"
        )
        msg = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user)])
        content = getattr(msg, "content", None) or ""
        if not content.strip():
            return "Recovery guidance currently unavailable (empty response from model)."
        return content
    except Exception as e:
        log.exception("[recovery] LLM call failed")
        return f"Recovery guidance currently unavailable (LLM request failed: {type(e).__name__}: {e})"
