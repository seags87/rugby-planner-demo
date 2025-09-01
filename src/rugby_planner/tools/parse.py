from __future__ import annotations

import os
import logging
from datetime import date as dt_date, datetime as dt_datetime, time as dt_time
from typing import Dict, Optional

try:
    from dateparser.search import search_dates as dp_search_dates  # type: ignore
except Exception:  # pragma: no cover
    dp_search_dates = None  # type: ignore


log = logging.getLogger(__name__)
from langsmith import traceable


@traceable(name="parse.extract_fixture_query", run_type="tool")
def extract_fixture_query(query: str, today: Optional[dt_date] = None) -> Dict[str, Optional[str]]:
    """Use an LLM to extract a target fixture date (YYYY-MM-DD) or opponent name.

    Returns dict with keys: 'date' (YYYY-MM-DD or None), 'opponent' (str or None), 'ha' ('h'|'a'|None).
    If OPENAI_API_KEY is not configured or model fails, returns all None.
    """
    today = today or dt_date.today()
    # Include both keys for backward compatibility (ha) and readability (home_away)
    out: Dict[str, Optional[str]] = {"date": None, "opponent": None, "ha": None, "home_away": None}

    if dp_search_dates is not None:
        try:
            base = dt_datetime.combine(today, dt_time.min)
            results = dp_search_dates(
                query,
                settings={
                    "RELATIVE_BASE": base,
                    "PREFER_DATES_FROM": "future",
                    "RETURN_AS_TIMEZONE_AWARE": False,
                },
                languages=["en"],
            )
            if results:
                _, dt = results[0]
                out["date"] = dt.date().isoformat()
        except Exception:
            pass

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return out
    try:
        # Dependency should be present via project requirements/langgraph.json
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatOpenAI(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            timeout=10,
        )
        system_prompt = (
            "You extract structured info from a user's rugby fixture question. "
            "Output STRICT JSON with keys: date, opponent, ha. "
            "- date must be ISO YYYY-MM-DD or null. Infer the year using today's date: "
            f"{today.isoformat()} and month names. If the date passed this year already, prefer the next year. "
            "- opponent is the opponent team name if mentioned, else null. Return short official name (e.g., 'Ormskirk'). "
            "- ha is 'h' for home or 'a' for away if clearly implied (e.g., 'at Ormskirk' -> 'a', 'home vs X' -> 'h'), else null."
        )
        prefix = "Resolved date: " + out["date"] + "\n\n" if out.get("date") else ""
        user_prompt = (
            prefix + "User query:\n" + query + "\n\n"  # noqa: E501
            "Return only JSON, no extra text. Example: {\"date\": \"2025-10-18\", \"opponent\": \"Ormskirk\", \"ha\": \"h\"}"
        )
        msg = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        content = getattr(msg, "content", "").strip()
        import json
        data = json.loads(content)
        if isinstance(data, dict):
            d = data.get("date")
            if out.get("date") is None and isinstance(d, str) and len(d) >= 10:
                out["date"] = d[:10]
            o = data.get("opponent")
            if isinstance(o, str) and o.strip():
                out["opponent"] = o.strip()
            h = data.get("ha")
            if isinstance(h, str) and h.lower() in ("h", "a"):
                out["ha"] = h.lower()
                out["home_away"] = h.lower()
        return out
    except Exception:
        log.exception("[parse] LLM extract failed")
        return out
