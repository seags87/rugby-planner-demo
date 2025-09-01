from __future__ import annotations

# Ensure package imports work when this file is imported by path (e.g., LangGraph Studio)
import sys
from pathlib import Path
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from datetime import date
from typing import Any, Dict
from typing_extensions import TypedDict

from langgraph.graph import StateGraph

from rugby_planner.tools.calendar import (
    infer_location_from_fixture,
    find_next_fixture_across_months,
    find_fixture_on_date,
    find_next_fixture_by_opponent,
)
from rugby_planner.tools.parse import extract_fixture_query
from rugby_planner.tools.weather import get_weather
from rugby_planner.tools.nutrition import nutrition_tips
from rugby_planner.tools.recovery import generate_recovery_plan


class PlannerState(TypedDict, total=False):
    # Original user question
    query: str
    # What we're handling: match | training | general | recovery
    event_type: str
    # ISO date (YYYY-MM-DD)
    date: str
    # Human-friendly location (e.g., "Douglas, Isle of Man")
    location: str
    # Opponent team name, when relevant
    opponent: str
    # Home/Away flag for output: 'H' | 'A'
    home_away: str
    # Weather snapshot used by tips
    weather: Dict[str, Any]
    # Final, user-facing plan text
    plan: str


def classify_node(state: PlannerState) -> PlannerState:
    """Classify the user's query into match/training/general or recovery."""
    ql = (state.get("query") or "").lower()
    if any(w in ql for w in ["injury", "hamstring", "ankle", "acl", "recover", "rehab", "pulled", "sprain", "tear"]):
        state["event_type"] = "recovery"
        return state
    if any(w in ql for w in ["match", "kickoff", "fixture", "game", "play ", "playing", " vs ", "versus"]):
        state["event_type"] = "match"
    elif any(w in ql for w in ["train", "training", "session", "practice"]):
        state["event_type"] = "training"
    else:
        state["event_type"] = "general"
    return state


def event_node(state: PlannerState) -> PlannerState:
    """Populate missing context (date/location/opponent/home_away) before tool calls.

    For matches, we try in this order:
    - exact fixture by date
    - next fixture vs an opponent
    - next suitable fixture scanning months forward (optionally filtered by H/A)
    Then we infer a friendly location and opponent/home_away if still missing.
    """
    event_type = state["event_type"]
    today = date.today()

    q = (state.get("query") or "")
    if not state.get("location"):
        ql = q.lower()
        if "home" in ql:
            state["location"] = "Isle of Man"
        else:
            if "douglas" in ql:
                state["location"] = "Douglas, Isle of Man"
            elif "ramsey" in ql:
                state["location"] = "Ramsey, Isle of Man"
            elif "isle of man" in ql:
                state["location"] = "Isle of Man"

    if event_type == "match":
        qtext = q
        parsed = extract_fixture_query(qtext, today=today)
        # Accept both new key (home_away) and legacy key (ha)
        desired_ha = (parsed.get("home_away") or parsed.get("ha"))
        target_date = parsed.get("date")
        team_hint = parsed.get("opponent")

        fixture = None
        if target_date:
            fixture = find_fixture_on_date(target_date)
        if fixture is None and team_hint:
            fixture = find_next_fixture_by_opponent(team_hint, from_date=today.isoformat(), home_away=desired_ha)
        if fixture is None:
            ql_match = qtext.lower()
            if desired_ha is None:
                if "home" in ql_match:
                    desired_ha = "h"
                elif "away" in ql_match:
                    desired_ha = "a"
            fixture = find_next_fixture_across_months(
                from_date=(target_date or today.isoformat()),
                home_away=desired_ha,
                require_club_is_home=(desired_ha == "h"),
            )
        if fixture:
            dt_str = fixture.get("dateTime", "")[:10]
            if dt_str and not state.get("date"):
                state["date"] = dt_str
            inferred_loc = infer_location_from_fixture(fixture)
            if inferred_loc and inferred_loc != "Unavailable":
                state["location"] = inferred_loc
            elif not state.get("location"):
                state["location"] = inferred_loc
            opponent = fixture.get("opponent")
            if not opponent:
                team_name = fixture.get("teamName")
                home_name = (fixture.get("homeSide") or {}).get("name")
                away_name = (fixture.get("awaySide") or {}).get("name")
                if team_name and home_name and away_name:
                    opponent = away_name if team_name == home_name else home_name
            if opponent:
                state["opponent"] = opponent
            ha = (fixture.get("ha") or "").lower()
            if ha in ("h", "a"):
                state["home_away"] = ha.upper()
            else:
                team_name = fixture.get("teamName")
                home_name = (fixture.get("homeSide") or {}).get("name")
                if team_name and home_name:
                    state["home_away"] = "H" if team_name == home_name else "A"
        elif not state.get("location"):
            state["location"] = "Douglas, Isle of Man"
    elif event_type == "training":
        # Next Tue/Thu from today
        weekday = today.weekday()  # Mon=0, Tue=1, Wed=2, Thu=3, ...
        targets = (1, 3)  # Tue, Thu
        deltas = [((t - weekday) % 7) for t in targets]
        delta = min(deltas) if deltas else 0
        when = today if delta == 0 else date.fromordinal(today.toordinal() + delta)
        state["date"] = when.isoformat()
        state["location"] = "Ramsey, Isle of Man"
    else:
        state.setdefault("location", "Isle of Man")
    return state


def advice_node(state: PlannerState) -> PlannerState:
    et = state.get("event_type", "general")
    when_str = state.get("date")
    when = date.fromisoformat(when_str) if when_str else date.today()
    loc = state.get("location") or "Isle of Man"
    w = get_weather(loc, for_date=when)
    state["weather"] = w
    state["plan"] = nutrition_tips(et, when, w)
    return state


def recovery_node(state: PlannerState) -> PlannerState:
    state["event_type"] = "recovery"
    q = state.get("query") or ""
    state["plan"] = generate_recovery_plan(q)
    return state


def output_node(state: PlannerState) -> PlannerState:
    et = (state.get("event_type") or "").upper()
    header = f"=== {et} ===" if et else "=== ==="
    if state.get("event_type") == "recovery":
        plan = state.get("plan") or ""
        state["plan"] = f"{header}\n{plan}".strip()
        return state
    summary = [header]
    if "date" in state:
        summary.append(f"Date: {state['date']}")
    if state.get("opponent"):
        ha = state.get("home_away")
        suffix = f" ({ha})" if ha in ("H", "A") else ""
        summary.append(f"Opposition: {state['opponent']}{suffix}")
    if "location" in state and state["location"] and state["location"] != "Unavailable":
        summary.append(f"Location: {state['location']}")
    w = state.get("weather") or {}
    if w:
        # Suppress N/A values: if unavailable, show only the summary
        if (w.get("source") == "unavailable"):
            summary.append(f"Weather: {w.get('summary')}")
        else:
            parts = [f"Weather: {w.get('summary')}"]
            temp = w.get("temp_c")
            precip = w.get("precip", 0)
            wind = w.get("wind_kph")
            if isinstance(temp, (int, float)):
                parts.append(f"{temp}°C")
            try:
                p = float(precip)
                parts.append(f"rain {int(100*p)}%")
            except Exception:
                pass
            if isinstance(wind, (int, float)):
                parts.append(f"wind {wind} kph")
            summary.append(" | ".join(parts))
    if "plan" in state:
        summary.append(state["plan"])
    state["plan"] = "\n".join(summary)
    return state


# --- Build graph ---

def build_graph():
    graph = StateGraph(PlannerState)
    graph.add_node("classify", classify_node)
    graph.add_node("event", event_node)
    graph.add_node("advice", advice_node)
    graph.add_node("recovery", recovery_node)
    graph.add_node("output", output_node)

    graph.set_entry_point("classify")
    # Branch on injury
    def to_event_or_recovery(state: PlannerState):
        return "recovery" if state.get("event_type") == "recovery" else "event"

    graph.add_conditional_edges(
        "classify",
        to_event_or_recovery,
        {"recovery": "recovery", "event": "event"},
    )
    graph.add_edge("event", "advice")
    graph.add_edge("advice", "output")
    graph.add_edge("recovery", "output")

    # Let the platform (e.g., LangGraph Studio/Cloud) handle persistence.
    # For local CLI runs, in-memory persistence is not required.
    return graph.compile()


class AgentRunner:
    """Tiny CLI wrapper around the compiled graph."""
    def __init__(self):
        self.app = build_graph()

    def run(self, query: str) -> Dict[str, Any]:
        state: PlannerState = {"query": query}
        out = self.app.invoke(state, config={"configurable": {"thread_id": "cli"}})
        return out

graph = build_graph()
