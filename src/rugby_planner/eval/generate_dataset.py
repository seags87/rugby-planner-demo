from __future__ import annotations

# Load .env automatically for any future API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import argparse
import calendar as cal
import json
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from ..tools.calendar import fetch_fixtures, infer_location_from_fixture


@dataclass
class FixtureLite:
    date: str
    opponent: str
    ha: str
    location: str


def get_month_fixtures(year: int, month: int) -> List[FixtureLite]:
    payload = fetch_fixtures(year, month, use_cache_first=True)
    days = payload.get("data", {}).get("days", [])
    out: List[FixtureLite] = []
    for d in days:
        for fx in d.get("fixtures", []) or []:
            loc = infer_location_from_fixture(fx)
            out.append(
                FixtureLite(
                    date=(fx.get("dateTime", "")[:10] or d.get("date") or ""),
                    opponent=fx.get("opponent", ""),
                    ha=fx.get("ha", ""),
                    location=loc,
                )
            )
    return out


def synthesize_prompts(fixtures: List[FixtureLite]) -> List[Dict[str, Any]]:
    prompts: List[Dict[str, Any]] = []
    for fx in fixtures:
        # Home/away hints in text for the agent
        if fx.ha == "h":
            p = f"We’re at home on {fx.date} vs {fx.opponent}. What kit and nutrition do you recommend?"
            prompts.append({"input": p, "expect_location_contains": "Isle of Man"})
        else:
            p = f"We’re away at {fx.opponent} on {fx.date}. What’s the forecast and what gear should I pack?"
            # Expectation: location contains opponent city string
            city_hint = fx.location.split(",")[0]
            prompts.append({"input": p, "expect_location_contains": city_hint})

        # Generic match prompt specifying town
        town = fx.location.split(",")[0]
        prompts.append({"input": f"Match in {town} this weekend — weather and kit?", "expect_location_contains": town})

    # Add training & recovery examples
    today = date.today()
    prompts += [
        {"input": "Tuesday training after work — nutrition and kit?", "expect_event_type": "training"},
        {"input": "Tweaked hamstring — recovery plan for 10 days please.", "expect_event_type": "recovery"},
    ]
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=date.today().year)
    ap.add_argument("--month", type=int, default=date.today().month)
    ap.add_argument("--out", type=str, default="generated_dataset.jsonl")
    args = ap.parse_args()

    fixes = get_month_fixtures(args.year, args.month)
    rows = synthesize_prompts(fixes)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
