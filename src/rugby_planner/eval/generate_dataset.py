from __future__ import annotations

import json
import argparse
from datetime import date
from typing import Any, Dict, List

from rugby_planner.tools.calendar import fetch_fixtures


def build_prompts_from_month(year: int, month: int, max_items: int = 20) -> List[Dict[str, Any]]:
    data = fetch_fixtures(year, month)
    rows: List[Dict[str, Any]] = []
    for day in (data.get("data", {}) or {}).get("days", []) or []:
        d = day.get("date")
        for fx in day.get("fixtures", []) or []:
            opp = fx.get("opponent") or "opponent"
            ha = (fx.get("ha") or "").lower()
            if ha == "h":
                q = f"When is our next home match vs {opp}?"
            elif ha == "a":
                q = f"When is our next away match vs {opp}?"
            else:
                q = f"When do we next play {opp}?"
            item: Dict[str, Any] = {"query": q}
            inc = ["=== MATCH ===", "Date:", "Location:"]
            sym = (ha.upper() if ha in ("h","a") else "[HA]")
            regex = [rf"Opposition: .* \\({sym}\\)"]
            item["outputs"] = {"expected_contains": inc, "expected_regex": regex}
            rows.append(item)
            if len(rows) >= max_items:
                return rows
    if not rows:
        # Fallback generic prompts with expectations
        rows = [
            {"query": "When's our next home match?", "outputs": {"expected_contains": ["=== MATCH ===", "Location:"]}},
            {"query": "Who are we playing next away?", "outputs": {"expected_contains": ["Opposition:"]}},
            {"query": "What should I bring for training this week?", "outputs": {"expected_contains": ["=== TRAINING ==="]}},
            {"query": "Pulled my hamstring, can you give me a 10-day recovery plan?", "outputs": {"expected_contains": ["=== RECOVERY ===", "Disclaimer"]}},
        ]
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    today = date.today()
    p.add_argument("--year", type=int, default=today.year)
    p.add_argument("--month", type=int, default=today.month)
    p.add_argument("--out", default="src/rugby_planner/eval/generated_dataset.jsonl")
    p.add_argument("--max-items", type=int, default=20)
    args = p.parse_args(argv)

    rows = build_prompts_from_month(args.year, args.month, max_items=args.max_items)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} examples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
