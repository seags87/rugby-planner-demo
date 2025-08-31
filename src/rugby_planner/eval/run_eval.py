from __future__ import annotations

# Load .env automatically
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import json
import os
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List

from ..graph import AgentRunner

# Optional: integrate with LangSmith if configured
try:
    from langsmith import client as ls_client
    from langsmith.evaluation import evaluate
except Exception:  # pragma: no cover - still allow offline
    ls_client = None
    evaluate = None


@dataclass
class EvalRecord:
    input: str
    output: Dict[str, Any]
    scores: Dict[str, float]


def expect_scores(output: Dict[str, Any], expect: Dict[str, Any]) -> Dict[str, float]:
    text = (output.get("plan") or "").lower()
    event_type = output.get("event_type")
    location = output.get("location", "")
    s: Dict[str, float] = {}

    if "expect_event_type" in expect:
        s["event_type_match"] = 1.0 if event_type == expect["expect_event_type"] else 0.0
    if "expect_location_contains" in expect:
        s["location_match"] = 1.0 if expect["expect_location_contains"].lower() in location.lower() else 0.0

    # Generic heuristics
    relevance = 1.0 if any(k in text for k in ["isle of man", "douglas", "ramsey", "ormskirk", "clitheroe", "colne"]) else 0.4
    helpful = 1.0 if len(text) > 50 else 0.4
    accurate = 0.85 if (output.get("weather") or {}).get("source") == "google" else 0.7
    s.update({"relevance": relevance, "helpfulness": helpful, "accuracy": accurate})
    return s


def run_local_eval(dataset_path: str) -> List[EvalRecord]:
    agent = AgentRunner()
    recs: List[EvalRecord] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            inp = row["input"]
            out = agent.run(inp)
            s = expect_scores(out, row)
            recs.append(EvalRecord(input=inp, output=out, scores=s))
    return recs


def maybe_run_langsmith(dataset_path: str) -> None:
    if ls_client is None or evaluate is None or not os.getenv("LANGSMITH_API_KEY"):
        print("LangSmith not configured; ran local-only eval.")
        return

    client = ls_client.Client()
    dataset_name = os.getenv("LANGSMITH_DATASET", "rugby-planner-dataset")
    # Create dataset if missing
    if not client.has_dataset(dataset_name):
        rows = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                rows.append({"inputs": {"input": row["input"]}, "outputs": {}})
        client.create_dataset(dataset_name)
        client.create_examples(dataset_name=dataset_name, examples=rows)

    # Define a simple runnable wrapper for the agent
    def agent_runnable(example: Dict[str, Any]):
        agent = AgentRunner()
        inp = example["input"] if isinstance(example, dict) else example["inputs"]["input"]
        out = agent.run(inp)
        return {"output": out.get("plan", "")}

    project = os.getenv("LANGSMITH_PROJECT", "rugby-planner-demo")
    res = evaluate(
        agent_runnable,
        data=dataset_name,
    evaluators=["qa"],
        experiment_prefix="rugby_planner_eval",
        metadata={"agent": "rugby-planner"},
        project_name=project,
    )
    print("LangSmith eval started:", res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.path.join(os.path.dirname(__file__), "dataset.jsonl"))
    args = parser.parse_args()

    recs = run_local_eval(args.dataset)
    print("Local eval results (subset):")
    for r in recs[:10]:
        print("-", r.input[:70], {k: round(v, 2) for k, v in r.scores.items()})
    maybe_run_langsmith(args.dataset)
