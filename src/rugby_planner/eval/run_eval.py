from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
import re

from rugby_planner.graph import build_graph


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _as_inputs(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize to {"query": ...} inputs for the agent."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        if isinstance(r, dict):
            if "query" in r:
                out.append({"query": r["query"], **{k: v for k, v in r.items() if k != "query"}})
            elif "input" in r:
                # Some datasets may use {input: "..."}
                v = r["input"]
                if isinstance(v, dict) and "query" in v:
                    out.append(v)
                else:
                    out.append({"query": str(v)})
            else:
                # Best-effort: stringify whole row
                out.append({"query": json.dumps(r)})
        else:
            out.append({"query": str(r)})
    return out


def run_local(dataset_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    app = build_graph()
    rows = _load_jsonl(dataset_path)
    inputs = _as_inputs(rows)
    if limit is not None:
        inputs = inputs[:limit]
    outputs: List[Dict[str, Any]] = []
    for i, inp in enumerate(inputs, 1):
        state = {"query": inp.get("query", "")}
        out = app.invoke(state, config={"configurable": {"thread_id": f"eval-{i}"}})
        outputs.append(out)
    return outputs


def ensure_langsmith_dataset(dataset_name: str, rows: List[Dict[str, Any]]):
    """Create/update a dataset in LangSmith with input rows.

    Requires LANGSMITH_API_KEY and LANGSMITH_ENDPOINT (optional) to be set.
    """
    import importlib
    Client = getattr(importlib.import_module("langsmith"), "Client")

    client = Client()
    # Try to get; if not exists, create
    try:
        ds = client.read_dataset(dataset_name=dataset_name)
    except Exception:
        ds = client.create_dataset(dataset_name)

    # Upsert examples
    for r in rows:
        input_payload = {"query": r.get("query", "")}
        output_payload = r.get("outputs") if isinstance(r, dict) else None
        try:
            client.create_example(inputs=input_payload, outputs=output_payload, dataset_id=ds.id)
        except Exception:
            # If duplicate or other error, ignore for idempotency
            pass
    return ds


def run_langsmith_eval(dataset_name: str, *, project: Optional[str] = None) -> str:
    """Trigger an evaluation run in LangSmith using the dataset.

    Uses LangSmith's run_on_dataset to create a UI-visible run.
    Returns the run URL if available.
    """
    import importlib
    Client = getattr(importlib.import_module("langsmith"), "Client")

    class _SimpleRunnable:
        def __init__(self, fn):
            self._fn = fn

        def invoke(self, inputs: Dict[str, Any]):
            return self._fn(inputs)

    client = Client()
    proj = project or os.getenv("LANGSMITH_PROJECT", "rugby-planner-demo")

    def factory():
        app = build_graph()

        def _call(inputs: Dict[str, Any]):
            q = (inputs or {}).get("query", "")
            state = {"query": q}
            out = app.invoke(state, config={"configurable": {"thread_id": "ls-dataset"}})
            return {"plan": out.get("plan", ""), **out}

        return _SimpleRunnable(_call)

    # --- Evaluators ---
    def _extract_text_from_run(run_obj: Any) -> str:
        try:
            outputs = getattr(run_obj, "outputs", None)
        except Exception:
            outputs = None
        text = ""
        if isinstance(outputs, dict):
            text = (
                outputs.get("plan")
                or outputs.get("output")
                or outputs.get("text")
                or json.dumps(outputs)
            )
        elif outputs is not None:
            text = str(outputs)
        return text or ""

    def includes_evaluator(run_obj, example_obj):
        """Checks that all expected substrings are present in plan output.

        Place expectations in example.outputs.expected_contains (list of strings).
        """
        try:
            expected = []
            if getattr(example_obj, "outputs", None):
                exp = example_obj.outputs or {}
                expected = list(exp.get("expected_contains") or exp.get("includes") or [])
            text = _extract_text_from_run(run_obj)
            missing = [s for s in expected if s not in text]
            total = len(expected) or 1
            score = (total - len(missing)) / total
            return {"key": "includes", "score": score, "commentary": f"missing: {missing}"}
        except Exception as e:
            return {"key": "includes", "score": 0.0, "commentary": f"evaluator error: {e}"}

    def regex_evaluator(run_obj, example_obj):
        """Checks regex patterns in example.outputs.expected_regex (list of patterns)."""
        try:
            patterns = []
            if getattr(example_obj, "outputs", None):
                exp = example_obj.outputs or {}
                patterns = list(exp.get("expected_regex") or exp.get("regex") or [])
            text = _extract_text_from_run(run_obj)
            missing = []
            for pat in patterns:
                try:
                    if not re.search(pat, text, flags=re.MULTILINE):
                        missing.append(pat)
                except re.error:
                    # Treat invalid regex as a miss
                    missing.append(pat)
            total = len(patterns) or 1
            score = (total - len(missing)) / total
            return {"key": "regex", "score": score, "commentary": f"missing: {missing}"}
        except Exception as e:
            return {"key": "regex", "score": 0.0, "commentary": f"evaluator error: {e}"}

    run = client.run_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=factory,
        project_name=proj,
        concurrency=4,
        evaluators=[includes_evaluator, regex_evaluator],
        description="Rugby planner dataset run",
    )
    return getattr(run, "url", "") or ""


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run local eval or push to LangSmith")
    parser.add_argument("--dataset", default=os.getenv("LANGSMITH_DATASET", "src/rugby_planner/eval/small.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--langsmith", action="store_true", help="Use LangSmith SDK: create dataset and log runs")
    parser.add_argument("--dataset-name", default=os.getenv("LANGSMITH_DATASET", "rugby-planner-dataset"))
    parser.add_argument("--project", default=os.getenv("LANGSMITH_PROJECT"))
    args = parser.parse_args(argv)

    if args.langsmith:
        rows = _as_inputs(_load_jsonl(args.dataset))
        ensure_langsmith_dataset(args.dataset_name, rows)
        url = run_langsmith_eval(args.dataset_name, project=args.project)
        print(f"LangSmith run started. Dataset={args.dataset_name}")
        if url:
            print(f"Run URL: {url}")
        return 0

    # Local mode
    outs = run_local(args.dataset, limit=args.limit)
    for i, o in enumerate(outs, 1):
        print("\n---")
        print(o.get("plan", ""))
        print("---\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
