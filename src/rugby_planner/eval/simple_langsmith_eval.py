from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from rugby_planner.graph import AgentRunner

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

import re


def _extract_field_lines(text: str) -> Dict[str, str | int]:
    header = ""
    opposition = ""
    location = ""
    weather = ""
    bullets = 0
    for line in text.splitlines():
        if not header:
            m = re.match(r"^===\s*([A-Z]+)\s*===\s*$", line.strip())
            if m:
                header = m.group(1).strip().upper()
        if not opposition and line.startswith("Opposition:"):
            opposition = line[len("Opposition:"):].strip()
        if not location and line.startswith("Location:"):
            location = line[len("Location:"):].strip()
        if not weather and line.startswith("Weather:"):
            weather = line.strip()
        if line.strip().startswith("- "):
            bullets += 1
    return {"header": header, "opposition": opposition, "location": location, "weather": weather, "bullets": bullets}


def _reason_and_score(output_text: str, reference_text: str) -> Dict[str, object]:
    o = _extract_field_lines(output_text or "")
    r = _extract_field_lines(reference_text or "")
    issues: list[str] = []

    # 1) Event type
    if r.get("header") and o.get("header") != r.get("header"):
        issues.append(f"Event type mismatch (got '{o.get('header')}', expected '{r.get('header')}')")

    # 2) Opposition and location
    ropp = str(r.get("opposition") or "").lower()
    oopp = str(o.get("opposition") or "").lower()
    if ropp and ropp != oopp:
        issues.append("Opposition (or H/A) mismatch")
    rloc = str(r.get("location") or "").lower()
    oloc = str(o.get("location") or "").lower()
    if rloc and rloc != oloc:
        issues.append("Location mismatch")

    # 3) Weather
    rweath = str(r.get("weather") or "")
    oweath = str(o.get("weather") or "")
    r_unavail = ("no forecast available" in rweath.lower()) or ("unavailable" in rweath.lower())
    o_unavail = ("no forecast available" in oweath.lower()) or ("unavailable" in oweath.lower())
    if rweath:
        if r_unavail and not o_unavail:
            issues.append("Weather should indicate unavailability to match reference")
        elif not r_unavail:
            # Reference has a weather line; ensure output has one too
            if not oweath:
                issues.append("Missing weather line")

    # 4) Bullet plan format
    if int(o.get("bullets") or 0) < 2:
        issues.append("Plan not in concise bullet format")

    score = 0 if issues else 1
    reason = "; ".join(issues) if issues else "All checks passed"
    return {"correctness": score, "reason": reason}


def perform_eval(run, example) -> Dict[str, object]:  # type: ignore[no-untyped-def]
    """Simple correctness evaluator.

    run: may be a dict with 'outputs' or a dict with 'plan', or a Run-like object.
    example: a dict with 'inputs' and 'outputs' (must contain 'referenceOutput').

    Returns: { "correctness": 1 or 0 }
    """
    plan = ""
    try:
        if isinstance(run, dict):
            outs = run.get("outputs") if isinstance(run.get("outputs"), dict) else None
            if isinstance(outs, dict):
                plan = str(outs.get("plan", ""))
            else:
                plan = str(run.get("plan", ""))
        else:
            outs = getattr(run, "outputs", {})
            if isinstance(outs, dict):
                plan = str(outs.get("plan", ""))
    except Exception:
        plan = ""

    ref = _extract_reference_output(example)
    return _reason_and_score(plan, ref)


def _extract_reference_output(example) -> str:
    # Handle dict-shaped examples (local)
    if isinstance(example, dict):
        ex_out = example.get("outputs") or {}
        if isinstance(ex_out, dict):
            val = ex_out.get("referenceOutput") or ex_out.get("plan")
            return str(val or "")
        return ""
    # Handle LangSmith Example objects
    try:
        ex_out = getattr(example, "outputs", None)
        if isinstance(ex_out, dict):
            val = ex_out.get("referenceOutput") or ex_out.get("plan")
            return str(val or "")
    except Exception:
        pass
    return ""


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a simple LangSmith eval of the agent using a JSONL dataset.")
    parser.add_argument(
        "--dataset-file",
        default="src/rugby_planner/eval/rugby-planner-dataset.jsonl",
        help="Path to JSONL with lines like {inputs:{query:...}, outputs:{referenceOutput:...}} or {input, referenceOutput}",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "langsmith"],
        default="local",
        help="Run locally and print results, or submit to LangSmith",
    )
    parser.add_argument(
        "--dataset-name",
        default=os.getenv("LANGSMITH_DATASET", "rugby-planner-dataset"),
        help="LangSmith dataset name",
    )
    parser.add_argument(
        "--project",
        default=os.getenv("LANGSMITH_PROJECT", "rugby-planner-demo"),
        help="LangSmith project name",
    )
    args = parser.parse_args(argv)

    # LangSmith only required for upload mode
    Client = None
    if args.mode == "langsmith":
        try:
            from langsmith import Client as _Client  # type: ignore
            Client = _Client
        except Exception:
            print("langsmith package is required for --mode langsmith. Install dependencies and try again.")
            return 2
        api_key = os.getenv("LANGSMITH_API_KEY")
        if not api_key:
            print("Missing LANGSMITH_API_KEY. Set it in your environment or .env.")
            return 2

    # Load JSONL examples
    examples: List[Dict[str, object]] = []
    with open(args.dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            # Normalize to {inputs:{query:...}, outputs:{referenceOutput:...}}
            if "inputs" in rec and "outputs" in rec:
                examples.append(rec)
            else:
                inp = rec.get("input") if isinstance(rec, dict) else None
                ref = rec.get("referenceOutput") if isinstance(rec, dict) else None
                if isinstance(inp, str) and isinstance(ref, str):
                    examples.append({"inputs": {"query": inp}, "outputs": {"referenceOutput": ref}})

    if not examples:
        print("No examples found in dataset file.")
        return 1

    if args.mode == "langsmith":
        client = Client()
        # Create or read dataset in LangSmith
        try:
            ds = client.read_dataset(dataset_name=args.dataset_name)
            ds_id = ds.id
        except Exception:
            ds = client.create_dataset(dataset_name=args.dataset_name, description="Rugby planner eval dataset")
            ds_id = ds.id

        # Upsert examples into dataset
        for ex in examples:
            client.create_example(inputs=ex["inputs"], outputs=ex["outputs"], dataset_id=ds_id)  # type: ignore[index]

    # Define the app function once
    def app_fn(inputs: Dict[str, object]) -> Dict[str, object]:
        agent = AgentRunner()
        res = agent.run(str(inputs.get("query", "")))
        return {"plan": res.get("plan", "")}

    if args.mode == "local":
        # Run locally and print per-example correctness, input, and plan
        passed = 0
        for idx, ex in enumerate(examples, start=1):
            out = app_fn(ex["inputs"])  # type: ignore[index]
            feedback = perform_eval({"outputs": out}, ex)  # type: ignore[arg-type]
            score = int(feedback.get("correctness", 0))
            status = "PASS" if score == 1 else "FAIL"
            inp = ex.get("inputs", {})
            query = str(inp.get("query", "")) if isinstance(inp, dict) else ""
            reason = str(feedback.get("reason", ""))
            print(f"[{idx:02d}] {status} | correctness={score}")
            if reason:
                print(f"Reason: {reason}")
            if query:
                print(f"Input: {query}")
            print(out.get("plan", ""))
            print("-" * 80)
            if score == 1:
                passed += 1
        total = len(examples)
        print(f"Summary: {passed}/{total} passed")
        return 0

    else:
        # Use LangSmith evaluate API (prefer run_evaluators, fallback to evaluation)
        try:
            from langsmith.run_evaluators import evaluate  # type: ignore[attr-defined]
        except Exception:
            from langsmith.evaluation import evaluate  # type: ignore

        # Ensure dataset exists and is populated (already upserted above)
        # Now trigger the evaluation run
        # Wrap our evaluator to return the structure expected by LangSmith
        def _ls_eval(run, example):  # type: ignore[no-untyped-def]
            res = perform_eval(run, example)
            score = int(res.get("correctness", 0)) if isinstance(res, dict) else 0
            comment = str(res.get("reason", "")) if isinstance(res, dict) else ""
            return {"score": score, "key": "correctness", "comment": comment}

        _ = evaluate(
            app_fn,
            data=args.dataset_name,
            evaluators=[_ls_eval],
            experiment_prefix="simple-correctness",
            description="Simple correctness eval with rule-based rubric against referenceOutput",
        )

        # Best-effort: show the LangSmith app URL if configured
        url = os.getenv("LANGSMITH_ENDPOINT", "https://smith.langchain.com")
        print(f"Submitted eval to LangSmith project '{args.project}'.")
        print(f"Open LangSmith to view results: {url}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
