from __future__ import annotations
import rugby_planner  # noqa: F401

import sys
from rich.console import Console  # noqa: F401 (kept for future rich usage)

from .graph import AgentRunner


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("Usage: python -m rugby_planner.main \"your question\"")
        return 2
    query = " ".join(argv)
    agent = AgentRunner()
    result = agent.run(query)
    print("\n---")
    print(result.get("plan", ""))
    print("---\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
