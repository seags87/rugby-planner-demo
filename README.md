# Rugby Match Planner Agent (LangGraph)

A small, stateful agent that helps plan rugby matches, training, and recovery using live data.

- Orchestration: LangGraph nodes — classify → info → tools → output (with a recovery branch)
- Fixtures: Pitchero club calendar (live-only)
- Location + Weather: Google Places + Google Weather (forecasts <= 10 days)
- Tips: OpenAI for nutrition/gear and recovery plans

## Project structure

- `src/rugby_planner/graph.py` — agent state and graph
- `src/rugby_planner/tools/` — calendar, weather, nutrition, recovery, parse
- `src/rugby_planner/main.py` — CLI runner
- `src/rugby_planner/eval/` — simple evaluation scripts
- `requirements.txt` — dependencies

## Setup

1) Python 3.10+
2) (Recommended) Create and activate a virtualenv
3) Install dependencies:

```bash
pip install -r requirements.txt
```

4) Configure the environment in `.env`:

- GOOGLE_MAPS_API_KEY — for Places + Weather
- PITCHERO_CLUB_ID — club ID for fixtures (e.g., 7732)
- PITCHERO_TEAM_NAME — default club/team name as shown on Pitchero (e.g., Ramsey (IoM))
- OPENAI_API_KEY — for nutrition and recovery outputs
- OPENAI_MODEL — optional (default: gpt-4o-mini)
- (Optional) LangSmith: LANGSMITH_API_KEY, LANGSMITH_PROJECT, LANGSMITH_ENDPOINT

Example:

```
GOOGLE_MAPS_API_KEY=your_key
PITCHERO_CLUB_ID=7732
PITCHERO_TEAM_NAME="Ramsey (IoM)"
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini
LANGSMITH_API_KEY=your_key
LANGSMITH_PROJECT=rugby-planner-demo
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_DATASET=rugby-planner-dataset
```

## How it works

- Classifies the query as match, training, general, or recovery
- For matches:
  - Parses natural dates (e.g., “this Saturday”, “in 3 weeks”) and uses an LLM to extract date/opponent/home/away
  - Resolves a specific fixture (by date), next vs opponent, or next matching H/A by scanning months forward
  - Enforces that “home” means the club is the homeSide
  - Infers a friendly location from the home team via Google Places (cached at `src/.cache/team_locations.json`)
- Weather via Google Weather days:lookup; only shows forecasts within 10 days of the event
- Nutrition tips (short bullets) and recovery plans are generated via OpenAI
- Output header format: `=== EVENT TYPE ===`, then Date → Opposition (H/A) → Location → Weather → Tips

The internal parse tool returns both a legacy `ha` (h/a) and a readable `home_away` (h/a) for compatibility; the graph normalizes to `H`/`A` for display.

Training occurs every Tuesday and Thursday; the next such session is returned.

## Run (CLI)

```bash
python -m rugby_planner.main "When’s our next home match?"
python -m rugby_planner.main "Who are we playing on 18th October?"
python -m rugby_planner.main "When do we train next?"
python -m rugby_planner.main "Tweaked hamstring — plan for 10 days?"
```

## Evaluation (optional)

Local:

```bash
python -m rugby_planner.eval.run_eval
```

Generate prompts from live fixtures:

```bash
python -m rugby_planner.eval.generate_dataset --year 2025 --month 10 --out src/rugby_planner/eval/generated_dataset.jsonl
```

If LangSmith is configured, the eval script will also start a run in your project.

## Notes

- Live-only: there are no offline fallbacks for weather, nutrition, or recovery; the agent returns clear “unavailable” messages on failure
- Forecasts are limited to 10 days ahead per Google Weather constraints
- Locations are inferred from the fixture’s home side and cached to `src/.cache/team_locations.json`

## LangGraph Studio

- The Studio config `langgraph.json` loads the graph from `./src/rugby_planner/graph.py:graph` and reads environment from `./.env`.
- Dependencies are installed via your environment/requirements; there’s no runtime pip-install in code.