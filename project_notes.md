# Issues & Future TODOs

- Add evaluators for individual LLM queries, not just who agent input/output.
- Unknown input handling:
    - need to return general output if asking about a day with no match (currently returns closest match).
    - need to return an informative response when asking about a team there's no fixtures against.
- Improve search algorithm on Pitchero's API to reduce calls.
- Find a Pitchero endpoint which returns the club ID so others can easily look up their own clubs.
- Find a weather API which returns a longer range forecast than 10 days.


# Friction Log

- LangGraph Studio: Tool calls were showing up as Chain events. Used decoraters as a quick fix but I probably should've used `.bind_tools()`.
- LangSmith: No way to add metadata across all dataset items in bulk.
- LangSmith: Datasets & Experiments count shows 2 when only 1 exists after delete.
