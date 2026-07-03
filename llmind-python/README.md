# llmind-python — LLMind backend

The FastAPI backend + data pipeline of **LLMind**, a research prototype for
LLM-assisted design-space exploration. Python 3.13+, managed with **uv**.

> This file is only a launcher. The real documentation:
>
> | Doc | Contents |
> |---|---|
> | [`BACKEND.md`](BACKEND.md) | **The SSOT**: every API endpoint, env vars, architecture, projection/annotation subsystems, error patterns, and the full data-pipeline & CLI reference |
> | [`CLAUDE.md`](CLAUDE.md) | Thin hub: environment, pipeline flow diagrams, the 768-d live-vs-default warning, structured-output rules |
> | [`../PROJECT-REPORT.md`](../PROJECT-REPORT.md) | What the system is, feature-by-feature, with the research argument |

## Run

```bash
uv sync
uv run fastapi dev backend/main.py     # → http://localhost:8000
```

The live stack is **fully local**: `VECTOR_STORE=local`, the 768-d
`nomic-embed-text-v1.5` index (`data/local_index.npz`, 209 MAB projects), and
LM Studio at `localhost:1234` serving both models (`.env` overrides the stale
`config.py` defaults — see BACKEND.md's env table).

## Test

```bash
uv run python test_projection.py       # offline suite (no servers needed)
uv run python test_projection.py --live  # + embedding/LLM round-trips
```
