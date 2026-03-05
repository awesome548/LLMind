# LLMind — Taxonomy Generator

Generates a structured **design-space taxonomy** from a corpus of project artefacts stored in Supabase, using an LLM with enforced structured output.

The taxonomy models the design space as a set of **Aspects** (key dimensions) and **Options** (concrete alternatives per dimension), ready for downstream embedding, clustering, and visualisation.

---

## How it works

```
Supabase artefacts
        │
        ▼
  build_artefacts()          ← filter by clustering ids or fetch all rows
        │
        ▼
  run_generate()             ← format prompt → call LLM → return Taxonomy
        │
        ▼
  _save_taxonomy()           ← write taxonomy_<mode>_<model>_<timestamp>.json
```

### Data model (`data/models.py`)

```
Taxonomy
└── aspects: list[Aspect]
        ├── name: str
        ├── desc: str
        └── options: list[Option]
                ├── name: str
                └── desc: str
```

All three types are Pydantic `BaseModel`s, so the LLM output is validated and parsed automatically.

---

## Backends

`OpenAIChat` supports two backends selected at runtime via `--base-url`:

| Backend | API call | Schema handling |
|---|---|---|
| **OpenAI** (default) | `client.beta.chat.completions.parse(response_format=Taxonomy)` | SDK applies strict transform automatically |
| **vLLM** (`--base-url` set) | `client.chat.completions.create(response_format=json_schema)` | `_make_strict_schema()` adds `additionalProperties: false` recursively |

Both paths return a validated `Taxonomy` instance. The `reasoning_effort` parameter is forwarded to OpenAI reasoning models and silently skipped for vLLM.

---

## Setup

```bash
# Install dependencies
uv sync

# Set environment variables
cp .env.example .env   # fill in OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY
```

`.env` required keys:

| Key | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (not required when using `--base-url`) |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase service role or anon key |
| `SUPABASE_TABLE` | Table name (defaults to `media_docs`) |

---

## Usage

### OpenAI (default)

```bash
uv run generate_taxonomy.py openai \
  --model gpt-4o \
  --source all_supabase \
  --mode both \
  --reasoning high
```

### OpenAI with pre-filtered artefact IDs

```bash
uv run generate_taxonomy.py openai \
  --model gpt-4o \
  -i data/selected_ids.json \
  --source selected \
  --mode both
```

### vLLM endpoint

```bash
uv run generate_taxonomy.py openai \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://localhost:8000/v1 \
  --source all_supabase
```

### Dev mode (inspect prompt without calling the LLM)

```bash
uv run generate_taxonomy.py openai --dev --source all_supabase
# writes full prompt to debug_artefacts.txt
```

---

## CLI options

| Option | Default | Description |
|---|---|---|
| `--model` | `gpt-5-nano-2025-08-07` | LLM model name |
| `--base-url` | `None` | vLLM server base URL; omit for OpenAI |
| `--source` | `selected` | `selected` (ids file) or `all_supabase` |
| `-i` | `None` | Path to clustering ids JSON (required when `--source selected`) |
| `--mode` | `both` | `details_only` or `both` (details + descriptions) |
| `--reasoning` | `medium` | Reasoning effort: `low`, `medium`, `high` (OpenAI only) |
| `--out-file` | `../results/taxonomy/schema` | Base path; timestamp and `.json` are appended |
| `--num` | `1` | Self-refine iterations (reserved, not active) |
| `--dev` | `False` | Print prompt and write `debug_artefacts.txt` |

---

## Output

Each run writes a timestamped JSON file:

```
../results/taxonomy/schema_<source>_<mode>_<model>_<YYYYMMDD_HHMM>.json
```

Example structure:

```json
{
  "aspects": [
    {
      "name": "Display technology",
      "desc": "The primary hardware that produces visible output.",
      "options": [
        { "name": "Addressable LED arrays", "desc": "High-brightness pixel-controllable LED panels." },
        { "name": "Projection mapping",     "desc": "High-lumen projectors mapped onto complex geometry." }
      ]
    }
  ]
}
```

---

## Project layout

```
llmind-python/
├── generate_taxonomy.py   # CLI entrypoint and LLM orchestration
├── farthest_clustering.py # Select diverse representative artefacts
├── scrape_projects.py     # Scrape and ingest project artefacts
├── database_pipeline.py   # Embed and upsert artefacts into Supabase
├── data/
│   ├── models.py          # Pydantic schema: Taxonomy, Aspect, Option
│   └── prompts.py         # System prompt and generation templates
└── utils/
    ├── supabase.py        # Supabase fetch helpers and artefact builder
    └── json.py            # JSON load/save utilities
```

---

## Self-refine loop

The iterative self-review loop (`IDEA_REFLECTION_PROMPT`) is implemented but commented out in `run_generate()`. To enable it, uncomment the loop block and pass `--num <rounds>`. Each round sends the current taxonomy back to the model as context and asks it to consolidate and refine the aspects and options.
