# LLMind
LLMind is an LLM-assisted system for exploring the design space. 

## Technical snapshot
- **Frontend:** React + TypeScript single-page app under `designspace-viz`, using Zustand for state and jsMind for graph rendering.
- **AI workflow:** `useOpenAI.ts` formats the active mind-map branch, injects prompt templates from `/prompts`, and calls OpenAI chat + embedding models to propose new nodes; responses are parsed back into the jsMind node array.
- **Project retrieval:** embeddings are sent to a Supabase to fetch related project metadata (name, description, assets) that get surfaced beside the map.
- **Data plumbing:** Python utilities (`transcribe.py`, `generate_taxonomy.py`, `database_pipeline.py`, `scrape_projects.py`) handle speech-to-text, clustering, and taxonomy serialization before the UI consumes the resulting schema JSON.

## llmind-python scripts

### `llmind-python` layout
- `scrape_projects.py`: Crawl Media Architecture Biennale project pages and save raw project JSON.
- `database_pipeline.py`: Analyze raw JSON, clean/filter records, and upsert vector embeddings into Supabase.
- `farthest_clustering.py`: Pull embeddings from Supabase, reduce to 2D, cluster, and produce either a scatter plot or a farthest-diverse ID subset.
- `generate_taxonomy.py`: Build design-space ideas from selected artefacts and write taxonomy JSON using Gemini/OpenAI prompts.
- `transcribe.py`: Transcribe audio with OpenAI (Whisper/diarization), or convert an existing transcription JSON to text.

### How to run

From repo root:

```bash
cd llmind-python
```

Recommended install/run flow:

```bash
uv sync
source .venv/bin/activate   # or use `uv run` in place of `python`
```

Core required environment variables:

- `SUPABASE_URL`, `SUPABASE_KEY` (pipeline scripts that read embeddings/data)
- `SUPABASE_TABLE` (optional, defaults to `media_docs`)
- `OPENAI_API_KEY` (for `database_pipeline.py`, `generate_taxonomy.py`, `transcribe.py`)
- `GEMINI_API_KEY` (only for `generate_taxonomy.py` `gemini` command)
- `BASE_URL`, `DEFAULT_LISTING_URL`, `DATA_DIR` (for `scrape_projects.py` output/input paths)
- `OPENAI_MODEL`, `DATA_DIR`, `ANALYSIS_DIR`, `PLOTS_DIR`, `CHROMA_DB_PATH` (used by `database_pipeline.py` defaults)

Scrape raw projects:

```bash
python scrape_projects.py scrape --limit 20 --out media_architecture.json
```

Analyze / clean / embed the dataset:

```bash
python database_pipeline.py analyze --file data/media_architecture.json
python database_pipeline.py clean --input data/media_architecture.json --output data/cleaned_media_architecture.json
python database_pipeline.py embed --file data/cleaned_media_architecture.json
```

Clustering and subset selection:

```bash
python farthest_clustering.py cluster --table media_docs --clusters 8 --plot
python farthest_clustering.py farthest --table media_docs --k 20 --json data/selected_projects.json
```

Generate taxonomy:

```bash
python generate_taxonomy.py openai --prompt-file data/prompts/system_prompt.json --ids-file data/selected_projects.json
python generate_taxonomy.py gemini --prompt-file data/prompts/system_prompt.json --source all_supabase
```

(Optional: for user interviews)
Transcription:

```bash
python transcribe.py path/to/audio.m4a --output transcript.txt
python transcribe.py path/to/audio.m4a --diarize --speaker-names "Alice" "Bob" --speaker-references alice.wav bob.wav
python transcribe.py transcript.json --json-to-text --diarize --output transcript.txt
```

You can also execute the same commands with `uv run` instead of `python` when using `uv`.
