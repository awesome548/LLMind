# LLMind
LLMind is an LLM-assisted system for exploring the design space. 

## Technical snapshot
- **Frontend:** React + TypeScript single-page app under `designspace-viz`, using Zustand for state and jsMind for graph rendering.
- **AI workflow:** `useOpenAI.ts` formats the active mind-map branch, injects prompt templates from `/prompts`, and calls OpenAI chat + embedding models to propose new nodes; responses are parsed back into the jsMind node array.
- **Project retrieval:** embeddings are sent to a Supabase to fetch related project metadata (name, description, assets) that get surfaced beside the map.
- **Data plumbing:** Python utilities (`transcribe.py`, `generate_taxonomy.py`, `_database.py`, `_scrape.py`) handle speech-to-text, clustering, and taxonomy serialization before the UI consumes the resulting schema JSON.
