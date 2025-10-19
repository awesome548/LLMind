# TaxonAI Designspace Visualizer

A React + TypeScript application that turns a taxonomy into an interactive design space explorer. The interface combines a jsMind-powered mind map, a tabular schema view, and a related projects panel that can be enriched with OpenAI suggestions and Supabase vector search.

## To Do
- [ ] Design Fix with project context banner
- [ ] Add addtional round of making the output concise?
- [ ] Add description to each option to improve the context? 

## Features
- Mind map view with contextual styling, node selection, and AI-assisted expansion for exploring adjacent concepts.
- Schema table view that mirrors the taxonomy JSON for quick scanning of aspects, descriptions, and options.
- Related projects sidebar that surfaces curated or Supabase-backed media architecture precedents based on the active node lineage.
- Prompt-driven OpenAI integration that proposes new nodes and allows selective insertion into the live mind map.

## Getting Started
1. Install dependencies: `npm install`
2. Launch the dev server: `npm run dev`
3. Open the app at the URL printed in the console (default `http://localhost:5173`)
4. Build for production when ready: `npm run build`

## Available Scripts
- `npm run dev` — start Vite in development mode with hot reloading.
- `npm run build` — type-check and build the production bundle.
- `npm run preview` — preview the production build locally.
- `npm run lint` — run ESLint across the project.

## Configuration
- Taxonomy data lives in `public/taxonomy/schema.json` and is fetched at runtime.
- To enable AI-assisted exploration, set `VITE_OPENAI_API_KEY` (and optionally override `VITE_OPENAI_EMBED_MODEL`).
- To turn on Supabase similarity search for projects, provide `VITE_SUPABASE_URL`, `VITE_SUPABASE_KEY`, and `VITE_SUPABASE_MATCH_FN`.
- Prompt templates reside in `public/prompts/` and can be edited to adjust tone or output expectations.