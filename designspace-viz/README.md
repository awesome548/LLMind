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

## Requirements
- Node.js 18+ and npm 10+ (Vite and the OpenAI SDK both expect a modern runtime).
- An OpenAI API key for AI-assisted features.
- Optional, but recommended: a Supabase project containing the `media_docs` table and the `match_media_docs` RPC function for similarity search.

## Getting Started
1. Clone the repository and move into the app folder:
   ```bash
   git clone <repo-url>
   cd designspace-viz
   ```
2. Install dependencies: `npm install`
3. Create a `.env.local` (or `.env`) file in `designspace-viz/` with the environment variables your workflow requires (the default taxonomy referenced by the app lives in `public/taxonomy/schema.json`):
   ```bash
   VITE_OPENAI_API_KEY=<sk-...>                  # Required for all OpenAI calls
   VITE_OPENAI_TAXONOMY_MODEL=gpt-4o-mini        # Optional override for taxonomy generation
   VITE_TAXONOMY_REFLECTIONS=1                   # Optional iterations for refinement prompts
   VITE_OPENAI_EMBED_MODEL=text-embedding-3-small

   VITE_SUPABASE_URL=<https://...supabase.co>    # Required for Supabase project search
   VITE_SUPABASE_KEY=<public-anon-key>
   VITE_SUPABASE_TABLE=media_docs                # Defaults to media_docs when omitted
   VITE_SUPABASE_MATCH_FN=match_media_docs       # RPC used by related projects panel
   ```
4. Launch the dev server: `npm run dev`
5. Open the app at the URL printed in the console (default `http://localhost:5173`)
6. Build for production when ready: `npm run build`

## Available Scripts
- `npm run dev` — start Vite in development mode with hot reloading.
- `npm run build` — type-check and build the production bundle.
- `npm run preview` — preview the production build locally.
- `npm run lint` — run ESLint across the project.

## Configuration
- Taxonomy data lives in `public/taxonomy/schema.json` and is fetched at runtime.
- To enable AI-assisted exploration, set `VITE_OPENAI_API_KEY` (and optionally override `VITE_OPENAI_EMBED_MODEL`).
- To turn on Supabase similarity search for projects, provide `VITE_SUPABASE_URL`, `VITE_SUPABASE_KEY`, `VITE_SUPABASE_TABLE`, and `VITE_SUPABASE_MATCH_FN`.
- Prompt templates reside in `public/prompts/` and can be edited to adjust tone or output expectations.
