# LLMind Designspace Explorer

A React + TypeScript application that turns a taxonomy into an interactive design space explorer. The interface combines a jsMind-powered mind map, a tabular schema view, and a related projects panel that can be enriched with OpenAI suggestions and Supabase vector search.

[Watch Demo Video](https://youtu.be/svZto4WgieU)

## Requirements
- Node.js 18+ and npm 10+ (Vite and the OpenAI SDK both expect a modern runtime).
- An OpenAI API key for AI-assisted features.

## Getting Started
1. Clone the repository and move into the app folder:
   ```bash
   git clone <repo-url>
   cd designspace-viz
   ```
2. Install dependencies: `npm install`
3. Create a `.env.local` (or `.env`) file in `designspace-viz/` with the environment variables your workflow requires:
   ```bash
   VITE_OPENAI_API_KEY=<sk-...>                  # Required for all OpenAI calls
   VITE_OPENAI_TAXONOMY_MODEL=gpt-4o-mini        # Optional override for taxonomy generation
   VITE_OPENAI_EMBED_MODEL=text-embedding-3-small

   VITE_SUPABASE_URL=<https://...supabase.co>    # Required for Supabase project search
   VITE_SUPABASE_KEY=<public-anon-key>
   VITE_SUPABASE_TABLE=media_docs                # Defaults to media_docs when omitted
   ```
4. Create the default taxonomy in `public/taxonomy/schema.json` (should be generated statically for now, sample json is below)
4. Launch the dev server: `npm run dev`
5. Open the app at the URL printed in the console (default `http://localhost:5173`)

### schema.json
```json
{
  "Taxonomy": [
    {
      "Aspect": "Display Medium",
      "Options": ["LED systems", "Projection mapping", "E-ink surfaces", "Reflective materials", "Mechanical light elements"],
      "Description": "Technology through which light, image, or motion becomes visible to audiences."
    },
    {
      "Aspect": "Spatial Scale",
      "Options": ["Object-scale", "Building-scale", "Plaza-scale", "Urban landmark", "Distributed network"],
      "Description": "Extent of the installation determining perception distance, visibility, and system complexity."
    },
    {
      "Aspect": "Urban Context",
      "Options": ["Public square", "Park or waterfront", "Transit hub", "Historic center", "Residential district"],
      "Description": "Social and spatial environment influencing audience behavior and meaning."
    },
    {
      "Aspect": "Content Type",
      "Options": ["Abstract light patterns", "Data visualization", "Narrative storytelling", "Ecological ambience", "Commemorative expression"],
      "Description": "Nature of the displayed information or emotion communicated through the medium."
    },
    {
      "Aspect": "Interaction Mode",
      "Options": ["Passive display", "Sensor-reactive", "User-controlled", "Crowdsourced input", "Remote or online interaction"],
      "Description": "How users or environments influence the experience in real time or asynchronously."
    },
    {
      "Aspect": "Sensing and Data Inputs",
      "Options": ["Environmental data", "Human motion or sound", "Camera vision", "Civic or mobility data", "User-generated media"],
      "Description": "Sources of information feeding generative or responsive behavior."
    },
    {
      "Aspect": "Narrative Role",
      "Options": ["Civic signal", "Environmental reflection", "Cultural memory", "Playful engagement", "Collective identity"],
      "Description": "Symbolic function shaping how people interpret and emotionally connect to the work."
    }
  ]
}
```

## Available Scripts
- `npm run dev` — start Vite in development mode with hot reloading.

## Configuration
- Taxonomy data lives in `public/taxonomy/schema.json` and is fetched at runtime.
- To enable AI-assisted exploration, set `VITE_OPENAI_API_KEY` (and optionally override `VITE_OPENAI_EMBED_MODEL`).
- To turn on Supabase similarity search for projects, provide `VITE_SUPABASE_URL`, `VITE_SUPABASE_KEY`, `VITE_SUPABASE_TABLE`.
