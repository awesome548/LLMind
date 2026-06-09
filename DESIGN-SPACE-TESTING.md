# Design-Space Visualization — Testing & Evaluation Protocol

How to verify the prototype works (correctness) **and** judge whether the
visualization is meaningful (quality). Two layers: automated checks you can run
unattended, and a manual protocol for evaluating the actual research value.

Prerequisites:
- Corpus index built: `data/local_index.npz` (+ `.meta.json`).
- Projection fit: `uv run python database_pipeline.py project` → `data/projection/{model.joblib,surface.json}`.
- For live tests: the embedding + LLM server in `.env` running (LM Studio / vLLM at `VLLM_BASE_URL`).

---

## 1. Automated tests (`llmind-python/test_projection.py`)

A self-contained harness (plain asserts, no pytest). Run from `llmind-python/`:

```bash
uv run python test_projection.py            # offline: math, artifacts, stability, service
uv run python test_projection.py --http     # + GET /api/projection/surface (backend running)
uv run python test_projection.py --live      # + /locate and /generate-at (embedding+LLM running)
```

Exit code is non-zero if any check fails. What each layer covers:

| Layer | Needs | Key invariants checked |
|---|---|---|
| Offline math | nothing | fit→transform **deterministic**; coords in `[0,1]`; OOD points **clipped**; clusters separate; grid round-trips; nearest-point correctness |
| Artifacts | `data/projection/*` | reloaded model **reproduces persisted surface exactly** (stability); embedding dim matches fit dim; empty cells exist |
| Service | artifacts | `load_surface`, `nearest_corpus` return well-formed data; nearest of a point is itself |
| HTTP | backend up | `GET /surface` → 200 with points + grid |
| Live | embed + LLM | `/locate` places 2 distinct prompts at **distinct** locations in `[0,1]`; `/generate-at` → 200 with nodes + seed neighbours + coordinates |

**Baseline result (this machine):** `PASSED 31  FAILED 0` with all servers up.

### Run the servers for `--http` / `--live`

```bash
# Backend (note: use uvicorn directly — `fastapi run`'s banner crashes on the
# Windows cp1252 console trying to print an emoji):
PYTHONIOENCODING=utf-8 uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Frontend:
cd ../llmind-web && BACKEND_URL=http://127.0.0.1:8000 bun dev
```

---

## 2. Manual smoke test (the round-trip)

1. Open `http://localhost:3000/mindmap`. Dismiss the taxonomy dialog (or generate a taxonomy).
2. Click **Design Space** (top-center toggle). Expect:
   - A lattice of dots fills the surface.
   - Warm/orange dots cluster where real projects are dense; grey dots are empty space.
   - Colored dots appear for taxonomy nodes (needs the embedding server, else only the background shows).
   - A legend (bottom-left).
3. Hover an empty dot → tooltip "empty / N nearby project(s) · generate here". Hover a colored dot → its topic.
4. **Selection sync:** click a colored node dot → the Context panel lineage updates. Toggle back to **Mind Map** → the same node is the active selection. (Two views, one selection.)
5. **Generate-at:** select a branch first, switch to Design Space, click an empty dot. Expect a spinner at that cell, then new colored nodes appear, *and* they are present in the Mind Map view too.

---

## 3. Quality evaluation (does the space mean anything?)

Code correctness ≠ a meaningful design space. These are the judgments that matter
for the research claim. None are pass/fail — record observations.

### 3.1 Neighbourhood faithfulness (UMAP trustworthiness)
Distances after UMAP are distorted. Quantify how much:

```bash
uv run python -c "
import numpy as np
from sklearn.manifold import trustworthiness
from pipeline import projection as proj
from config import settings
d = np.load(settings.local_index_path, allow_pickle=True)
X = d['vectors']
m = proj.load_model(settings.projection_dir)
print('trustworthiness:', round(float(trustworthiness(X, m.transform(X), n_neighbors=10)), 3))
"
```
**Interpret:** ~0.9+ is good for UMAP; <0.8 means the layout misleads and the
neighbourhood-seeded generation rests on shaky ground. Re-fit with larger
`--neighbors` / different `--min-dist` and compare.

### 3.2 Semantic coherence of regions
Pick 3–4 dense (warm) cells. For each, `GET /api/projection/surface`, take that
cell's nearest corpus ids, and read their names/descriptions (in `local_index.npz.meta.json`).
**Ask:** do the projects in one region share a recognizable theme (e.g. kinetic
façades vs. participatory installations)? If neighbouring warm cells are
thematically unrelated, the manifold isn't capturing design structure.

### 3.3 Do taxonomy nodes land where they should?
After `/locate`, check whether an option dot sits near corpus projects that
exemplify it. E.g. a "kinetic façade" option should land in/near the region of
actual kinetic-façade projects. Mismatch = the node text and the corpus text
embed into different neighbourhoods (often a content-mode or prompt-wording issue).

### 3.4 Is generation actually *filling the gap*?
The core interaction claim. For a clicked empty cell at `(x,y)`:
- Inspect `seed_neighbours` in the `/generate-at` response — are they genuinely the
  nearest real projects to that spot?
- Re-`/locate` the generated node's topic. **Does it land near the clicked cell**,
  or drift far away? Drift is expected sometimes (the LLM's idea didn't fit the
  gap) and is itself a signal — but *systematic* large drift means the
  spatial-neighbour seed isn't steering generation. Measure mean drift over ~10
  clicks across the surface.

### 3.5 Stability across growth (the "two sides of one coin" guarantee)
Generate several times, expand branches, reload the page. Existing dots must **not**
move. (Automated stability test covers the math; this confirms the UX honours it.)

### 3.6 Determinism / reproducibility
Re-run `database_pipeline.py project` with the same seed → `surface.json` coordinates
should be identical. Different `--seed` → different layout (UMAP is seed-sensitive);
decide whether to pin the seed for your study.

---

## 4. Failure modes & what they mean

| Symptom | Likely cause | Fix |
|---|---|---|
| `/locate` → 502 "Embedding dim … != fit dim" | runtime embedding model ≠ the one the index was built with | rebuild index + `project`, or set `VLLM_EMBED_MODEL` back |
| `/locate` / `/generate-at` → 502 "Failed to embed" | LM Studio / vLLM not running or wrong `VLLM_BASE_URL` | start the server; check `.env` |
| Design Space shows "surface unavailable" | no `data/projection/surface.json`, or backend down | run `database_pipeline.py project`; start backend |
| Colored node dots never appear | embedding server down (locate is best-effort, swallowed) | start the server; reselect/regenerate to retry |
| All nodes pile into one cell | degenerate projection (too few corpus points / bad params) | increase corpus size; tune `--neighbors`, `--pre-pca` |
| Coordinates differ run-to-run | `--seed` not pinned, or index changed | pin seed; rebuild deterministically |
| Spinner never clears after click; backend logs 200 but UI stuck | Next.js dev `rewrites()` proxy doesn't deliver responses for long (50s+) requests | fixed: frontend calls the backend directly (`NEXT_PUBLIC_API_BASE_URL`, default `http://localhost:8000`) with CORS — do **not** route long LLM calls through the proxy |
| Dot-click → 502 "Failed to generate nodes…" | generation backend mismatch (e.g. OpenAI mode with no key) | `/generate-at` derives the backend from `VECTOR_STORE` (local→vllm); the 502 detail now includes the underlying cause |

---

## 5. What is *not* covered (known limitations of this prototype)

- **Corpus points are shown as density only**, not individually selectable dots
  (the `surface.json` carries them; rendering them is a future toggle).
- **No prompt-level "fill the gap" instruction** yet — generation is steered purely
  by passing spatially-nearest projects as context (spatial-neighbour RAG). The
  explicit instruction and UMAP `inverse_transform` seeding are deferred (see
  DESIGN-SPACE-VIZ.md §10, M4).
- **2D only.** The pipeline supports `--dims 3`, but the renderer is 2D.
- **Multimodal** is out of scope by design (text-only for now).
- **Cell collisions:** when two nodes map to one cell, only the first is drawn.
- Frontend types for `/api/projection/*` are hand-written in
  `src/features/design-space/types.ts`; regenerate `openapi.ts` to fold them in.
