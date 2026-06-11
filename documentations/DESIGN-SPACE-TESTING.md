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

**Baseline result (this machine):** `PASSED 87  FAILED 0` offline (2026-06-12).

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

## 6. Iteration H — placement validity (manual UI checks)

> **§6.1–6.2 are SUPERSEDED by Iteration J (§8):** the geometric "beyond corpus
> range" band was retired when placement became evidence-anchored — there is no
> band, no clipped tooltip phrase, and no out-of-hull rendering on the primary
> path. They are kept as the record of what Iteration H shipped. §6.3 onward
> (support fill, register correction) still apply.

Prereqs: backend running with `data/projection/register_map.npz` present (fit it
with `uv run python database_pipeline.py project-align`), frontend on :3000.
Automated counterparts: `test_projection.py` (soft-clip, alignment, support) and
`uv run python database_pipeline.py project-diagnose` (the numbers).

### 6.1 The "beyond corpus range" band (superseded — see §8)

1. Open the Design Space view. **Expect:** a subtle grey band frames the
   lattice on all four sides, separated by a dashed boundary; the bottom band
   reads *"beyond corpus range"*; the legend has an *"outer band = beyond
   corpus range"* row.
2. Zoom/pan (wheel/drag) and Reset view. **Expect:** the band moves with the
   surface as one canvas; nothing clips away at the viewport edges.

### 6.2 Out-of-hull placements land IN the band, not ON the edge (superseded — see §8)

1. Generate at an empty cell near (but not at) the map's edge, or locate a node
   whose text is far from the corpus (e.g. rename a mind-map node to something
   non-architectural like "quantum accounting spreadsheet" — it re-locates).
2. **Expect:** any clipped placement renders as a dashed-ring dot *inside the
   grey band* at a continuous position — NOT stacked at the exact corner/edge,
   and NOT snapped to a lattice cell. Its tooltip says *"beyond corpus range"*.
3. **Expect:** clipped dots never join collision badges (no "2" badge that
   mixes a clipped and an in-bounds node), and the generation trail line ends
   at the dot in the band.

### 6.3 Corpus-support fill

1. Hover any generated (option) dot. **Expect:** the tooltip ends with
   *"corpus support NN%"*.
2. Compare a taxonomy aspect dot (in-distribution wording) with a freshly
   generated idea. **Expect:** the generated idea's fill is visibly more
   washed-out (low support percentile), while its stroke/colour stays the
   branch colour; legend row *"washed-out = little corpus support"*.
3. Old sessions (saved before Iteration H) still load: their nodes simply have
   no support value and render at full fill.

### 6.4 Register alignment is observable end-to-end

1. With the backend running, check `GET localhost:8000/api/projection/locate`
   responses (browser dev tools → Network) include `support` and `clipped`.
2. After one generation, the newest row of
   `llmind-python/data/projection/generate_log.jsonl` has
   `"register_aligned": true`, `"prompt_version": 3`, a per-node `"desc"`
   (2–4 sentences, project-style), and per-node `"support"`.
3. `uv run python database_pipeline.py project-log-stats` shows the v3/aligned
   rows as their own variant line (the A/B readout for H1+H2).
4. Toggle off: set `REGISTER_ALIGNMENT=false`, restart the backend, generate
   again → the new row logs `"register_aligned": false`. (Restore afterwards.)

## 7. Iteration I — dual-layer candidates + Examine (manual UI checks)

Prereqs: backend + embed/LLM servers running, frontend on :3000. Automated
counterparts: `test_projection.py` (alignment scoring, metrics, prompts v4) and
the bun suite (store brief/trail/rubric, examine utils).

### 7.1 The brief (identity layer)

1. Create a candidate; choose options for 2–3 aspects. **Expect:** the Candidate
   panel shows a "Brief — what this design is" textarea and a "Draft from
   choices" button (disabled until ≥1 choice).
2. Click **Draft from choices**. **Expect:** a spinner, then a 3–5 sentence
   project-style description appears in the textarea (~10–30 s, local LLM); the
   text embodies the chosen options rather than listing them.
3. Visit the Design Space view. **Expect:** the candidate's star sits at the
   BRIEF's position (it re-locates when the brief changes); the precedents list
   reflects the brief.
4. Edit the brief substantially and revisit the space. **Expect:** the star
   moves and a faint dashed violet **trail** connects its previous position(s).

### 7.2 Examine (Perspectives, revamped)

1. Click **Examine** in the Candidate panel. **Expect:** the Perspectives view
   opens on the "Examine" tab with the candidate pre-selected; without a brief
   it teaches instead ("write or draft the brief first").
2. With a brief: **Expect** a headline "The brief matches the composed choices
   NN%" (+ "largest divergence on ⟨aspect⟩" when applicable), and one
   **consistency strip per chosen aspect**: pole A = your choice, pole B = the
   strongest alternative (picked by data), corpus rug ticks, a violet star, and
   a percentile sentence ("more ⟨pole⟩ than NN% of real projects — scaled to
   this corpus").
3. **The leans badge:** when the brief reads closer to the rejected
   alternative, the strip shows an amber "leans to the alternative" badge —
   verify the percentile sentence agrees (low % toward the chosen pole).
4. **Rubric:** add a metric (aspect → two poles) — a "rubric" strip appears and
   persists across reloads and session save/load; deleting tree nodes GC's
   dangling rubric metrics. Redundant rubric pairs (>60% correlation) warn.
5. **Cross two metrics** tab: the old bipolar scatter, unchanged.
6. Generate at a gap while a briefed candidate is active. **Expect:** the
   newest `generate_log.jsonl` row has `"brief_context": true` and
   `"prompt_version": 4` (`project-log-stats` shows the variant under `brief`).

Validated 2026-06-11: drafted brief 709 chars; star moved from the composed
position (0.957, 0.141) to the brief's (0.647, 0.423) with support 0.46 and one
trail segment drawn; agreement 83%; 3 consistency strips + 1 rubric strip;
"leans to the alternative" correctly fired on Interaction Model (the draft read
"sensor-driven reactive", the choice was "passive viewing" — 9% percentile
agreed); zero console errors.

---

## 8. Manual checks — evidence-anchored placement (Iteration J)

After Part 11 (`/locate` places nodes at the weighted centroid of their top-5
corpus precedents; the geometric "beyond corpus range" band is retired):

1. **No band:** the surface fills its frame — no dashed outer zone, no
   "outer band = beyond corpus range" legend row, no tooltip phrase. Node
   tooltips may still say "placement approximate" (low confidence) and always
   show "corpus support NN%".
2. **The LED check (the trigger case):** locate "LED wall panels" (default
   schema). **Expect:** the dot sits in the LED-facade neighbourhood
   (≈ (0.50, 0.16) — near Taman Anggrek / SWFC / Chanel Ginza Tower diamonds),
   support ≈ 66%, never outside the map. Before Part 11 it rendered in the
   margin band — high support + "beyond corpus range" was the contradiction
   that triggered the change.
3. **Position = evidence:** select a node, open Related Projects — the
   highlighted precedent diamonds should cluster around the node's dot (they
   are the anchors that placed it). Spread anchors → dashed dot (low Jaccard
   confidence), which is the honest "my neighbours disagree" signal.
4. **Stale coords refresh:** coords cached before the change (localStorage)
   refresh on the first Design Space visit of the session (once-per-session
   relocate) — no manual clearing needed.
5. **Drift log:** new `generate_log.jsonl` rows carry `"placement": "knn"`;
   `project-log-stats` shows a `placed` column and never merges knn/umap rows.
6. **Reproducible record:** `uv run python database_pipeline.py project-align`
   prints the three-way held-out comparison — raw/corrected transform vs
   `knn (k=5)`; the kNN row should dominate (median ≈ 0.149 vs ≈ 0.179,
   clipped 0%).

Validated 2026-06-12: LED at (0.501, 0.158) amid its five anchors, support 66%,
clipped false; plaza (0.544, 0.579) support 12%; olfactory support 1% inside
the map; band/legend/tooltip phrase gone (viewBox `0 0 1000 1000`); backend
87/87, frontend 39/39 + tsc + eslint clean.
