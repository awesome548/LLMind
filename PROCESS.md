# PROCESS.md — session handoff (written 2026-06-12, mid–Iteration K)

Working memory for the next session. The authoritative plan is
[Part 12 of the iteration plan](documentations/DESIGN-SPACE-ITERATION-PLAN.md)
("Iteration K: the living schema"); this file records where execution stands,
the reasoning that is easy to lose in context compression, and the exact next
steps. Supersedes nothing — update or delete it as phases complete.

---

## 1. Where we are

**Iteration K, Phase A (the schema spine) is implemented; its final gate was
in flight when this file was written.**

Done and verified:
- **A2 backend** — `POST /api/corpus/annotate` (202 job): per option,
  register-corrected embedding shortlist (top-30) → chunked local-LLM
  membership calls → `{count, project_ids, projects:[{id,name}]}` receipts +
  Halskov granularity diagnostics. Cached per option content hash under
  `data/projection/annotations/` (`ANNOTATION_VERSION` salts the hash).
  Module: `backend/corpus/annotate.py`; prompt `ANNOTATE_OPTION_PROMPT` in
  `utils/prompts.py`. Backend suite 99/99.
- **A1 frontend** — `schema-table.tsx`: aspects × options table; chosen =
  ring, rejected = struck, generated = italic (Halskov's "informed"), count
  badges → receipts popover → opens project in Related Projects panel;
  in-table choose / reject / reopen / add-option (manual informing). Pure
  view models + tests in `features/design-space/schema-utils.ts`.
- **A3 facets** — transient ± include/exclude per option (page-level state,
  never persisted); non-matching corpus glyphs FADE on the map
  (`facetMatched` prop on the surface). `computeFacetMatches` is pure+tested.
- **Structure mode** (user review decision): the schema is NOT a fourth
  mode — the nav has **Structure | Design Space | Perspectives**, and
  Structure hosts a top-center **Tree ↔ Schema** toggle (`view` values
  `'map'` / `'schema'` internally). Panels icon-collapse below xl in schema
  view, same grammar as Perspectives.
- Gates: frontend tsc/eslint clean, 45/45 bun tests; openapi.ts regenerated;
  TESTING.md §9 walkthrough; BACKEND/FRONTEND/REACT-QUERY.md rows added.

**Pending when this file was written** (check before anything else):
- The **v3 annotation run** over the default schema's 26 options — verify via
  `data/projection/annotations/` (26 files) and that "LED wall panels" lists
  Taman Anggrek-class receipts (ground truth: corpus is LED-saturated; ANY
  zero there = judgment still broken). The spot-check protocol is the gate.
- `diag_check.py` / `diag_full_annotate.py` in `llmind-python/` are
  THROWAWAY diagnostics — delete after the gate passes.

## 2. Hard-won local-stack constraints (do not relearn these)

The user's LM Studio stack: **Qwen3.6 35B A3B (thinking model), 4096-token
context, plus nomic-embed for embeddings.** Three failures taught three rules:

1. **4096 covers prompt + thinking + answer.** A 30-project prompt exceeded
   it outright; evidence must be CHUNKED (`JUDGE_BATCH = 5`).
2. **Thinking CANNOT be disabled on Qwen3.6** — it is thinking-only:
   `/no_think` and `chat_template_kwargs.enable_thinking=false` are both
   ignored (verified live; both produced 100% reasoning, empty content).
   Capped budgets burn entirely inside reasoning and answer nothing.
   The working recipe (annotate.py v4 + `backend/corpus/llm.py`): chunks
   small enough that the deliberation finishes (~5 items ≈ 0.7k prompt +
   ~1.5k think), dynamic `max_tokens = LOCAL_CTX − estimate_tokens(prompt)
   − margin`, and a reasoning-tail salvage (`salvage_from_reasoning`) when
   the cap is still hit. Cost: ~75 s/chunk → ~6-10 min/option cold; the
   per-option cache amortises it. TWO corollaries learned the hard way:
   (a) token estimates must be CHARSET-AWARE — CJK text runs ≥1 token/char,
   so a flat chars//3 understated a Chinese-heavy chunk and the request
   overflowed the window mid-run; (b) even so, treat the estimate as a
   guess: `budgeted_completion` catches the server's context-size 400 and
   retries with a halved budget (the rejected call never ran — retrying is
   free).
3. **Cache poison is silent** — wrong verdicts cache like right ones. Bump
   `ANNOTATION_VERSION` (salts every option hash) whenever prompt/summary
   content changes, and wipe `data/projection/annotations/`.

These rules apply to ALL future LLM features (B2 cell generation, B3
steering, C2 reflections). Also remember: **uvicorn has no --reload — kill
and restart the :8000 backend after every backend edit** (memory file exists
for this).

## 3. The reasoning chain (compressed)

How we got here — each step recorded fully in the docs noted:

1. **Part 11** (iteration plan): `/locate` placement = similarity-weighted
   centroid of top-5 corpus anchors (UMAP `.transform()` retired to fallback;
   measured: kNN median displacement 0.149 vs 0.179, clip 0% vs 35%).
   "Beyond corpus range" band removed. Evidence rule established: **deltas
   as rulers and briefs, never constructors** (analogy-arithmetic literature
   is half-folklore; contrast directions are measurements).
2. **Support became receipts-bound**: percentile vs short-register baseline
   (in `register_map.npz`), but the real designer answer is *which projects*
   — which led to…
3. **Six-source evidence review** (Part 12 K0 table): Halskov & Lundqvist
   (filtering = INVESTIGATION not pruning; filter→inform loop), Halskov MAB
   (annotated schema, counts, cross-tabs, empty cells = exact gaps),
   Luminate (dimension re-layout validated; ungrounded dimensions weak),
   Onarheim & Biskjaer (choices/rejections/briefs = self-imposed
   constraints), Dalsgaard & Halskov PRT (reflection capture, burden-
   inverted), dissertation (table + rationale debts).
4. **The re-centering** (Part 12 K1): the deep model is a **living
   design-space schema**; views are lenses (Structure tree/table, Map =
   evidence lens, Cross-tabs = morphological lens, Inspector = filtering
   instruments). The map is no longer the protagonist.
5. PROJECT-REPORT.md §5 carries the critical reflection (incl. §5.5: the
   honesty layer audits the math, not the interface; retrieval paths not yet
   unified — report §6 item 5).

## 4. Next steps, in order

**Phase B (B1 inspector dock, B2 cross-tab lens, B3 steering v1) is
IMPLEMENTED** — all suites green (backend 114, frontend 51, tsc/eslint
clean); B1 verified live in the preview. Remaining:

1. ~~Phase A gate~~ **CLOSED** (2026-06-12): full 26-option table verified —
   LED 9 receipts (Taman Anggrek / Dash Wall / Xindie / Novartis…), spread
   18→0, no too-broad, 5 unprecedented, `mean_shortlist_acceptance` 0.231.
   Recorded in Part 12 K2 notes; diag scripts deleted.
2. ~~Backend restart + openapi regen~~ **DONE** (includes `jobs.submit_keyed`
   dedup: concurrent clients share one annotation job).
3. ~~B3 live verification~~ **DONE** (strip rail +0.70 → veto card, requested
   +0.70 vs achieved +0.22, named qualities). B2 live verification in
   progress (cross-tab counts confirmed real; generate-into-gap →
   keep-as-candidate being exercised).
4. ~~Adversarial reviews~~ **DONE** (Phase B code fleet: 17 confirmed
   findings fixed; UI/UX fleet: 23 confirmed findings triaged/fixed —
   canvas grammar now covers schema + cross-tab, popovers flip at edges,
   panel columns fit all heights/widths ≥900px, first-run dialog offers
   once).
5. ~~Phase C~~ **IMPLEMENTED + verified live** (2026-06-12): C3 events slice
   (capped 500, kind-specific refs, recorded inside store actions + component
   commit points) + schema replay slider (`buildReplayOverlay`, read-only);
   C2 reflections (`POST /api/reflections/draft`, chip fills only if the
   designer hasn't typed, Enter/edit/Esc, in sessions + markdown export);
   C1 proposal chips (steer named_qualities + kept cell ideas → accepted
   options carry provenance `steer`/`cell`). Deferred: the alignment
   uncovered-quality emitter (needs its own LLM pass + caching; the proposal
   channel is ready for it).
6. **The study** can run NOW — Iteration K is complete end to end.
7. ~~Timeline upgrade~~ **DONE** (2026-06-13): the replay slider became a
   Fusion-style marker timeline (`replay-timeline.tsx`) — icon markers,
   playhead bubble, detail card with reflections + Reconsider; scrubbing
   ghosts not-yet-existing options and amber-outlines the step's subjects.
   Decisions recorded in ITERATION-PLAN **K9**: universal timeline DEFERRED
   (staged: tree-diff commits → thin investigation events → full replay),
   restart-from-moment WORTH BUILDING (scoped to commitments/filters,
   append-only `rolled_back` event — say the word), git-like branching
   REJECTED (candidates + session save/load are the branching mechanism).
8. **Design language locked** (2026-06-13, user directive): clear information
   hierarchy via luminance/saturation, never dark-border emphasis; apply
   Norman/Nielsen to every UI change. Codified in FRONTEND.md §Design
   language; surface + axes corpus glyphs muted vs vivid accordingly.
9. ~~Heuristic-critique remediation~~ **DONE** (2026-06-13): six-lens
   Norman/Nielsen/Gestalt review fleet → user picked all but
   touch-reachability + keyboard access (out of scope: desktop research
   prototype). Landed: replay read-only banner in the schema status strip
   (step N of M + Back to now); armed pick-mode global banner with Esc
   cancel; plain-language pass (map reliability, corpus-support tooltip,
   cell-grammar tooltip, "unexplored combination", percentile phrasing);
   chips offset accounts for the OPEN timeline; two-click candidate delete
   (3s auto-disarm); lens pill always names its anchor. All verified live.
10. **Iteration L round 1 DONE (2026-06-13): L-A + modified L-B** (user
    chose from the Part 13 menu; status in ITERATION-PLAN **13.3**).
    L-A: `/api/corpus/rationale` (per-aspect why under schema headers +
    Context panel; cached per aspect+counts in `data/projection/rationales/`)
    and the coverage probe (`poorlyCoveredProjects` + `/api/corpus/
    missing-aspect` → aspect-kind proposal chips → new schema column with
    provenance `coverage`). Live-verified: probe proposed
    "Spatial-Perceptual Integration", accepted into the schema. L-B became
    a once-only first-run choice dialog (brief-first → taxonomy dialog vs
    discover-first), verified via localStorage swap. Gates: backend
    134/134, frontend 66 tests, tsc/eslint clean; openapi regenerated;
    backend restarted (task b80yka2f0).
11. **Study paperwork drafted:** USER-FLOWS.md (F0–F9, per-flow
    instrumentation) + USER-TESTING-PLAN.md (VP-centred: 4 VP components ×
    5 tasks, trust-delta probe for the rationale layer, CSI, synthesis →
    investment mapping). Next: pilot session, then L-C/L-D/L-E/L-F per the
    menu.
12. **Code-verification sweep (2026-07-03):** every load-bearing PROJECT-REPORT
    claim audited against the implementation (multi-agent sweep, adversarially
    verified findings). Corrections applied in place; **PROJECT-REPORT §5.7** is
    the authoritative record. Six confirmed defects await fixing — headline:
    the annotation "too-broad" diagnostic is mathematically unreachable
    (`annotate.py:124`, threshold 80%-of-corpus vs counts capped at 30) and
    `parse_membership` silently drops quoted-number JSON arrays
    (`annotate.py:90`) — both §5.4-item-5-class findings whose green tests
    validate impossible inputs. Also: session-load validation (session-io.ts:46),
    duplicate-label mind-map highlight (simple-mindmap.tsx:214), placeholder row
    counted as a project (page.tsx:2138), submit_keyed race (jobs.py:79). Test
    counts as of this date: backend 152 checks (139 offline), frontend 66.
    **Execution spec: ITERATION-M-PLAN.md** — Wave 1 (pilot-blocking fixes +
    study-mode instrumentation) → pilot → Wave 3; five owner decisions are queued
    in its §II.B (M-R9c), incl. the ANNOTATION_VERSION bump/re-run.
13. **Steering-rail polish + brief editing (2026-06-13):** the "pixelated"
    strip star was the clipped-score dashed outline breaking into dots at
    20px → clipped stars now render solid at 0.7 opacity, clamped to the
    rail edge, with a tooltip (TESTING §12.1b). Rails are draggable
    (pointer-captured; `draggingRef` not state — same-frame moves) and
    Steer/Cancel render INSIDE the strip card. L-B follow-up: `projectBrief`
    persisted in the store/sessions; after first generation the navigator
    button becomes "Edit Brief & Taxonomy", reopening the dialog prefilled
    (derived-value pattern). All live-verified except the relabel itself
    (this session's `taxonomy` is null — the tree grew from the default
    schema — so the label correctly stays "Generate Taxonomy" until a real
    brief+generation round).

## 5. Environment notes

- Branch `local-experiements`; everything uncommitted (user has not asked to
  commit). Backend on :8000 runs as a background task; LM Studio at
  `VLLM_BASE_URL` (:1234) serves both models. Preview/dev server :3000.
- The frontend annotation query fires only in the Schema view
  (`useAnnotationQuery`, staleTime ∞, runJob timeout 20 min); server-side
  per-option cache makes re-entry cheap. First run on a NEW taxonomy costs
  26×3 capped calls (fast without thinking).
- Raising LM Studio's context-length load setting would relax §2.1 — user's
  call; the chunked design intentionally doesn't depend on it.
