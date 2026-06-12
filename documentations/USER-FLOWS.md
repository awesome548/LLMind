# USER-FLOWS.md — LLMind user flows (drafted 2026-06-13, Iteration L)

The prototype's journeys, written for two audiences: design reviewers (does
each flow serve the value proposition?) and study facilitators (each flow is
a task template; the **Logged** lines name the instrumentation the session
records for free). Flows reference the value-proposition components defined
in [USER-TESTING-PLAN.md](USER-TESTING-PLAN.md) §2 (VP1 grounded structure,
VP2 manipulable constraints, VP3 honest instruments, VP4 reflective record).

Conventions: **[view]** = where the step happens. All LLM steps are async
jobs with visible status and a cancel/escape path; none block exploration.

---

## F0 — First-run entry choice (inform-first vs discover-first)

*The dissertation participant's layered model: "write down what you're
imagining… and then it gives you ideas depending on that."*

1. First launch (no taxonomy, never offered before) → **choice dialog**:
   **Start from your brief** or **Discover first**.
2. Brief-first → Generate Taxonomy dialog; the project overview field IS the
   brief → taxonomy generated → tree + schema rebuilt around it; the corpus
   similarity notice warns when the brief sits far from the corpus.
3. Discover-first → dialog closes; the prebuilt media-architecture space is
   the starting point. Generate Taxonomy stays one click away (navigator).
4. Offered once, ever (usage-tracked) — either choice is valid.
5. **The brief stays editable:** after the first generation the navigator's
   "Generate Taxonomy" becomes **"Edit Brief & Taxonomy"**, reopening the
   dialog prefilled with the persisted brief — revise and regenerate (the
   timeline keeps the previous space's history across the boundary).

**Logged:** `first_run_brief` / `first_run_discover`, `taxonomy_set` event,
`projectBrief` in sessions.

## F1 — Orientation (tree + context + precedents)

1. **[tree]** Select any node → Context panel shows lineage breadcrumb
   (ancestors clickable), description, provenance chips; Related Projects
   panel shows corpus matches with images.
2. Drill down/up freely; corpus project click pins it for inspection.

**Logged:** selection-driven queries only (no event noise — attention is not
commitment).

## F2 — Understanding the structure (schema + evidence + why)

*VP1's core loop: structure you can interrogate.*

1. **[schema]** Structure → Schema: aspects as columns, options as cells;
   ring = chosen, struck = rejected, italic = informed (legend in the strip
   tooltip).
2. Annotation runs (first run minutes, cached after) → per-option **counts
   with receipts**: click a badge → the exemplifying projects, click through
   to inspect. Granularity diagnostics flag too-broad / unprecedented.
3. **Rationale layer:** each aspect header gains a one-line *why* ("why:
   …"), grounded in the counts; the Context panel repeats it when the aspect
   is selected, labelled "(AI, from corpus evidence)".
4. **Coverage probe:** the strip counts projects the taxonomy describes
   poorly → "N projects fit poorly — probe for a missing dimension" → LLM
   names what they exemplify → arrives as accept/dismiss chips (F4b).

**Logged:** annotation cache state, `coverage_probe`, rationale cache state.

## F3 — Filtering (commit, reject, facet) + reflection capture

*VP2 (filtering = self-imposed constraints) + VP4 (the why survives).*

1. **[any view]** Choose an option for the active candidate (Context panel
   button, schema hover ✓, or the armed pick flow — global violet banner,
   Esc cancels). Reject with an optional reason; reopen lifts it.
2. **[schema]** ± facet chips → the map fades non-matching corpus projects
   (fade, never remove).
3. After choose/reject/steer/keep: a **reflection chip** appears bottom-right
   with an AI-drafted one-line why — Enter accepts, typing edits, Esc skips.
   Never modal.

**Logged:** `choose`/`unchoose`/`reject`/`reopen` events with refs,
`facet_toggle`, reflections with `edited` flag.

## F4 — Informing: new vocabulary (options)

1. **[tree]** Generate Nodes under a selected node (related projects as
   context), or
2. **[space]** click an empty cell → **gap preview** (seeds + nearby ideas,
   no LLM committed) → Generate here → options arrive with descriptions,
   coordinates, drift, and provenance chips, or
3. **[schema]** + add option (manual typing), or
4. accept a **proposal chip** (steer qualities / kept cell ideas — F6/F7).

**Logged:** `generated` / `option_added` events, provenance
(`generate-at`/`generate-nodes`/`manual`/`steer`/`cell`), generate_log.

## F4b — Informing: new structure (aspects, via the coverage probe)

*The structure-level informing loop the dissertation asked for ("is there no
more?").*

1. **[schema]** Probe chip (F2.4) → 1–2 proposed missing dimensions arrive
   as amber chips ("Add as a new dimension?") with the motivating projects
   as evidence.
2. Accept → a new aspect column (provenance `coverage`, no options yet) —
   fill by hand or generation. Dismiss → recorded; reconsider later from the
   timeline.

**Logged:** `coverage_probe`, `proposal_accepted` / `proposal_dismissed`
(full proposal in event detail), `option_added` event for the new aspect.

## F5 — Spatial discovery (the surface)

1. **[space]** Pan/zoom the corpus map: pale amber = context, vivid = related
   to selection, node dots colored by branch (fill = corpus support, dashed =
   approximate), candidate stars, density heat. Legend explains every glyph;
   "map reliability" labels the projection honestly.
2. **Relevance lens** (switch): paints the corpus by similarity to the
   selected node or active candidate (anchor named in the pill, switchable).
3. Click a corpus diamond → inspect the real project.

**Logged:** `lens_on`, `view_space`, locate/relocate logs.

## F6 — Composing and steering a candidate

*VP3's core loop: move in language, measure honestly, veto freely.*

1. Choose one option per aspect (F3) → the composition embeds as a **star**
   among its closest real precedents; Inspector dock shows
   concept↔commitments agreement + per-choice consistency strips.
2. Write/draft the **brief** (identity layer) — the star moves, leaving a
   trail.
3. **Steer:** click or DRAG along a strip rail to aim the ghost star (the
   Steer/Cancel controls sit inside the same strip card), or use a
   precedent's ⇢pull/⇠push → ONE deliberate LLM revision → **veto card**
   with requested vs achieved (along/orthogonal) → Apply or discard.
4. Applied steers propose their named qualities as options (chips).

**Logged:** `candidate_created`/`candidate_deleted`, `steer_applied`,
steer_log.jsonl, proposal events.

## F7 — Gap hunting (cross-tab)

1. **[cross-tab]** Pick two aspects → option×option grid; cells list the
   real projects combining both; empty = unexplored combination.
2. Empty cell → **Generate into this gap** (seeded with half-matching
   precedents) → veto preview → **Keep as candidate** → candidate skeleton
   (the two choices + the concept as brief), Candidate panel opens, the
   concept proposes itself under both aspects.

**Logged:** `cell_kept` event, generate_log (`kind: cell`).

## F8 — Reflecting and revisiting (the timeline)

1. **[schema]** Timeline pill → Fusion-style marker strip: one icon per
   step, playhead bubble, reflection badges.
2. Click a marker → the schema AS IT STOOD (read-only banner with step N of
   M + Back to now; not-yet-existing options ghosted; the step's subjects
   outlined amber); the detail card shows the full label, time, kept
   reflection, and **Reconsider** for dismissed suggestions.
3. Back to now → live editing resumes.

**Logged:** `replay_opened`, `proposal_reconsidered`.

## F9 — Capture and exit

1. Save session (full state JSON) / load (replaces, with confirm).
2. Export the exploration record (markdown: taxonomy + states, candidates,
   provenance, stats, reflections).
3. The exploration stats strip (options, generated, rejected, chosen
   aspects, cells explored, candidate spread) doubles as the study's
   instrument.

**Logged:** the session file is itself the complete instrument (events +
reflections + usage + stats).
