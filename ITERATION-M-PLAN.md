# ITERATION-M-PLAN — from the verification sweep to the study

*Drafted 2026-07-03, from PROJECT-REPORT §5.7 (the code-verification sweep), §6 (ways
forward), and the §5.6 open questions. Two tracks with different readers:*

- **Part I — Engineering** is an execution spec: each item has a concrete fix design,
  a *meaning-level gate* (the §5.4 lesson: tests must ask what a result should MEAN,
  not just whether the math ran), and enough file/line precision to act on without
  re-derivation. Nothing in Part I is done yet.
- **Part II — Design-research** is for reading: the questions the code cannot answer,
  organized by *who* can answer them (study participants / an offline experiment /
  the project owner), each tied to the report section that raised it.
- **Part III — Sequencing** says what must land before the pilot session, and why.

The standing constraint over everything: **§6 item 2 — the study — is the bottleneck
for every claim.** Part I exists to protect the study's validity and the
participants' first contact; it must not become another month of building instead of
recruiting.

---

## Part I — Engineering track

### I.A Confirmed defects (PROJECT-REPORT §5.7, D1–D6)

#### M-E1 · D1 — Make the "too-broad" diagnostic reachable
`backend/corpus/annotate.py:116–129` (`diagnostics_for`), caller at `:284`.

- **Problem.** Counts are censored at the judged shortlist (`k = min(30, corpus)`),
  but the threshold is `0.8 × full corpus` (= 167.2 of 209). No option can ever
  flag. The unit test passes only by feeding `count=180`, an input the pipeline
  cannot produce (`test_projection.py` ~`:369`).
- **Fix design.** Change the semantics from "share of the corpus" (unmeasurable — we
  never judge the whole corpus) to **shortlist saturation**: pass the shortlist size
  into `diagnostics_for(counts, n_projects, shortlist_k)` and flag
  `c >= TOO_BROAD_SHARE * min(shortlist_k, n_projects)` (= 24 of 30 under current
  config). Update the docstring to state the censoring explicitly: *when
  `count == shortlist_k` the true count is a lower bound* — the option matched
  everything we showed the judge. Update the schema-table badge copy to match the
  honest reading ("matches ≥24 of its 30 closest corpus projects — likely too broad
  to discriminate").
- **Gate (meaning-level).** Against the *current cached annotations*: the known
  count-30/30 option (cache file `61dba4978bf0712d.json`) MUST flag; "LED wall
  panels" (count 9) MUST NOT. Replace the impossible-input unit test with one that
  derives its threshold from a realistic shortlist size.
- **Effort:** small. **Risk:** none — pure function + one call site + one test.

#### M-E2 · D2 — Stop dropping quoted-number membership arrays
`backend/corpus/annotate.py:79–103` (`parse_membership`).

- **Problem.** `["1","2"]` (a common local-LLM formatting) parses as valid JSON,
  then every string element fails `isinstance(v, (int, float))` → the chunk counts
  **zero** members, silently, and the wrong count is cached. A second latent nit
  found while specifying the fix: `bool` is an `int` subclass in Python, so a stray
  `[true]` currently counts as project 1.
- **Fix design.** Per-element coercion in the array branch:
  exclude `bool`; accept `int`/`float`; accept `str` whose stripped value is all
  digits (`int(v)`); drop everything else. Keep the existing except-branch salvage
  and the bare-integer fallback untouched.
- **Cache consequence — decision required (see M-R9c).** Fixing the parser changes
  what past runs *would* have counted. Per the PROCESS.md §2 rule ("cache poison is
  silent"), the correct move is to **bump `ANNOTATION_VERSION`** and re-annotate —
  which costs a cold run (~6–10 min/option × 26 options on the local stack,
  overnight job). The cheap alternative (keep the cache, fix forward) leaves counts
  of unknown correctness under the receipts the study will show participants.
  Recommended: bump + re-run before the pilot; re-verify the §5.6 gate ("LED wall
  panels" lists the known LED facades) afterward.
- **Gate.** Unit: `parse_membership('["1","2"]', 5) == [1, 2]`;
  `parse_membership('[true, 2]', 5) == [2]`; existing cases unchanged.
  Meaning-level: post-re-annotation, the LED gate passes and
  `mean_shortlist_acceptance` is recorded in the report with its new date.
- **Effort:** small (fix) + one overnight re-annotation run. **Risk:** low.

#### M-E3 · D3 — Sanitize session files on load
`llmind-web/src/lib/session-io.ts:46` (`parseSessionFile`), store `restoreSession`
at `mindmap-store.ts:537`.

- **Problem.** Only `nodes` is validated; an explicit `"coords": null` (corrupt or
  hand-edited file) passes, spreads over the store default, and crashes at render —
  outside the load handler's try/catch, with no error boundary → white screen.
  Session files are the **study's capture format** (§2.10), so this is study-data
  robustness, not just polish.
- **Fix design.** A hand-rolled `sanitizeSnapshot` in `session-io.ts` (no new
  dependency): for each slice, type-check against its expected shape (`coords`
  object, `candidates` object-of-objects, `events` array, `reflections` object,
  `optionState` object, strings as strings); on mismatch, substitute the
  initial-state default and collect a warning list. `parseSessionFile` returns
  `{snapshot, warnings}`; the load handler surfaces warnings as a toast ("session
  loaded; 2 malformed sections were reset") instead of silently dropping data.
- **Gate.** Unit: files with `coords: null`, `candidates: []`, `events: null`,
  `reflections: 7` each load with defaults + the right warning; a valid file
  round-trips **byte-identical** (sanitizer must be a no-op on good data).
- **Effort:** small-medium. **Risk:** low; pure function + one call site.

#### M-E4 · D3-adjacent — Add a render error boundary
`llmind-web/src/app/` (new `error.tsx`).

- **Problem.** Any future render throw (not only D3's) is an unrecoverable white
  screen mid-study-session.
- **Fix design.** A minimal App-Router `error.tsx`: shows the error, offers
  "Reload view" — and, because the store persists, a reload loses nothing. One
  line in the study protocol: if the boundary ever appears during a session, note
  the timestamp (the event log gives the reproduction).
- **Gate.** Manually throw in a component in dev; boundary renders; reload restores
  the exploration.
- **Effort:** trivial.

#### M-E5 · D4 — Resolve mind-map highlight by node id, not label
`llmind-web/src/components/mindmap/simple-mindmap.tsx:214` (+ `convertNode` `:46`),
caller `page.tsx:2034`.

- **Problem.** External selection sync resolves via `topicToId[label]`, which keeps
  only the first id per duplicate label — selecting the second of two same-named
  options (which the schema explicitly supports) highlights the wrong tree node.
  Every other view resolves by `selection.nodeId`; the mind map is the anomaly.
- **Fix design.** Pass `activeNodeId={selection.nodeId}` into `SimpleMindMap`;
  in the sync effect, prefer `activeNodeId` (mind-elixir nodes already carry the
  same ids — `convertNode` maps them) and fall back to `topicToId[activeTopic]`
  only when no id is present.
- **Gate.** With two "Modular" options under different aspects: selecting the
  second in the schema highlights the *second* in the tree. Existing tree tests
  unchanged.
- **Effort:** small.

#### M-E6 · D5 — Filter the placeholder row everywhere it leaks
`llmind-web/src/app/mindmap/page.tsx:2138` (badge), `:2155` (panel);
the filter already exists at `:831` and `:1296` — it is simply absent at two sites.

- **Fix design.** Extract one `isPlaceholderProject(p)` helper (name match, as the
  existing guards do); filter `data.projects` once into `realProjects`; badge counts
  it, panel receives it, and the panel gets a proper empty state ("no closely
  related corpus projects for this node") instead of a selectable non-project.
- **Gate.** With retrieval returning only the placeholder: badge shows 0, panel
  shows the empty state, map highlights none — the three surfaces agree (the D5
  finding was precisely their disagreement).
- **Effort:** small.

#### M-E7 · D6 — Close the submit_keyed race
`llmind-python/backend/jobs.py:73–86`.

- **Problem.** Check-then-act: the lock is released between the existence check and
  `submit()` + `_keyed[key]` write, so two simultaneous identical requests can both
  start full annotation runs. Note: `submit()` itself acquires `_lock` (`:44`), so
  naively holding a plain `Lock` across the whole body would **deadlock**.
- **Fix design (pick one; (a) recommended).**
  (a) Change `_lock` to `threading.RLock()` and hold it across
  check → `submit()` → `_keyed` write — minimal diff, re-entrant by construction.
  (b) Restructure: under one lock hold, check, create the `_jobs` entry inline,
  write `_keyed`, and only then hand `_run` to the executor.
- **Gate.** A threaded unit test: N threads hit `submit_keyed` with the same key
  through a barrier; exactly one job id comes back for all N (extend the existing
  `test_jobs_dedup`).
- **Effort:** small. **Risk:** low, but touch nothing else in the file.

### I.B Hygiene (from §5.7)

#### M-E8 — Replace the raw NUL byte with an escape
`llmind-web/src/features/design-space/hooks/use-annotation-query.ts` (~byte 1149).
Same string value, written as `' '` — git/grep stop treating the file as
binary. Gate: `git diff` shows text; `bun test src` green; the query key value is
unchanged (annotation cache behavior identical).

#### M-E9 — Fix the landing page
`llmind-web/src/app/page.tsx`. Remove the dead `/projects` card; describe the tool
honestly; one card → `/mindmap` ("Open LLMind"). Trivial; it is the first thing a
study participant could accidentally see.

### I.C Refinements (pre-existing debts the sweep re-confirmed)

#### M-E10 — The Self-Refine decision, instrumented (blocks on M-R9a)
`generate_taxonomy.py:225–242` (commented-out loop), API field `num_reflections`.

- **Problem.** §2.1's corrected state: the API advertises a mechanism the code
  doesn't run. Leaving it is a standing honesty gap.
- **Options.**
  (a) **Re-enable** behind `num_reflections > 1` — on OpenAI mode only at first;
  on the local 4k-context thinking stack a full reflection round likely cannot fit
  (taxonomy JSON + prompt + thinking), and PROCESS §2 rules apply.
  (b) **Remove** the parameter from API + frontend + docs — honest one-shot.
  (c) **Replace** with one cheap critique-then-revise pass (a smaller second call).
- **Recommendation:** implement (a) minimally *with logging* so M-R9a (an offline
  quality experiment) can decide whether reflection earns its latency; if it
  doesn't, fall back to (b). Don't ship a default change before the experiment.
- **Gate.** With `num_reflections=2` on OpenAI mode: two LLM calls logged; the
  second receives the first's taxonomy; output still schema-valid. With `=1`:
  byte-identical behavior to today.

#### M-E11 — Unify the retrieval behind one query + one correction policy
(§6 item 5's prerequisite; §5.5's architectural inconsistency, re-confirmed.)

- **Problem.** Placement/support embed `"topic. desc"` **with** register
  correction; the Related Projects panel embeds
  `"lineage | description | topic"` **raw** (`related_projects/service.py:338`);
  candidate precedents (`/corpus/similar`) and the relevance lens also search raw.
  So the five projects that *place* a node are not guaranteed to be the five the
  panel *shows* — the §5.3 "one evidence source" claim is only true for two of the
  three signals.
- **Fix design.** (1) One `compose_query(node)` helper used by both paths — decide
  the canonical composition (recommend placement's `"topic. desc"`, since it is the
  measured one; lineage can be evaluated as an A/B in the generate-log style before
  deletion). (2) Extend the shared `embed_texts` entry
  (`corpus/service.py:139`) with an `apply_register: bool` flag; panel, `/similar`,
  and relevance all opt in. (3) `/related-projects/search` local path returns the
  SAME anchors `/locate` used.
- **The §5.4-weakness-3 trade, addressed rather than ignored.** Full coherence
  removes the diagnostic tripwire that caught the LED bug (two independent
  mechanisms disagreeing). Mitigation: keep `project-diagnose` printing a
  *raw-vs-corrected top-5 comparison* per probe text — the tripwire moves from the
  UI (where it confused a user) into the diagnostics CLI (where it belongs).
- **Gate (meaning-level).** For every node in the default taxonomy: the panel's
  top-5 == the placement anchors (assert via a new offline check). The LED probe
  still lists the known LED facades. Relevance-lens painting unchanged in rank
  order for a spot-checked anchor.
- **Effort:** medium — the largest Part-I item. **Risk:** medium (touches three
  retrieval surfaces); do it AFTER the pilot unless the pilot protocol depends on
  panel/anchor identity (it doesn't — see Part III).

#### M-E12 — Study-mode instrumentation (the §6 item 2 remainder)
The one *new* build the study needs; deliberately thin:

1. **Participant tag**: an optional `participantId` field in the store/session
   snapshot, set via a small dialog (or URL param `?p=P2`), stamped into every
   event-log entry and session filename.
2. **One-click study bundle**: a navigator action that exports one archive —
   session JSON + event log + exploration stats + usage counters + the markdown
   export — named `bundle-<participant>-<timestamp>`.
3. **Pre-warm checklist as a script**: a `study-warm` CLI (or documented curl
   sequence) that runs annotation + rationale for the default schema so cold-cache
   latency (§5.6 weakness 4) never lands on a participant.
- **Gate.** A full mock session produces one bundle; every event row carries the
  tag; loading the bundle's session restores the exploration.
- **Effort:** small-medium. **This is pilot-blocking** (Part III).

#### M-E13 — Reproducible statistics for the report
Small CLI addition (`database_pipeline.py annotation-stats` or extend
`project-diagnose`): print `mean_shortlist_acceptance`, count spread, and
diagnostics from the current annotation cache — so report figures like §5.6's
0.231/0.262 are always regenerable with a dated command instead of hand-computed.
Gate: command output matches a hand check once. Effort: trivial-small.

---

## Part II — Design-research track

*The engineering above protects instruments; none of it produces knowledge. These
are the open questions, by who can answer them. Each names the report section that
raised it and the decision it informs.*

### II.A Questions only study participants can answer

**M-R1 — The study itself (§6 item 2 — the only zero-progress item).**
Everything in this plan is subordinate to running it. Status: USER-FLOWS.md and
USER-TESTING-PLAN.md are drafted; missing are a pilot, recruitment (3–5
participants, deliberately including non-novices — the dissertation flags the
novice-only limit), and M-E12. The two conditions (mind-map-only vs
mind-map+space) directly test the §3 embedding/point-cloud argument on the only
terms that matter: does the spatial view measurably change what designers notice,
generate, and choose?

**M-R2 — Does the rationale layer buy *calibrated* trust, or just more trust?
(§5.6 weakness 1 — the sharpest open question.)**
The model that generated the taxonomy also explains it, post-hoc, from evidence.
A fluent self-justification that *reads* grounded is exactly what a trusting
participant cannot distinguish from a true one. The drafted trust-delta probe
measures *gain*; it cannot measure *calibration*.
*Method sketch:* within-subject, rationale visible for half the aspects; include
**one planted weak dimension** with a fluent, evidence-styled rationale. Calibrated
trust = participants accept good dimensions more AND still reject the plant;
mere trust = the plant gets accepted too. *Needs from engineering:* a study flag to
inject a planted aspect + rationale (half a day, fold into M-E12 if this probe is
adopted). *Decision informed:* whether the rationale layer ships as a default-on
feature or an on-demand one.

**M-R3 — Does structure-level AI proposal anchor harder than option-level?
(§5.6 weakness 2; Wadinambiarachchi et al., 2024.)**
The coverage probe proposes *dimensions*; accepting one reshapes everything
downstream. The mitigation (veto chips, reconsiderable dismissals) is untested at
this level. *Method sketch:* the event log already records per-aspect attention
(selections, generations, choices); compare its distribution before/after an
accepted probe proposal vs after a manually added aspect. *Needs from
engineering:* nothing — `exploration-stats` computes per-aspect activity; add a
before/after cut in analysis, not in the tool. *Decision informed:* whether probe
proposals need friction (e.g., a mandatory "what would this dimension miss?"
prompt) or are safe as designed.

**M-R4 — Are the honesty signals read at all? (§5.4 weakness 1; §6 item 4.)**
The single-trust-cue collapse is *blocked on this*: collapsing four signals into
one is only right once we know which ones carry designer-relevant information.
*Method sketch:* think-aloud "how seriously would you take this dot?" moments in
the study tasks; instrument legend opens and tooltip hovers (usage counters
exist). Watch specifically for the two §5.4 probes already drafted: does anyone
notice the washed-out fill (the false-familiarity risk the placement trade
created), and is a dot read as "related to these five" (intended) or "exactly
here" (over-reading)? *Decision informed:* §6 item 4's remaining design — which
signal becomes the one on-canvas cue.

**M-R5 — The lattice question (§6 item 6): affordance or artifact?**
Do designers aim generations at *cells* or at *regions*? *Method sketch:*
observation first, no A/B — watch where participants click relative to visual
cluster structure; ask "what did you expect to happen when you clicked there?"
If nobody thinks in cells, the freeform "generate around here" redesign deletes a
family of micro-features (collision badges, cell-snapping, discovered-cell stats).
*Decision informed:* whether any future feature may couple to cells.

**M-R6 — Does anyone reach Perspectives? (§4.2's residual; the K5 decision.)**
The testing plan deliberately leaves F4–F8 untutored to measure discoverability.
If no participant travels cross-tab → scatter → axes unprompted, the K5 lens-bar
end state (dissolving the standalone mode) stops being a question and becomes the
answer. *Decision informed:* K5.

**M-R7 — Is brief-first viable live? (§5.6 weakness 4.)**
A fresh taxonomy costs minutes of annotation + rationale on the local stack even
pre-warmed for the *default* schema — a participant's own brief generates a *new*
taxonomy whose annotation is necessarily cold. *Method sketch:* in the pilot, time
the full brief-first path honestly; decide whether the study runs brief-first as a
final task (latency absorbed in a debrief), with a "warming" experience, or
discover-first-only with brief-first as a demo. *Decision informed:* study task
order; possibly a progressive-annotation design (annotate aspects on demand as
they're viewed) as a later engineering item.

### II.B Questions answerable offline (no participants needed)

**M-R9a — Does Self-Refine improve taxonomy quality? (pairs with M-E10.)**
Generate N taxonomies from the same brief with reflections 1 vs 2–3 (OpenAI mode);
compare with instruments that already exist: annotation count spread (does
refinement reduce unprecedented/saturated options?), coverage-probe results (fewer
poorly-covered projects?), corpus similarity, plus a blinded side-by-side judgment
by the owner. Cheap, and it converts M-E10 from a taste decision into a measured
one. *(Honest caveat: the owner judging blind is one rater; treat the result as
directional, not decisive.)*

**M-R9b — What did D2 actually cost?**
After the parser fix, re-annotate (M-E2) and diff per-option counts old-vs-new.
If counts barely move, the defect was latent, and that fact belongs in §5.7's
record; if they move materially, the gate-run conclusions (spread 18→0, five
unprecedented) need re-verification — before the study shows anyone receipts.

**M-R9c — Owner decisions needed now (blocking Part I ordering):**
1. **ANNOTATION_VERSION bump + overnight re-run** — approve? (M-E2; recommended
   yes, before the pilot.)
2. **Self-Refine**: option (a) re-enable-and-measure / (b) remove / (c) redesign
   (M-E10; recommended (a) then decide by M-R9a).
3. **Retrieval unification timing** — after the pilot (recommended) or before?
4. **Recruitment**: who are the 3–5, and does at least one have media-architecture
   or adjacent practice experience (the dissertation's named limitation)?
5. **Planted-dimension probe (M-R2)** — adopt it? It is the only probe here that
   tests the *failure mode* of the project's central trust bet, but it adds a
   deception element the ethics protocol must cover.

### II.C The horizon item (unchanged, deliberately last)

**M-R10 — Corpus expansion and the second domain (§6 item 7).**
Nothing in this plan advances it, on purpose: the annotation already names *where*
the corpus is thin (five unprecedented options in the gate run), and the study
will show where thinness actually hurt designers. Expansion before that evidence
would repeat the §4.3 pattern (optimizing what is measurable without users).

---

## Part III — Sequencing

**Wave 1 — pilot-blocking (do first, in one pass):**
M-E3 + M-E4 (session/render robustness — the study's capture format must not
white-screen), M-E5, M-E6, M-E9 (participant-facing correctness), M-E2 + M-E1 with
the ANNOTATION_VERSION bump and overnight re-run + M-R9b diff (the receipts shown
to participants must be right), M-E12 (study mode), M-E13, M-E8.

**Wave 2 — the pilot itself (M-R1):** one participant, full protocol, timing the
brief-first path (M-R7); adopt or drop the planted-dimension probe (M-R9c-5)
before recruiting the rest.

**Wave 3 — post-pilot engineering:** M-E7 (race — real but low-harm; any time),
M-E11 (retrieval unification — after, unless the pilot protocol grows a dependency
on panel/anchor identity), M-E10 + M-R9a (Self-Refine, in parallel with study
sessions since it is offline).

**Deliberately not in this plan:** the single on-canvas trust cue (§6 item 4 —
blocked on M-R4), the K5 lens-bar end state (blocked on M-R6), lattice removal
(blocked on M-R5), restart-from-moment (K9 — "say the word"), corpus expansion
(M-R10). Each is blocked on evidence this plan is designed to produce, and building
any of them now would be the §4.3 risk pattern again.
