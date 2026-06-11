# LLMind — Unified Project Report & Critical Reflection

*Compiled 2026-06-11; updated 2026-06-12 with the placement iterations (§5: support
recalibration + evidence-anchored placement, ITERATION-PLAN Parts 10–11) and a
references section (§8). Supersedes-as-overview (does not replace) the working documents
now archived in [`documentations/`](documentations/). Grounded in the original research
dissertation ("Automated Generation and Exploration of Design Space with Large Language
Models", Uchikoga, USYD) from which the mind-map core of this system derives.*

---

## 1. What this project is

LLMind is a research prototype for **LLM-assisted design-space exploration**, currently
instantiated in the media-architecture domain. It descends directly from the
dissertation's framework and prototype:

- The **design space** is treated as a *dynamic, constraint-centered conceptual
  structure* — not a fixed list of alternatives. Aspects are dimensions of creative
  constraint; options are positions on those dimensions.
- Designers interact with that structure through two operations: **informing**
  (expanding the constraint configuration — new aspects, options, perspectives) and
  **filtering** (contracting attention — selecting, pruning, focusing). Divergence and
  convergence are both inside scope; the tool's job is to keep the designer *moving*
  between them instead of fixating.
- The structure is externalized visually, because external representations are
  *operative media*: they shape what both the human and the LLM can do next.

The dissertation's prototype implemented this as: LLM-generated taxonomy → interactive
mind map → related-precedent panel → "Explore with AI" expansion. Its single-participant
study found the generated dimensions **coherent and genuinely insight-producing**
("it expands to not just being creative, but also considerate"), but **perceived as
incomplete and unexplained** ("Why these seven though, is there a reason?"), and found
the mind map good for local exploration but weak for overview (the participant preferred
a table "because you can see everything at once" and proposed a hybrid).

The present system keeps that core and adds a second, *empirical* formalization of the
design space: a frozen 2-D embedding projection of 209 real projects, into which the
designer's evolving taxonomy, generated ideas, and composed design candidates are all
placed. Everything since (Iterations A–J) is an elaboration, correction, or
instrumentation of that bet.

---

## 2. The system as a logic chain: feature → what it does for the designer

The features are listed in the order a designer would meet them, and each is tied to the
mechanism it relies on and the *specific* contribution it makes to design thinking. The
informing/filtering vocabulary is the dissertation's.

### 2.1 Taxonomy generation (project brief → structured dimensions)
**Mechanism:** the designer writes a project overview; the LLM (with self-reflection
rounds, retrieval over the corpus) returns Aspects/Options with descriptions.
**Contribution:** *informing at the domain level.* This is the dissertation's
"conceptual primer" — its study showed it surfaces constraints a designer's initial
mental model omits (the participant immediately recognized environmental data and
mobility as considerations their past studio project had missed). It converts the blank
page into a manipulable constraint structure within seconds, which is exactly the
cognitive-cost problem (Horner & Atwood, 2006) the design-space literature identifies.

### 2.2 The mind map (manipulable constraint structure)
**Mechanism:** the taxonomy rendered as an editable radial tree (add, rename, delete,
re-organize; selection drives every other panel).
**Contribution:** *filtering made physical.* Choosing where to look, pruning branches,
and renaming nodes are constraint manipulations — the literature's core requirement that
constraints be "explicit, manipulable entities". The dissertation validated the metaphor
(self-explanatory to a novice) while flagging layout/overview weaknesses — which is one
of the standing reasons the spatial view exists.

### 2.3 Related precedents (every abstract idea, anchored in real work)
**Mechanism:** the selected node's text is embedded and matched against the scraped
MAB corpus; matching projects (name, description, image) appear beside the map.
**Contribution:** *grounding.* An option stops being a plausible-sounding LLM phrase
and becomes "a thing real designers have built, here are five of them". This is both an
interpretability device (what does 'kinetic facade' actually mean in practice?) and a
fixation counterweight — the precedent set pulls attention outward to work the designer
has not seen.

### 2.4 The Design Space view (the embedding bet)
**Mechanism:** a frozen PCA→UMAP projection fit once on the corpus embeddings; corpus
projects are diamonds, taxonomy nodes are colored dots placed at the similarity-weighted
centroid of their top-5 corpus precedents (evidence-anchored placement, Part 11 — the
same anchors that drive corpus support), density shading shows where real work clusters,
and a 48×48 lattice of clickable empty cells covers the rest. Selection is shared with
the mind map — two views, one state.
**Contribution — four distinct ones:**
1. **Overview.** The whole exploration is visible at once in a stable frame — the thing
   the dissertation's participant asked for and the mind map could not give.
2. **Neighborhood reasoning.** "My idea sits among *these* real projects" is a judgment
   the tree cannot express at all. Proximity, painted honestly (see 2.7), lets the
   designer read affinity and difference spatially.
3. **Gaps as first-class objects.** Empty regions *between real work* are visible,
   pointable territory. The classical morphological promise — see what has *not* been
   tried — gets an empirical rendering: unexplored ≈ no precedent nearby.
4. **A stable map of the journey.** Because coordinates never relayout, the designer's
   accumulated ideas, candidates, and discoveries stay where they were — the view doubles
   as a record of the exploration (the "evolving, not snapshot" requirement).

### 2.5 Generate-at-gap with preview (aimed informing)
**Mechanism:** clicking an empty cell opens a **gap preview** — the real projects that
bracket that location (deterministic seeds), nearby already-explored ideas, and which
aspect new options would join — *before* any LLM time is spent. Confirming runs a
location-conditioned generation: seeds that surround the gap, an explicit
"fill the gap, don't imitate or average" prompt, descriptions written project-style,
and the result drawn with a trail from the clicked cell to where each idea actually
landed (drift), parented under the spatially-nearest aspect.
**Contribution:** *informing with intent.* Standard LLM ideation answers "give me more
like X". This answers "give me what belongs in the space between these precedents that
nobody has built" — a question a designer cannot even pose in a chat interface. The
preview step doubles as transparency: the designer sees and can veto the evidence the
generation will be conditioned on (a direct answer, at interaction level, to the
dissertation's finding that unexplained AI structure erodes trust).

### 2.6 Relevance lens (query-responsive painting)
**Mechanism:** toggle the lens, pick an anchor (selected node or active candidate);
every corpus project is recolored by its **true 768-d cosine similarity** to the anchor
(cool→warm), bypassing 2-D distortion entirely.
**Contribution:** *filtering the precedent field through one idea.* Where the related
panel gives the top-5, the lens shows the **whole landscape's response** to an idea —
including secondary clusters of relevance in regions the 2-D layout placed far away.
Most valuable with a candidate as anchor: "where does my composed design resonate
across everything that exists?"

### 2.7 The honesty layer (calibrated trust in the map)
**Mechanism:** every claim the map makes carries a measured caveat — layout
**trustworthiness** (0.76) in the legend; per-node placement **confidence**
(true-vs-2D neighborhood overlap; dashed when low); per-node **corpus support**
(percentile of real-metric evidence; washed-out fill when low — since Part 11 this is
also the *only* outsideness signal: placement is a convex combination of precedent
positions, so "off the map" geometry can no longer occur, and the retired "beyond
corpus range" band's job moved to the channel that was honest all along); generation
**drift** drawn as trails rather than hidden.
**Contribution:** *the difference between a picture and an instrument.* A designer can
act on spatial claims only if the tool says when they are unreliable — and "this idea
has little precedent evidence" is itself design information (it may mean *novel*, or
*incoherently phrased*; the support score plus precedents lets the designer tell which).
This layer is also the project's scientific answer to the known failure mode of evocative
spatial visualizations (see §3).

### 2.8 Perspectives / semantic axes (designer-defined dimensions)
**Mechanism:** pick two aspects; each becomes a bipolar axis between two of its option
poles; corpus and options are scored by exact cosine difference — no UMAP, no
stochasticity — with diagnostics that warn when poles are too similar or axes redundant.
**Contribution:** *re-projection on the designer's own terms.* The UMAP map's axes mean
nothing; here the axes are the designer's constraints, so an empty quadrant has a
readable morphological meaning: "no real project is strongly ⟨A⟩ *and* strongly ⟨B⟩."
It is the one view where the literature's design space and the visualization coincide
exactly.

### 2.9 Design candidates (a point in the space is a *configuration*)
**Mechanism:** compose one option per aspect (from any view, including a click-to-pick
flow in the mind map); the composition is embedded and drawn as a star on the map; its
closest real precedents are retrieved in the true metric; candidates can be compared and
exported; rejecting an option visibly invalidates candidates that used it.
**Contribution:** *closing the deepest conceptual gap.* In morphological terms a point
in a design space is a combination of choices — not an option, not a project. Until
Iteration C the system had no representation of the very object designers are trying to
create. The star + its precedent neighborhood + comparison turns the map from "a field
of fragments" into a place where *designs* can be positioned, judged against precedent,
and contrasted — convergence support inside a divergence tool.

### 2.10 Memory of the exploration (provenance, sessions, stats, export)
**Mechanism:** generated nodes carry provenance (which click, which seeds); the whole
exploration state saves/loads as versioned JSON; a live strip counts options, generated
ideas, rejections, explored cells, candidate diversity; export produces a markdown
record of the structure, choices, and evidence.
**Contribution:** *reflection and accountability* (Dove et al., 2016, "design space
reflection"): the record of *why an idea exists* and *what the exploration covered* is
what lets a designer — or a researcher — revisit, justify, and resume. These are also,
explicitly, the study instruments for evaluating everything above.

### 2.11 Researcher backstage (not designer-facing, by design)
Drift/clip/support logging per generation; `project-log-stats`, `project-calibrate`,
`project-align`, `project-diagnose` CLIs; register-gap correction; 100+ automated tests.
**Contribution:** the validity case. None of the designer-facing claims above (gaps are
meaningful, placements are usable, generation fills gaps) is left as an assertion — each
has a number, a log, and a reproducible check behind it.

---

## 3. The point-cloud argument, revisited with evidence

The dissertation reviewed three representation traditions and rejected each in part:
QOC trees (rigorous but rigid, no ripple effects), the **Design Space Schema** (pragmatic
but static), and **Heape's point clouds** — dynamic and evocative, but "dense and
idiosyncratic, requiring extensive explanation… more inspirational artifacts than
practical tools for systematic reflection, communication, or comparison." Mind mapping
was chosen as the workable compromise.

The post-dissertation argument embodied in this codebase is: **embeddings change the
point-cloud verdict.** The assessment, grounded in what was actually measured here:

**What embeddings genuinely fix about point clouds:**
- **Idiosyncrasy → shared geometry.** Heape's clouds were hand-authored; every diagram
  needed its author. Here positions are computed from data under one frozen metric —
  the same idea lands in the same place tomorrow, for anyone.
- **Inspirational → queryable.** The map answers questions: what is near this point
  (seeds), how relevant is everything to this idea (lens), where does my *composition*
  sit (candidates), what lies between these precedents (generate-at). A hand-drawn cloud
  answers none.
- **Uninterpretable → anchored in precedent.** Regions mean something because real,
  inspectable projects occupy them. Interpretation-by-example replaces
  interpretation-by-author.
- **Incomparable → cumulative.** The frozen projection makes explorations comparable
  across sessions, users, and (eventually) studies — the property Heape's notation
  most lacked.

**What the project's own measurements say the embedding map is — and is not:**
- Trustworthiness 0.760 at k=15 (Venna & Kaski, 2001): about a quarter of true
  neighborhood structure does not survive projection. Corpus self-confidence ≈ 0.25.
  Before Iteration H, 47% of generated ideas silently pinned to the surface edge; the
  register fixes (Part 9) then cut held-out short-text displacement only modestly
  (~12 lattice cells), and Part 11 located the residual culprit in the *placement
  mechanism itself*: UMAP's out-of-sample transform mis-flagged 30–35% of **real
  corpus projects** as "beyond corpus range" on round-trips of their own short texts.
  Replacing it with evidence-anchored interpolation (§5) cut median displacement to
  ~8 cells and made off-the-map geometry structurally impossible.
- Corpus support, after the Part 10 recalibration, is read against what a real project
  scores *when described at node length* — under the old full-register yardstick every
  node-length text flattened toward the 0th percentile regardless of how precedented
  the idea was. The default taxonomy now spans 0–66% (mean 18%), so corpus poverty is
  finally legible per category instead of uniform noise.
- Therefore the honest claim is: **the map is meaningful and precise as *relative
  neighborhood structure over precedent evidence* — not as absolute coordinates.** Dots
  are neighborhoods, not positions; gaps are "no precedent nearby", not guaranteed
  voids; positions of ideas phrased unlike project descriptions are extrapolations and
  are now visibly marked as such.
- A subtler conceptual limit (Iteration-plan Part 2): an embedding manifold is a
  similarity landscape of *texts*. It is not, by itself, the morphological space of
  *configurations* the literature means by "design space". The candidate stars and the
  semantic-axes view are precisely the bridges built over that gap — one places
  configurations into the landscape, the other rebuilds axes that mean what the
  designer's constraints mean.

**Verdict:** the argument survives contact with the evidence, with one inversion worth
stating plainly. Heape's clouds failed because they required *extensive explanation*.
The embedding map **also requires explanation** — legend, confidence, support — but the
explanation is now quantitative, per-element, and machine-generated rather than
idiosyncratic and authorial. The cost did not disappear; it became measurable and
honest. Part 11 sharpened the inversion: one entire explanation (the "beyond corpus
range" band) was deleted not by hiding it but by making the geometry incapable of
producing the artifact it explained. That is a real and defensible improvement, and the
remaining explanations are still a UX burden the next iteration must reckon with
(see §4).

---

## 4. Critical reflection: what tightly serves the core purpose — and what doesn't

Core purpose, restated from the dissertation: *help designers diverge — construct,
navigate, and reshape a constraint-centered design space; surface overlooked
perspectives; resist fixation; keep the designer in the informing↔filtering loop.*

### 4.1 Tightly connected
- **Taxonomy → mind map → precedents** — the validated core; the only part any real
  user has ever touched (P1), and the dissertation's evidence is positive on exactly
  the core-purpose terms (surfaced implicit constraints, broadened framing).
- **Design space + gap preview + generate-at-gap** — the strongest *new* contribution:
  it operationalizes "explore what hasn't been tried" in a way chat interaction
  structurally cannot, and the preview keeps the designer in command of the evidence.
- **Candidates** — closes the configuration gap; gives divergence a destination
  (convergence with provenance).

### 4.2 Loosely connected, with reasons

- **The Perspectives (axes) view.** Conceptually the purest feature in the system — and
  the most disconnected in practice. It is read-only (no generation, no candidate flow),
  lives behind a third navigator tab, and demands the most statistical literacy (bipolar
  cosine scores, pole quality, axis correlation). Nothing in the designer's journey
  *leads* to it. As shipped it is a researcher's instrument wearing a designer's UI.
  **Either integrate it into the loop (generate-in-axes, candidate reading as a
  consistency check, axes suggested from the designer's current focus) or demote it to
  a diagnostic.** Its current state is the clearest case of W1 (feature accretion).

- **The relevance lens.** Honest and cheap, but its designer task overlaps the related-
  projects highlight it generalizes, and the per-query "relative" normalization makes
  cross-anchor comparison — its most natural use — quietly invalid. Candidate-anchored
  relevance is its one distinctive job; consider reducing the feature to exactly that.

- **The honesty stack, as UI.** Four distinct epistemic signals still share one canvas
  (trustworthiness, confidence dashes, support fill, drift trails — the margin band
  retired in Part 11, its job folded into support). Each is
  individually justified; together they ask the designer to learn a visual epistemology
  before trusting a single dot. The *measurements* are core-purpose (they are what makes
  the map an instrument); the *presentation* has drifted toward serving the research
  argument rather than the design flow. **Collapse to one designer-facing trust cue**
  (e.g. a single "how seriously to take this dot" encoding with a tooltip), with the
  full decomposition in a diagnostics drawer.

- **The lattice itself deserves the question.** The 48×48 grid was a presentation
  convenience (clickable empty cells) that has since generated its own feature debt:
  collision badges, cell-snapping, "discovered cells", cell-granular stats. Designers
  think in ideas and regions, not cells. The gap *preview* — not the grid — turned out
  to be what makes empty space explorable. A continuous surface with freeform
  "generate around here" targeting could delete an entire family of micro-features.

- **Session/usage/stats instrumentation** is scaffolding for the study, not designer
  value — correctly built, but it should be counted as method, not feature, when
  weighing where effort goes next.

### 4.3 The honest accounting

How much did the approaches actually help the objective? Split the claims:

- **Technical claims — measured, mostly held.** Coordinate stability, bracket seeding,
  drift, clipping, the register gap: all instrumented, several already corrected on
  evidence (47% clip → diagnosed as register gap → prompt + alignment + soft margin).
  This part of the project practices what it preaches.
- **Design claims — still hypotheses.** *No designer has used any post-dissertation
  feature.* The entire feature→value chain in §2.4–2.11 is plausible, internally
  coherent, instrumented — and unvalidated. The only empirical design evidence the
  project owns (P1) validates the *old* core and two needs the new work has only
  partially served: **overview** (the spatial view is an answer, but the participant's
  literal request — a table/hybrid — was never built) and **rationale** ("why these
  seven?" remains unanswered in the taxonomy flow to this day, while transparency was
  engineered into the *spatial generation* flow instead).
- The risk this pattern names: the project has been optimizing the parts that are
  *measurable without users* (geometry, drift, register) over the parts the one real
  user actually asked for. Iteration H looked like the right way to finish that thread.
  §5 records why it wasn't finished — and why the continuation was, this time, a
  different kind of work.

---

## 5. The placement iterations (2026-06-12): what was done, and what it teaches

Two observations — both made by the project's owner *while actually using the tool* —
drove one more round of geometry work. The full record (diagnostics, alternatives,
measured results) is ITERATION-PLAN Parts 10–11; this is the report-level account and
its critical reading.

### 5.1 What was done

1. **Corpus support recalibrated** (trigger: *"many nodes show ~0% support — is this
   normal?"*). The support percentile was being read against the corpus's
   full-description self-support — a yardstick node-length text structurally cannot
   reach (real corpus projects, re-described in two sentences, averaged 11%; "Public
   plaza/square" scored 0% despite abundant precedent). The metric was signalling text
   length, not evidence. Fix: `project-align` now also fits and persists a
   **short-register support baseline** (out-of-fold corrected short corpus texts,
   self-excluded); `/locate` reads support as "as much corpus evidence as a real
   project described at this length." Validated spread: LED wall panels 33%→84% on the
   crafted probe, default-taxonomy mean 18%, honest zeros only for genuinely thin
   categories.
2. **Stale-value refresh.** The recalibration was then invisible in the UI — coords
   (with their support) persist in localStorage and only missing nodes were ever
   re-located, so values cached under the old calibration never updated. Every node now
   re-locates once per session. A small fix with a methodological moral (§5.4).
3. **Evidence-anchored placement** (trigger: *"LED wall panels is 'beyond corpus range'
   at 66% support — that contradicts itself"*). Diagnosis: the point sat 1.7% past the
   corpus bounding box (a binary flag overstating a trivial overshoot), and — worse —
   UMAP's `.transform()` had placed it 0.29 normalized units from the centroid of its
   own top-5 precedents, all real LED facades. On the only available ground truth
   (corpus short-register round-trips), the transform mis-flagged 30–35% of real
   projects as beyond range. `/locate` now places every query at the
   similarity-weighted centroid of its top-5 corpus neighbours' frozen coordinates —
   the same five anchors that drive support and the precedents panel — and the
   geometric "beyond corpus range" band is retired from the UI.

### 5.2 Why the placement change is defensible (the faithfulness question)

An earlier iteration had declined neighbour-interpolation placement on faithfulness
grounds, so the change was re-litigated rather than assumed. Three findings settled it:

- **UMAP has no principled out-of-sample extension to preserve.** Its `.transform()`
  initialises new points from their nearest *training-set* neighbours and then runs a
  few stochastic optimisation epochs (McInnes et al., 2018); the project documentation
  itself concedes transformed points land "concentrated on top of existing classes or
  spread between them" and points to a learned parametric encoder or interpolation as
  remedies (McInnes, n.d.; Sainburg et al., 2021). The choice was never "faithful
  manifold extension vs. naive kNN" — it was a noisy neighbour-based method vs. a clean
  one. The shipped placement is a top-k Nadaraya–Watson estimator (Nadaraya, 1964;
  Watson, 1964) over the frozen layout, squarely in the classical out-of-sample
  tradition of extending a fixed embedding to new points via kernel-weighted known
  points (Bengio et al., 2004).
- **The alternatives were tested, not argued.** Parametric UMAP was rejected on stated
  grounds (a deep-learning dependency, 209 training samples for a 768→2 map, and
  refitting relayouts the frozen space every persisted session depends on). Kernel
  ridge regression — the lightweight literature remedy, via scikit-learn (Pedregosa et
  al., 2011) — was CV-tuned over an (α, γ) grid and still lost to kNN on every
  statistic (median displacement 0.178 vs **0.147**; transform: 0.179).
- **Outsideness moved to the channel that was honest all along.** A convex combination
  of precedent positions cannot leave the corpus footprint, so geometric novelty
  claims are gone *by construction* — but the round-trip numbers show that channel was
  ~⅓ false positives anyway. Genuine out-of-domain-ness is carried by corpus support,
  measured in the original 768-d metric where it is faithful.

### 5.3 Strengths, read critically

- **Pulled by use, not pushed by instruments.** Unlike Iterations E–H, every change
  here began as a user-noticed wrongness ("0% everywhere", "the map contradicts the
  number") and ended as a measured, reproducible fix. That is the §4.3 risk pattern
  partially answered: geometry work is defensible when *use* demands it.
- **Honesty and simplicity moved together — rare.** The epistemic stack shrank (five
  on-canvas signals to four) while placement accuracy improved 19–45% across
  statistics. Usually instrument-honesty is bought with UI complexity; this iteration
  refunded some.
- **Position now means something a designer can verify.** "This dot sits amid these
  five real projects" is a checkable sentence — click the anchors, read them. Position,
  support, and the precedents panel draw on one evidence source instead of three
  partially-contradicting mechanisms, which is what made the LED contradiction possible
  in the first place.
- **The decision is permanently re-litigable.** `project-align` prints the three-way
  transform/kNN comparison on every refit; if a future corpus or embedding model flips
  the verdict, the evidence will say so unprompted.

### 5.4 Weaknesses, owned

1. **The failure mode was chosen, not eliminated.** The old geometry erred toward
   *false alienation* (precedented ideas exiled beyond the border); the new one errs
   toward *false familiarity* — a genuinely novel idea is pulled into the corpus
   footprint and only the washed-out fill says so. A designer who reads position but
   not fill now overestimates how precedented their idea is. This trade was taken
   knowingly (the old channel was mostly noise), but it is a trade, and the study
   (§6.2) must test whether the fill actually gets read.
2. **Void placements survive.** When an idea's five anchors straddle distinct clusters,
   its centroid lands between them, in a region belonging to nothing. The Jaccard
   confidence catches this (dashed dot), but flagging is not solving: the dot still
   *has* a position, and positions invite reading. LED itself ships with confidence
   0.11.
3. **Coherence cuts both ways.** Position, support, and precedents now share one
   retrieval. A failure in that retrieval (a register-correction artifact, an
   embedding-model quirk) no longer produces a visible contradiction — it fails
   *consistently* across all three signals, which makes errors more convincing.
   The LED bug was found *because* two independent mechanisms disagreed; that
   diagnostic tripwire has been traded away for coherence.
4. **The ground truth is a proxy.** The 19–45% improvement is measured on corpus
   round-trips — short texts *of corpus projects*. Designer briefs and generated ideas
   have no ground-truth coordinates; the claim that kNN places them better is an
   extrapolation from the proxy, not a measurement.
5. **The recalibration's real lesson is about testing.** A support metric that
   flattened every node to ~0 had shipped with a green harness — the tests verified
   the *math* (percentiles, exclusions, persistence) and never asked the
   *meaning*-level question "what should a public plaza score?" Meaning-level
   walkthroughs now exist in the testing protocol (§6–8 of DESIGN-SPACE-TESTING), but
   the pattern — instruments validating instruments — is exactly the §4.3 critique
   wearing a lab coat, and it will recur wherever a number has no human check.

### 5.5 Postscript (same day): the sticky pin, and what tracing it exposed

A third user-found defect: *"Taman Anggrek is stuck as the first related project for
every node."* The retrieval was healthy end-to-end — correct per-node queries, correct
content-specific responses — but clicking a corpus glyph ("click to view", which the
relevance lens actively invites) pinned that project to the top of the Related
Projects panel and **nothing ever released the pin**. Every later node selection
fetched the right evidence and displayed it *behind* a frozen first entry. Fixed: a
node selection now releases the pin.

Two things this small bug teaches, beyond its fix:

- **The honesty layer audits the math, not the interface.** Trustworthiness,
  confidence, support, drift — every instrument scores the *geometry*, and all of
  them were green while the panel showed the wrong evidence for every node. The
  binding between a claim ("related to your selection") and the evidence displayed
  under it is itself a correctness surface, and nothing watches it. No amount of
  measurement rigor below the UI protects the designer from a stale `useState` above
  it.
- **Tracing it exposed a real architectural inconsistency.** The §5.3 claim that
  position, support, and precedents "draw on one evidence source" is only true for
  position and support. The Related Projects panel embeds a *different query text*
  ("lineage | description | topic") with *no register correction*, and the candidate
  precedents and relevance lens also search raw — so the five anchors that *place* a
  node are not guaranteed to be the five projects the panel *shows* beside it. The
  coherence shipped in Part 11 is real but narrower than claimed; unifying the
  retrieval (one query composition, one correction policy) is now part of the
  receipts work (§6 item 5).

All three defects in this section were found by one person *using* the prototype for
minutes at a time — none by the 126-test harness. That is the strongest argument item
2 of §6 has.

---

## 6. Ways forward (next iteration — approach-level, not tech-level)

1. **Give the tool a spine, then test it.** Define the canonical journey (brief →
   dimensions → map → gap → generate → judge against precedent → compose → export
   rationale) and make the UI *teach* it through progressive disclosure — Candidate,
   lens, and Perspectives appear when the journey reaches them. Every feature that
   cannot find a place on the spine is a candidate for the diagnostics drawer or
   deletion.

2. **Run the deferred study — it is now the bottleneck for every claim.** 3–5
   participants (include non-novices this time; the dissertation flags the
   novice-only limit), think-aloud, two conditions: mind-map-only vs mind-map+space.
   That comparison directly tests the embedding/point-cloud argument (§3) on the only
   terms that matter: does the spatial view measurably change what designers notice,
   generate, and choose? The instrumentation (sessions, stats, usage counters, CSI)
   exists; the study design in the dissertation is reusable nearly verbatim. Two
   placement-specific probes now belong in it (§5.4): do participants notice the
   washed-out fill (the false-familiarity check), and do they read a dot amid its
   precedents as "related to these" (the intended semantics) or as "exactly here"
   (over-reading)?

3. **Pay the two debts owed to the only real user.** (a) A taxonomy *overview* — the
   hybrid table the participant asked for: all aspects × options with descriptions in
   one screen, clickable into map/space. Cheap, validated demand, and it would also
   become the natural home of the candidate-composition flow. (b) **Rationale for
   dimensions** — have generation return a one-line "why this dimension, anchored to
   which precedents" per aspect, shown on demand. Both findings are three years of
   iterations old and still unimplemented; both are more evidenced than anything in
   Iterations E–H.

4. **Consolidate the epistemics into design language.** Part 11 started this from the
   geometry side (the band is gone; one signal fewer). The remaining work is the UI
   side: one trust cue on canvas; plain-words tooltips ("placement is approximate —
   treat as neighborhood"; "little precedent evidence — possibly novel, possibly
   vague"); the full decomposition behind a "how this map works" panel. The honesty
   layer should make designers *braver in the right places*, not more hesitant
   everywhere.

5. **Turn support from a score into receipts — and unify the retrieval behind it.**
   The five anchor projects behind a node's support are now also the anchors behind
   its *position* — they exist, named, one click away. Replace "corpus support 12%"
   as the primary reading with the evidence itself: qualitative bands (established /
   explored / thin / uncharted) that click through to the anchor projects;
   per-aspect support aggregates in the Context panel ("Display Technology averages
   57%, Data Source 4%" — *which dimensions has the field actually explored?*);
   "uncharted" framed as invitation, not failure. Prerequisite discovered in §5.5:
   the Related Projects panel, candidate precedents, and relevance lens still search
   with a different query text and no register correction — converge all retrieval
   on one query composition and one correction policy, so the panel literally shows
   the placement anchors. The percentile stays in the diagnostics layer where it
   earns its keep. This is the report's answer to "is support a designer value or a
   research instrument?" — it is an instrument until its receipts are surfaced.

6. **Decide the lattice question deliberately** (affordance or artifact?) before any
   further feature is built on cells.

7. **Then, and only then, the corpus.** The recalibrated support makes the ceiling
   quantitative *and category-specific*: technology options read 40–66% while
   operational and sensory categories floor at 0–16% — corpus poverty is no longer a
   uniform suspicion but a map of where evidence is thin. Expansion (and a second
   domain as a generality probe) is the named next thread after the study — which
   will also show *where* corpus poverty actually hurt designers, steering what gets
   collected.

---

## 7. Document map

All prior working documents are archived, unmodified in content, in
[`documentations/`](documentations/):

| Document | What it holds |
|---|---|
| [`DESIGN-SPACE-VIZ.md`](documentations/DESIGN-SPACE-VIZ.md) | Original design-space concept, invariants, the three hard problems, M0–M3 build record |
| [`DESIGN-SPACE-ITERATION-PLAN.md`](documentations/DESIGN-SPACE-ITERATION-PLAN.md) | The full critique→iteration history: Parts 1–11 (weaknesses, Iterations A–J, measurements) |
| [`DESIGN-SPACE-PERSPECTIVES-PLAN.md`](documentations/DESIGN-SPACE-PERSPECTIVES-PLAN.md) | F1 lens + F2 axes design rationale and risks |
| [`DESIGN-SPACE-TESTING.md`](documentations/DESIGN-SPACE-TESTING.md) | Test protocol: automated harness + manual UI checks (§6 = Iteration H, §7 = Iteration I, §8 = Iteration J) |
| [`PROJECT_DEV.md`](documentations/PROJECT_DEV.md) | Early development log with per-change justifications |
| [`LEARN.md`](documentations/LEARN.md) | Layer-by-layer learning guide to the whole codebase (designer-friendly) |

Live reference docs stay with their subsystems: [`CLAUDE.md`](CLAUDE.md) (hub),
[`llmind-python/BACKEND.md`](llmind-python/BACKEND.md),
[`llmind-web/FRONTEND.md`](llmind-web/FRONTEND.md),
[`llmind-web/ZUSTAND.md`](llmind-web/ZUSTAND.md),
[`llmind-web/REACT-QUERY.md`](llmind-web/REACT-QUERY.md).

---

## 8. References

Sources consulted while building and justifying the placement iterations (§5); the
foundational works name-checked elsewhere in this report follow below them.

Bengio, Y., Paiement, J.-F., Vincent, P., Delalleau, O., Le Roux, N., & Ouimet, M.
(2004). Out-of-sample extensions for LLE, Isomap, MDS, Eigenmaps, and Spectral
Clustering. In S. Thrun, L. K. Saul, & B. Schölkopf (Eds.), *Advances in Neural
Information Processing Systems 16* (pp. 177–184). MIT Press.
https://proceedings.neurips.cc/paper/2003/hash/cf05968255451bdefe3c5bc64d550517-Abstract.html

McInnes, L. (n.d.). *Transforming new data with UMAP*. UMAP documentation. Retrieved
June 12, 2026, from https://umap-learn.readthedocs.io/en/latest/transform.html

McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation
and Projection for dimension reduction*. arXiv. https://arxiv.org/abs/1802.03426

Nadaraya, E. A. (1964). On estimating regression. *Theory of Probability & Its
Applications, 9*(1), 141–142. https://doi.org/10.1137/1109020

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O.,
Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A.,
Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn:
Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
https://jmlr.org/papers/v12/pedregosa11a.html

Sainburg, T., McInnes, L., & Gentner, T. Q. (2021). Parametric UMAP embeddings for
representation and semisupervised learning. *Neural Computation, 33*(11), 2881–2907.
https://doi.org/10.1162/neco_a_01434

Venna, J., & Kaski, S. (2001). Neighborhood preservation in nonlinear projection
methods: An experimental study. In G. Dorffner, H. Bischof, & K. Hornik (Eds.),
*Artificial Neural Networks — ICANN 2001* (pp. 485–491). Springer.
https://doi.org/10.1007/3-540-44668-0_68

Watson, G. S. (1964). Smooth regression analysis. *Sankhyā: The Indian Journal of
Statistics, Series A, 26*(4), 359–372. https://www.jstor.org/stable/25049340

**Foundations cited elsewhere in this report:**

Dove, G., Hansen, N. B., & Halskov, K. (2016). An argument for design space
reflection. In *Proceedings of the 9th Nordic Conference on Human-Computer
Interaction (NordiCHI '16)* (Article 20). ACM. https://doi.org/10.1145/2971485.2971528

Heape, C. (2007). *The design space: The design process as the construction,
exploration and expansion of a conceptual space* [Doctoral dissertation, University
of Southern Denmark].

Horner, J., & Atwood, M. E. (2006). Effective design rationale: Understanding the
barriers. In A. H. Dutoit, R. McCall, I. Mistrík, & B. Paech (Eds.), *Rationale
management in software engineering* (pp. 73–90). Springer.
https://doi.org/10.1007/978-3-540-30998-7_3

Uchikoga. (n.d.). *Automated generation and exploration of design space with large
language models* [Unpublished dissertation]. The University of Sydney.
