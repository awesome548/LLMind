# USER-TESTING-PLAN.md — draft study plan (2026-06-13, Iteration L)

A value-proposition-centred test plan. Method skeleton inherits the
dissertation's protocol (think-aloud + semi-structured interview + CSI); the
*framing* follows the prototyping/testing guide's chain: research → value
proposition → concepts → what to measure → synthesis → considerations. The
guide's core philosophy, adopted throughout: **test the value proposition,
not the interface** — every task and question traces to a VP component, the
prototype needs only enough fidelity for people to get a genuine sense of
that proposition (ours exceeds it: major workflows are real, not mocked),
and synthesis must answer *which aspects resonated, which more than others,
and what investment each requires* — not "did they like it."

---

## 1. Research grounding (audience, journeys, pains/gains)

**Target audience.** (a) *Experienced* interaction/media-architecture
designers and design researchers — the dissertation's named gap (n=1 novice
"limits the depth of our findings… future research should involve
experienced practitioners"); they can judge structure against established
taxonomies. (b) *Novice* design students — contrast stratum; the prior study
showed the tool acts as a "conceptual primer" for them.

**Key journeys.** The flows in [USER-FLOWS.md](USER-FLOWS.md): entry choice
(F0), interrogate structure (F2), filter + reflect (F3), inform vocabulary
and structure (F4/F4b), discover spatially (F5), compose + steer (F6), hunt
gaps (F7), revisit (F8).

**Pains** (from the literature + prior study): constructing/maintaining
design spaces is cognitively expensive (Horner & Atwood); prompt–response
LLM use converges prematurely; AI structures arrive unexplained ("why these
seven though?"), which corrodes trust; documentation tools die of
documentation burden (Dalsgaard & Halskov).

**Gains sought:** divergence with structure; evidence one can interrogate;
a process record that writes itself.

**Risks & challenges to probe** (not paper over): fixation on AI-generated
vocabulary (Wadinambiarachchi); trust without ground truth; jargon density;
local-LLM latency tolerance; small-corpus grounding limits.

## 2. The value proposition and its components

> **LLMind turns design-space construction from a documentation burden into
> a live, evidence-grounded instrument: the AI structures and explains, the
> designer commits and steers, and the whole exploration stays inspectable
> and replayable.**

| # | Component | One-line claim | Embodied by |
|---|---|---|---|
| VP1 | **Grounded, explained structure** | Every dimension and count can answer "why?" with real-project evidence | taxonomy + annotation receipts + rationale layer + coverage probe (F2, F4b) |
| VP2 | **Manipulable constraints** | Informing and filtering are direct operations, never prompt engineering | choose/reject/reopen/facets/add/proposals (F3, F4) |
| VP3 | **Honest instruments** | Measurement advises and is veto-able; the system never silently decides | steering veto cards, requested-vs-achieved, fidelity/confidence labels (F5, F6) |
| VP4 | **A reflective record that writes itself** | The why survives without documentation burden | AI-drafted reflections, event log, replayable timeline (F3, F8) |

## 3. Concepts explored and what each addresses

Per the guide: be clear about the different concepts and how each addresses
the VP — they are alternatives *within* one tool, so the test can compare
their resonance directly.

| Concept (view/instrument) | Addresses | The bet being tested |
|---|---|---|
| Schema table + receipts + rationale | VP1 | overview + evidence is what builds trust (the prior study's table preference) |
| Design-space surface + lens | VP1, VP2 | spatial discovery surfaces gaps that lists hide |
| Cross-tab + generate-into-gap | VP1, VP2 | *exact, nameable* gaps beat diffuse map gaps for actionability |
| Candidate + strips + steering | VP3 | designers accept AI moves when the move is single, measured, veto-able |
| Timeline + reflections + proposals | VP4 | a self-writing record gets revisited, not just recorded |
| First-run brief-first vs discover-first | VP2 | entry framing changes how owned the space feels |

## 4. What to measure (guide: feelings, importance, resonance)

1. **Feelings toward the VP** (+ / − / ?) — elicited per VP component at
   interview, after the tasks that exercised it; "?" (confusion) is a
   first-class outcome, not noise.
2. **Importance/relevance level** — would this change your real workflow?
   Which component would you miss most if removed? (forced ranking of VP1–4
   at the end.)
3. **Behavioural signals** (free, from the event log per session): options
   generated vs rejected vs chosen; proposals accepted/dismissed/
   reconsidered; probe used; replay opened; reflections accepted as-drafted
   vs edited vs skipped (the `edited` flag is the burden-inversion metric);
   facet and lens usage.
4. **CSI questionnaire** (Cherry & Latulipe) — comparable across
   participants and with the prior study.
5. **Trust delta (RQ1 extension):** coherence/completeness/trust questions
   asked once after seeing the schema WITHOUT the rationale layer visible
   (facilitator collapses it), once with — within-subject, order fixed,
   ~5 min. This directly tests whether the rationale layer answers the
   prior study's "why these seven?" finding.

## 5. Session protocol (60–75 min, think-aloud throughout)

1. **Intro + consent** (5) — recording, AI disclosure, design-space primer
   (same script as the prior study for comparability).
2. **Tutorial** (10) — facilitator walkthrough of F1/F2/F3 only; F4–F8 are
   left for discovery (discoverability is itself a measure).
3. **Blind ideation** (2) — before seeing the generated structure: "name the
   dimensions YOU would explore for this brief." Kept as the anti-fixation
   baseline; overlap with the system's aspects is coded later. (Protocol
   step now; candidate for an in-tool feature — ITERATION-PLAN L-C.)
4. **Tasks** (30):
   - **T1 (VP1):** "Get an overview of this design space. What does each
     dimension mean, and do you believe the counts?" — schema, receipts,
     rationales; includes the §4.5 trust-delta probe.
   - **T2 (VP2):** "Commit to a direction for the brief: choose three,
     reject one with a reason." — reflection chips fire naturally.
   - **T3 (VP1/VP2):** "Find something nobody has built and make the tool
     generate it." — free choice between surface gap and cross-tab cell
     (which they pick is data; both exist for this comparison).
   - **T4 (VP3):** "Your candidate should feel more interactive — make the
     tool move it, then judge the move." — steering + veto card.
   - **T5 (VP4):** "Walk me back through what you did and why. Anything you
     dismissed that deserves another look?" — timeline, reflections,
     reconsider.
5. **Semi-structured interview** (15) — per VP component: feelings (+/−/?),
   the "would you miss it" ranking, co-agency question (collaborating vs
   using a tool — comparability with the prior study), trust and rationale
   probing.
6. **CSI** (5–10).

**Participants:** 3–5 experienced + 2–3 novice. **Pilot first** (1 session)
to time the tasks and tune wording.

## 6. Prototype scope for the test (guide: enough fidelity for the VP)

Already met: interaction supports the major workflows end-to-end (no wizard
-of-oz), visual polish pass done (luminance hierarchy, plain-language
labels). Known and disclosed honestly rather than hidden: local-LLM latency
(annotation cold runs minutes — facilitator pre-warms the cache before
sessions; generation ~1 min is part of the experience and worth observing),
small corpus (209 projects, one domain), single-user. Out of scope for the
test: corpus expansion, collaboration, mobile.

## 7. Synthesis plan (guide: resonance → investment)

After all sessions, per VP component: (a) what resonated / did not (coded
think-aloud + interview feelings), (b) which resonated MORE (the forced
ranking + behavioural signals — e.g. if nobody reopens the timeline
unprompted, VP4's "gets revisited" bet failed regardless of what interviews
say), (c) of those that did, what investment is required to support them
(map to the ITERATION-PLAN Part 13 menu: e.g. VP1 resonates but trust-delta
is flat → invest in receipts UX, not more rationale text; VP4 resonates →
build L-D SnapShot compare).

## 8. Considerations (stakeholders / UX / visual / engineering)

- **Stakeholders:** supervisor (study design sign-off, ethics), participants
  (consent, recording, AI-content disclosure), the dissertation timeline.
- **UX:** the heuristic-review backlog (TESTING §14 round) is the known-issue
  ledger — test findings should be triaged against it, not re-discovered.
- **Visual design:** color semantics are load-bearing (amber=corpus,
  violet=commitment, emerald=generated) — color-vision check in the screener
  or rely on the redundant cues (icons, italics, position).
- **Engineering:** sessions need backend + LM Studio up with caches
  pre-warmed (annotation + rationale for the session taxonomy); a fallback
  taxonomy ready if a participant's brief-first generation fails live; save
  the session JSON at the END of every session (it is the data).
- **Next steps / priorities:** filled by §7's output — the synthesis decides
  which Part 13 items get built, in which order.
