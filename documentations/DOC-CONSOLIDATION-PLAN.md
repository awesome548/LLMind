# DOC-CONSOLIDATION-PLAN — merging overlap, cutting redundancy

*Drafted 2026-07-03. **EXECUTED same day** with the owner's decisions: LEARN.md
slimmed-to-path (option a, now live with §9.2/§11.5 fixed in place); live-vs-archived
split as recommended (stated in root CLAUDE.md; USER-TESTING-PLAN + USER-FLOWS
declared live study SSOT, with the ITERATION-M probes folded into USER-TESTING-PLAN
§9); backend triplication collapsed (README folded into BACKEND.md "Data pipeline &
corpus CLIs", CLAUDE.md trimmed to hub); Mind-elixir.md deleted (wrap-note added to
FRONTEND.md); PROJECT_DEV.md banner-superseded; PROCESS.md §3 cut to pointer; plus
the owner-requested **PROJECT-REPORT §1.3** (post-dissertation justifications + the
relationship map). This file is now itself a record — safe to delete once the
CLAUDE.md doc table feels sufficient. Scope was documentation only (no code).
Original plan below, unmodified.*

*Originally: Plan only — nothing here is executed yet. The factual cross-doc
inconsistencies found in the 2026-07-03
verification sweep (embedding dims, base-URL, module layout, version pins,
boilerplate READMEs, stale-architecture banners) were **already fixed** in that
pass — what remains is **structural**: 20 doc files, ~8,000 lines, with several
topics described in three or four places at once.*

---

## 1. The problem, measured

| Doc | Lines | Role today | Overlaps with |
|---|---:|---|---|
| `PROJECT-REPORT.md` | 1258 | Master report (what/why/reflection) | ITERATION-PLAN, VIZ, PERSPECTIVES, PROCESS |
| `documentations/LEARN.md` | 1765 | Designer learning guide (whole codebase) | BACKEND, FRONTEND, CLAUDE ×2 — and partly stale |
| `documentations/DESIGN-SPACE-ITERATION-PLAN.md` | 1345 | Full critique→iteration history (Parts 1–13) | PROJECT-REPORT §5, PROCESS, ITERATION-M-PLAN |
| `documentations/DESIGN-SPACE-TESTING.md` | 528 | Test protocol (auto + manual walkthroughs) | the test files, PROCESS |
| `ITERATION-M-PLAN.md` | 384 | Next-iteration plan (engineering + research) | PROJECT-REPORT §5.7/§6, USER-TESTING-PLAN |
| `llmind-python/README.md` | 357 | Backend data-pipeline manual (CLI/env/layout) | BACKEND.md, llmind-python/CLAUDE.md |
| `llmind-python/BACKEND.md` | 312 | Backend API/arch/env reference | llmind-python/README.md, CLAUDE.md |
| `llmind-web/Mind-elixir.md` | 254 | **Vendored** mind-elixir library readme copy | node_modules/mind-elixir/readme.md |
| `documentations/DESIGN-SPACE-VIZ.md` | 232 | Original projection concept + M0–M3 build | BACKEND projection §, PROJECT-REPORT §2.4/§3 |
| `PROCESS.md` | 230 | Session handoff (state + local-stack rules) | ITERATION-PLAN, ITERATION-M-PLAN |
| `documentations/USER-FLOWS.md` | 172 | Ten study flows F0–F9 | USER-TESTING-PLAN, PROJECT-REPORT §6.2 |
| `documentations/DESIGN-SPACE-PERSPECTIVES-PLAN.md` | 168 | F1 lens + F2 axes rationale | FRONTEND axes rows, PROJECT-REPORT §2.6/2.8 |
| `documentations/USER-TESTING-PLAN.md` | 161 | Drafted study protocol | USER-FLOWS, ITERATION-M-PLAN Part II |
| `llmind-web/ZUSTAND.md` | 155 | Store shape/actions/persistence | FRONTEND store row |
| `documentations/PROJECT_DEV.md` | 142 | Early dev log (per-change justifications) | PROJECT-REPORT, ITERATION-PLAN |
| `llmind-python/CLAUDE.md` | 114 | Backend hub (env/pipeline/module map) | BACKEND.md, llmind-python/README.md |
| `CLAUDE.md` | 88 | Root hub (structure, quick start, doc table) | README.md |
| `llmind-web/REACT-QUERY.md` | 43 | Hooks reference | FRONTEND hooks rows |
| `README.md` | 33 | Root overview → pointers | CLAUDE.md |
| `llmind-web/README.md` | 34 | Frontend launcher → pointers | FRONTEND.md |
| `AGENTS.md` | 1 | Literally `CLAUDE.md` (agent-tool pointer) | — |

**The four redundancy clusters** (where a reader gets three answers to one question):

- **C1 — Backend triplication.** `llmind-python/{CLAUDE.md, README.md, BACKEND.md}`
  each carry a module map, env-var table, and pipeline/CLI list. 783 lines, ~⅓
  duplicated. A change to an env default must currently be made in three places
  (exactly how the 384-d / base-URL drift happened).
- **C2 — Iteration history in four voices.** The Parts 1–13 narrative
  (ITERATION-PLAN), its report-level retelling (PROJECT-REPORT §5), the
  session-handoff retelling (PROCESS), and the next-step plan (ITERATION-M-PLAN)
  overlap heavily and cross-cite each other by section number, which makes them
  brittle.
- **C3 — Study docs.** USER-FLOWS + USER-TESTING-PLAN + PROJECT-REPORT §6.2 +
  ITERATION-M-PLAN Part II all describe the (still-unrun) study.
- **C4 — LEARN.md.** A 1,765-line teaching doc that re-explains the pipeline,
  backend, and frontend from scratch — the single largest overlap surface, and
  the one most prone to going stale (its API-connection chapter already taught
  the reversed proxy architecture; now banner-flagged).

---

## 2. Target: one owner per topic

The fix is not "delete docs" — it is to declare a **single source of truth (SSOT)**
per topic and make every other mention a one-line pointer, not a parallel copy.

| Topic | SSOT (authoritative, kept current) | Everything else → |
|---|---|---|
| What the system is / research argument / reflection | `PROJECT-REPORT.md` | pointer |
| Backend API endpoints, env vars, architecture, CLIs | `llmind-python/BACKEND.md` | pointer |
| Backend data-pipeline HOW-TO (scrape→embed→cluster→taxonomy) | a `## Pipeline` section **inside BACKEND.md** | fold README here |
| Frontend architecture / components / flows / design language | `llmind-web/FRONTEND.md` | pointer |
| Store shape | `llmind-web/ZUSTAND.md` | pointer |
| Query/mutation hooks | `llmind-web/REACT-QUERY.md` | pointer |
| Local-stack operational rules (the hard-won LLM recipe) | `PROCESS.md` §2 | pointer |
| Iteration HISTORY (frozen record) | `documentations/DESIGN-SPACE-ITERATION-PLAN.md` | PROJECT-REPORT §5 keeps only the *report-level* synthesis + pointer |
| NEXT iteration plan | `ITERATION-M-PLAN.md` | supersedes PROCESS "next steps" |
| Study design | `documentations/USER-TESTING-PLAN.md` (protocol) + `USER-FLOWS.md` (task templates) | PROJECT-REPORT §6.2 + ITERATION-M-PLAN Part II point here |
| Onboarding / learning path | `documentations/LEARN.md` (slimmed, see §3.C4) | — |
| Doc index (what to read for what) | `CLAUDE.md` doc table | README points here |

---

## 3. Actions, by cluster

### C1 — Collapse the backend triplication (highest value, lowest risk)
- **Fold `llmind-python/README.md` into `BACKEND.md`.** Move the pipeline HOW-TO
  (scrape/analyze/ingest/cluster/farthest command reference + the end-to-end run)
  into a `## Data pipeline` section of BACKEND.md. Replace README.md with a ~15-line
  launcher (like the new `llmind-web/README.md`): run commands + a pointer to
  BACKEND.md and the root CLAUDE.md. Kills ~300 lines of duplicated CLI/env tables.
- **Trim `llmind-python/CLAUDE.md` to a true hub.** Keep: environment (uv), the
  one-line pipeline diagram, the OpenAI-structured-output rules, and a module-map
  *pointer* into BACKEND.md — remove the full module table and command list now
  duplicated in BACKEND.md. Keep the 768-d ⚠ box (it is the SSOT for the
  default-vs-deployed trap) OR move it to BACKEND.md's env table and point.
- **Net:** one env-var table, one module map, one CLI list. Result ≈ BACKEND.md
  (grows ~80 lines) + two thin hubs.

### C2 — De-duplicate the iteration narrative
- **Declare ITERATION-PLAN the frozen history** and PROJECT-REPORT §5 the
  *synthesis* — audit §5 for passages that merely re-narrate Parts 10–13 and
  replace them with a one-line "full record: ITERATION-PLAN Part N". (§5 already
  does this in places; make it consistent.)
- **PROCESS.md is working memory, not history.** Its numbered "reasoning chain"
  (§3) duplicates the ITERATION-PLAN. Cut §3 to a 3-line summary + pointer; keep
  §2 (local-stack rules — genuinely unique SSOT) and §4/§1 (current state).
- **Stop cross-citing by section number where avoidable** — the §-number web is
  the brittleness. Prefer named anchors ("the register-alignment round") over
  "§5.2" when a doc references another.

### C3 — Consolidate study docs
- Keep **USER-TESTING-PLAN.md** (protocol) and **USER-FLOWS.md** (task templates)
  as the study SSOT. Make **PROJECT-REPORT §6.2** and **ITERATION-M-PLAN Part II**
  reference them rather than re-describe the tasks. (ITERATION-M-PLAN Part II is
  the newest and adds the trust-delta/planted-dimension probes — fold those two
  probes *into* USER-TESTING-PLAN.md so the protocol is complete in one place, and
  leave Part II as the rationale/decision layer.)

### C4 — Slim and re-scope LEARN.md
- LEARN.md is valuable as an onboarding path but is the biggest staleness risk.
  Two options, decide deliberately:
  - **(a) Slim to a guided path (recommended):** keep the layer-by-layer *narrative
    and mental models*; replace every concrete API/module/endpoint listing with a
    pointer to BACKEND/FRONTEND (the SSOT). Cuts ~40% and removes the drift surface.
  - **(b) Freeze + banner:** leave it, treat as a historical teaching snapshot
    (banner already added 2026-07-03). Cheaper now, but it keeps rotting.

### Small items
- **`llmind-web/Mind-elixir.md`** is a copy of the library's own readme — **delete**
  (the API lives in `node_modules/mind-elixir/readme.md` and the library docs).
  If a project-specific integration note is wanted, replace with a ~10-line
  "how we wrap mind-elixir" note in FRONTEND.md instead.
- **`documentations/PROJECT_DEV.md`** (early dev log) is superseded by
  PROJECT-REPORT + ITERATION-PLAN — **mark superseded** with a banner (keep for
  provenance, per the archival policy) rather than delete.
- **`AGENTS.md`** — leave (it is the agent-tool pointer to CLAUDE.md).
- **Root `README.md` / `CLAUDE.md`** — already lean; just ensure the doc-index
  table in CLAUDE.md lists ITERATION-M-PLAN and DOC-CONSOLIDATION-PLAN (done for
  the former) and reflects any renames above.

---

## 4. The archival constraint (why "merge" ≠ "rewrite" for `documentations/`)

PROJECT-REPORT §7 states the `documentations/` files are **archived, unmodified in
content** — they are the project's historical record. So for archived docs (VIZ,
ITERATION-PLAN, PERSPECTIVES-PLAN, PROJECT_DEV, and LEARN if frozen), consolidation
means **superseding banners + pointers**, not editing the bodies. Only the *live*
reference docs (BACKEND, FRONTEND, ZUSTAND, REACT-QUERY, the CLAUDE hubs, PROCESS,
the two READMEs) are freely rewritable. This split should itself be stated once, at
the top of the CLAUDE.md doc table: **live vs archived**.

---

## 5. Suggested order (if/when executed)

1. **C1** (backend triplication) — mechanical, high value, zero research risk.
2. **Small items** (delete Mind-elixir.md, banner PROJECT_DEV.md).
3. **C3** (fold the two probes into USER-TESTING-PLAN; point §6.2 / Part II).
4. **C2** (trim PROCESS §3; make §5 pointers consistent).
5. **C4** — decide (a) vs (b) with the owner; only (a) is real work.
6. Add the **live-vs-archived** split + updated index to CLAUDE.md.

Estimated reduction: ~900–1,400 lines removed/de-duplicated (mostly C1 + C4),
with **one** authoritative place per topic — which is what stops the next
384-d-style drift at the source.

---

## 6. Open decisions for the owner
- **LEARN.md:** slim-to-path (a) or freeze-and-banner (b)?
- **`llmind-python/README.md`:** fold fully into BACKEND.md (recommended), or keep
  a fuller standalone pipeline manual?
- **Archival policy:** confirm the live-vs-archived split so C2/C4 know what is
  rewritable vs banner-only.
- Anything in `documentations/` you consider **live** (must-stay-current) rather
  than historical? (Currently only PROJECT-REPORT/PROCESS/subsystem docs are
  treated as live.)
