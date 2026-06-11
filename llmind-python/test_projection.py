#!/usr/bin/env python3
"""Self-contained test harness for the design-space projection.

No pytest dependency — plain asserts with a PASS/FAIL summary, matching the
project's script style. Run with:

    uv run python test_projection.py                 # offline tests only
    uv run python test_projection.py --http           # + GET /surface over HTTP
    uv run python test_projection.py --live            # + /locate, /generate-at (needs LM Studio + LLM)

Offline tests need only the fitted artifacts in ``data/projection/`` (build them
with ``uv run python database_pipeline.py project``). The ``--live`` tests need
the embedding/LLM server configured in ``.env`` to be running.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

from config import settings
from pipeline import projection as proj

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    mark = "PASS" if condition else "FAIL"
    if condition:
        PASSED += 1
    else:
        FAILED += 1
    line = f"[{mark}] {name}"
    if detail:
        line += f"  — {detail}"
    print(line)


# ── Offline: pure projection math ─────────────────────────────────────────────


def test_fit_transform_unit_synthetic() -> None:
    """Fit on synthetic clusters; transform must be deterministic and in [0,1]."""
    rng = np.random.RandomState(0)
    a = rng.normal(0, 1, (40, 32)) + 5
    b = rng.normal(0, 1, (40, 32)) - 5
    X = np.vstack([a, b])

    model = proj.fit_projection(X, dims=2, pre_pca=16, n_neighbors=10)
    c1 = model.transform(X)
    c2 = model.transform(X)

    check("synthetic: transform deterministic", np.allclose(c1, c2))
    check("synthetic: coords within [0,1]", bool((c1 >= 0).all() and (c1 <= 1).all()))
    check(
        "synthetic: two clusters separate in 2D",
        float(np.linalg.norm(c1[:40].mean(0) - c1[40:].mean(0))) > 0.2,
        f"centroid gap={np.linalg.norm(c1[:40].mean(0) - c1[40:].mean(0)):.3f}",
    )

    # An out-of-distribution point lands within the soft-clip band at worst.
    oob = model.transform(rng.normal(0, 1, (1, 32)) * 50)
    m = proj.SOFT_MARGIN
    check("synthetic: OOD point within the soft-clip band",
          bool((oob >= -m).all() and (oob <= 1 + m).all()))


def test_grid_helpers() -> None:
    res = 48
    check("grid: corner (0,0) → cell (0,0)", proj.to_cell(0.0, 0.0, res) == (0, 0))
    check("grid: corner (1,1) → cell (R-1,R-1)", proj.to_cell(1.0, 1.0, res) == (res - 1, res - 1))
    gx, gy = proj.to_cell(0.5, 0.5, res)
    cxv, cyv = proj.cell_center(gx, gy, res)
    check("grid: cell_center round-trips into same cell", proj.to_cell(cxv, cyv, res) == (gx, gy))

    pts = np.array([[0.1, 0.1], [0.9, 0.9], [0.5, 0.5]])
    dens = proj.density_grid(pts, res)
    check("grid: density grid is R×R", len(dens) == res and len(dens[0]) == res)
    check("grid: density counts all points", sum(sum(r) for r in dens) == 3)


def test_nearest() -> None:
    pts = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.51, 0.49]])
    idx = proj.nearest_indices(pts, [0.5, 0.5], 2)
    check("nearest: returns the two closest", set(idx) == {2, 3}, f"got {idx}")
    check("nearest: empty input safe", proj.nearest_indices(np.empty((0, 2)), [0.5, 0.5], 3) == [])


def test_surface_payload() -> None:
    ids = ["a", "b", "c"]
    coords = np.array([[0.1, 0.2], [0.9, 0.8], [0.5, 0.5]])
    payload = proj.build_surface_payload(
        ids=ids, coords=coords, dims=2, resolution=16, clusters=[0, 1, 0]
    )
    check("payload: one point per id", len(payload["points"]) == len(ids))
    check("payload: density present and R×R", len(payload["density"]) == 16)
    p0 = payload["points"][0]
    check("payload: point has x/y/cell/cluster", all(k in p0 for k in ("x", "y", "cell", "cluster")))


# ── Offline: persisted artifacts + stability invariant ────────────────────────


def test_persisted_artifacts() -> None:
    model_path = settings.projection_dir / proj.MODEL_FILENAME
    surface_path = settings.projection_dir / proj.SURFACE_FILENAME
    if not model_path.exists() or not surface_path.exists():
        check("artifacts: model.joblib + surface.json exist", False,
              "run `uv run python database_pipeline.py project` first")
        return
    check("artifacts: model.joblib + surface.json exist", True)

    model = proj.load_model(settings.projection_dir)
    surface = json.loads(surface_path.read_text(encoding="utf-8"))

    # Stability: reloading + re-transforming the corpus reproduces persisted coords.
    index_path = settings.local_index_path
    if index_path.exists():
        data = np.load(index_path, allow_pickle=True)
        X = data["vectors"]
        recomputed = model.transform(X)
        persisted = np.array([[p["x"], p["y"]] for p in surface["points"]])
        check(
            "stability: reloaded transform == persisted surface",
            recomputed.shape == persisted.shape and np.allclose(recomputed, persisted, atol=1e-5),
        )
        check(
            "stability: input dim matches model meta",
            X.shape[1] == model.meta.get("input_dims"),
            f"index={X.shape[1]} meta={model.meta.get('input_dims')}",
        )

    check("artifacts: surface dims in {2,3}", surface["dims"] in (2, 3))
    check("artifacts: empty cells exist (generation affordances)",
          sum(1 for r in surface["density"] for c in r if c == 0) > 0)


def test_service_offline() -> None:
    from backend.projection import service

    surface = service.load_surface()
    check("service: load_surface returns points", len(surface.get("points", [])) > 0)

    pts = surface["points"]
    x, y = pts[0]["x"], pts[0]["y"]
    nb = service.nearest_corpus(x, y, 3)
    check("service: nearest_corpus returns ≤k neighbours with metadata",
          0 < len(nb) <= 3 and "Name" in nb[0])
    check("service: nearest_corpus closest is the query point itself",
          nb[0]["id"] == str(pts[0]["id"]))


def test_corpus_service() -> None:
    from backend.corpus import service as corpus

    meta = corpus.load_index_meta()
    if not meta:
        check("corpus: metadata sidecar present", False, "run build_local_index.py first")
        return
    check("corpus: metadata sidecar present", True)

    pid = next(iter(meta))
    record = corpus.get_project(pid)
    check("corpus: get_project returns normalised record",
          record["id"] == pid and "Name" in record and "Descriptions" in record)
    try:
        corpus.get_project("__not_a_project__")
        check("corpus: unknown id raises KeyError", False)
    except KeyError:
        check("corpus: unknown id raises KeyError", True)


def test_clipped_detection() -> None:
    """Out-of-bounds projections are flagged and soft-clipped into the margin
    band — direction and ordering preserved, never silently pinned."""
    m = proj.SOFT_MARGIN
    model = proj.ProjectionModel(
        reducer=None, pca=None, bounds=[(0.0, 10.0), (0.0, 10.0)], dims=2
    )
    coords, clipped = model._normalize(
        np.array([[5.0, 5.0], [-2.0, 4.0], [3.0, 12.0], [-9.0, 4.0]])
    )
    check("clipped: in-bounds point not flagged, identity preserved",
          not clipped[0] and coords[0][0] == 0.5)
    check("clipped: out-of-bounds points flagged",
          bool(clipped[1]) and bool(clipped[2]) and bool(clipped[3]))
    check("clipped: overshoot lands INSIDE the band, not on the edge",
          -m < coords[1][0] < 0.0 and 1.0 < coords[2][1] < 1.0 + m,
          f"x={coords[1][0]:.4f}, y={coords[2][1]:.4f}")
    check("clipped: ordering among out-of-hull points survives (-9 < -2 side)",
          coords[3][0] < coords[1][0])
    # The squash is bounded: even an extreme point stays within the band.
    far, _ = model._normalize(np.array([[1e6, -1e6]]))
    check("clipped: extreme point asymptotes at the band edge",
          1.0 < far[0][0] <= 1.0 + m and -m <= far[0][1] < 0.0)
    # Boundary values are NOT out-of-hull.
    edge, edge_clip = model._normalize(np.array([[0.0, 10.0]]))
    check("clipped: exact bounds count as in-hull",
          not edge_clip[0] and edge[0][0] == 0.0 and edge[0][1] == 1.0)


def test_place_by_neighbors() -> None:
    """Evidence-anchored out-of-sample placement (Part 11): queries land at the
    similarity-weighted centroid of their top-k corpus anchors' coordinates."""
    corpus = np.eye(3)  # three orthogonal unit vectors
    coords = np.array([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]])

    placed = proj.place_by_neighbors(np.eye(3)[:1], corpus, coords, k=1)
    check("place: exact corpus match at k=1 lands on its anchor",
          np.allclose(placed[0], coords[0]))

    q = np.array([[1.0, 1.0, 0.0]]) / np.sqrt(2)
    placed = proj.place_by_neighbors(q, corpus, coords, k=2)
    check("place: equal-similarity anchors → midpoint",
          np.allclose(placed[0], coords[:2].mean(axis=0)))

    # k beyond the corpus clamps; the zero-similarity anchor gets zero weight.
    placed = proj.place_by_neighbors(q, corpus, coords, k=10)
    check("place: k clamps to corpus size, zero-sim anchor has no pull",
          np.allclose(placed[0], coords[:2].mean(axis=0)))

    # Self-exclusion (fit-time diagnostics): the perfect match is ignored.
    placed = proj.place_by_neighbors(np.eye(3)[:1], corpus, coords, k=1, exclude_rows=[0])
    check("place: self-exclusion ignores the perfect match",
          not np.allclose(placed[0], coords[0]))

    # All-non-positive similarities degrade to a uniform centroid, never NaN.
    placed = proj.place_by_neighbors(-np.ones((1, 3)) / np.sqrt(3), corpus, coords, k=3)
    check("place: non-positive sims fall back to uniform centroid",
          np.allclose(placed[0], coords.mean(axis=0)))

    # Convexity: placements can never leave the corpus footprint.
    rng = np.random.default_rng(0)
    qs = rng.normal(size=(5, 3))
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    placed = proj.place_by_neighbors(qs, corpus, coords, k=3)
    inside = all(
        coords[:, a].min() - 1e-12 <= placed[:, a].min()
        and placed[:, a].max() <= coords[:, a].max() + 1e-12
        for a in range(2)
    )
    check("place: placements stay inside the corpus footprint", inside)


def test_log_stats() -> None:
    from pipeline.log_stats import aggregate_generate_log

    rows = [
        {  # current-format row: one clean node, one clipped node
            "prompt_version": 2, "seed_strategy": "bracket",
            "nodes": [
                {"drift": 0.1, "clipped": False, "x": 0.4, "y": 0.4},
                {"drift": 0.5, "clipped": True, "x": 0.0, "y": 0.2},
            ],
        },
        {  # legacy row (no clipped flag): edge-pinned node counts as clipped
            "prompt_version": 2, "seed_strategy": "bracket",
            "nodes": [
                {"drift": 0.3, "x": 0.6, "y": 0.5},
                {"drift": 0.9, "x": 1.0, "y": 0.5},
            ],
        },
        {"prompt_version": 1, "seed_strategy": "anchor", "nodes": [{"drift": 0.2, "clipped": False}]},
        {  # register-aligned rows form their OWN variant group
            "prompt_version": 2, "seed_strategy": "bracket", "register_aligned": True,
            "nodes": [{"drift": 0.4, "clipped": False, "x": 0.3, "y": 0.3}],
        },
        {  # brief-conditioned rows form their OWN variant group too (Part 10)
            "prompt_version": 4, "seed_strategy": "bracket", "register_aligned": True,
            "brief_context": True,
            "nodes": [{"drift": 0.25, "clipped": False, "x": 0.3, "y": 0.3}],
        },
        {  # placement regimes never aggregate together (Part 11)
            "prompt_version": 4, "seed_strategy": "bracket", "register_aligned": True,
            "brief_context": True, "placement": "knn",
            "nodes": [{"drift": 0.15, "clipped": False, "x": 0.3, "y": 0.3}],
        },
    ]
    stats = aggregate_generate_log(rows)
    check("log-stats: one group per variant (incl. aligned/brief/placement)", len(stats) == 5)
    bracket = next(
        s for s in stats if s["seed_strategy"] == "bracket" and not s["register_aligned"]
    )
    check("log-stats: bracket counts gens/nodes", bracket["generations"] == 2 and bracket["nodes"] == 4)
    check("log-stats: drift over non-clipped only (0.1, 0.3)",
          abs(bracket["drift_mean"] - 0.2) < 1e-9, f"mean={bracket['drift_mean']}")
    check("log-stats: clipped rate counts legacy edge-pin", bracket["clipped_rate"] == 0.5)
    aligned = next(s for s in stats if s["register_aligned"] and not s["brief_context"])
    check("log-stats: aligned group aggregates separately",
          aligned["nodes"] == 1 and abs(aligned["drift_mean"] - 0.4) < 1e-9)
    briefed = next(s for s in stats if s["brief_context"] and s["placement"] == "umap")
    check("log-stats: brief-context group aggregates separately",
          briefed["nodes"] == 1 and abs(briefed["drift_mean"] - 0.25) < 1e-9)
    knn = next(s for s in stats if s["placement"] == "knn")
    check("log-stats: placement regimes aggregate separately (legacy rows = umap)",
          knn["nodes"] == 1 and abs(knn["drift_mean"] - 0.15) < 1e-9
          and all(s["placement"] == "umap" for s in stats if s is not knn))


def test_candidate_alignment() -> None:
    """Alignment scoring: agreement + per-aspect lean, on synthetic vectors."""
    from backend.candidates.service import render_draft_brief_prompt, score_alignment

    prompt = render_draft_brief_prompt(
        [{"aspect": "Display", "option": "LED mesh", "desc": "low-res facade grid"},
         {"aspect": "Input", "option": "Gesture"}],
        "A riverside pavilion",
    )
    check("alignment: draft prompt embeds choices and overview",
          "Display: LED mesh — low-res facade grid" in prompt
          and "- Input: Gesture" in prompt
          and "A riverside pavilion" in prompt)
    check("alignment: empty overview reads as none provided",
          "(none provided)" in render_draft_brief_prompt([{"aspect": "A", "option": "B"}], " "))

    # Orthonormal basis: brief along e0, chosen along e0 (agrees), alternative e1.
    e = np.eye(4)
    agree = score_alignment(
        e[0], e[0],
        [{"aspect_id": "a1", "chosen": {"id": "c", "vec": e[0]},
          "alternatives": [{"id": "alt", "vec": e[1]}]}],
    )
    check("alignment: brief expressing the choice does not lean away",
          agree["agreement"] == 1.0
          and not agree["per_aspect"][0]["leans_away"]
          and agree["per_aspect"][0]["chosen_score"] == 1.0)

    # Brief midway but closer to the alternative → leans_away + correct top pick.
    brief = (0.4 * e[0] + 0.9 * e[1])
    brief = brief / np.linalg.norm(brief)
    lean = score_alignment(
        brief, e[2],
        [{"aspect_id": "a1", "chosen": {"id": "c", "vec": e[0]},
          "alternatives": [{"id": "weak", "vec": e[3]}, {"id": "strong", "vec": e[1]}]}],
    )
    row = lean["per_aspect"][0]
    check("alignment: top alternative picked by data, lean detected",
          row["leans_away"] and row["top_alternative"]["id"] == "strong"
          and row["top_alternative"]["score"] > row["chosen_score"])
    check("alignment: agreement is cos(brief, composition)",
          abs(lean["agreement"]) < 1e-9)
    check("alignment: no alternatives → no lean",
          not score_alignment(e[0], e[0], [
              {"aspect_id": "a1", "chosen": {"id": "c", "vec": e[0]}, "alternatives": []}
          ])["per_aspect"][0]["leans_away"])


def test_compute_metrics_offline() -> None:
    """/metrics scoring against the real corpus, with a stubbed embedder."""
    from backend.corpus.service import load_corpus_vectors
    from backend.projection import service

    ids, vecs = load_corpus_vectors()
    if not ids:
        check("metrics: corpus available", False, "local index missing")
        return

    # Poles = two real corpus vectors; item = exactly the pole_a project.
    fake = np.vstack([vecs[0], vecs[1], vecs[0]])
    original = service._embed_texts
    service._embed_texts = lambda texts: fake[: len(texts)]
    try:
        result = service.compute_metrics(
            metrics=[{"pole_a": "a", "pole_b": "b"}],
            items=[{"node_id": "n1", "text": "x"}],
        )
    finally:
        service._embed_texts = original

    metric = result["metrics"][0]
    scores = np.array(metric["corpus"])
    check("metrics: corpus scores span [-1, 1] exactly",
          abs(scores.min() + 1) < 1e-9 and abs(scores.max() - 1) < 1e-9)
    # The item IS the pole_a project — it must score at the pole_a extreme. It
    # may legitimately exceed the corpus max (clip-flagged, capped at 1.0).
    item = metric["items"][0]
    check("metrics: item identical to pole_a scores at the pole_a end",
          item["score"] >= 0.99 and (not item["clipped"] or item["score"] == 1.0),
          f"score={item['score']:.3f} clipped={item['clipped']}")
    check("metrics: pole similarity reported",
          0.0 < metric["pole_sim"] < 1.0, f"pole_sim={metric['pole_sim']:.3f}")
    check("metrics: correlation matrix has unit diagonal",
          result["corr"] == [[1.0]])


def test_prompts_v4() -> None:
    from utils.prompts import (
        DRAFT_BRIEF_PROMPT,
        GENERATE_AT_PROMPT,
        GENERATE_AT_PROMPT_VERSION,
    )

    check("prompts: generate-at v4 carries the designer-brief block",
          GENERATE_AT_PROMPT_VERSION == 4 and "{{DESIGNER_BRIEF}}" in GENERATE_AT_PROMPT)
    check("prompts: draft-brief asks for project-register prose",
          "3-5 sentences" in DRAFT_BRIEF_PROMPT and "{{CHOICES}}" in DRAFT_BRIEF_PROMPT)


def test_register_alignment() -> None:
    """The short→long correction must beat the raw baseline on held-out pairs."""
    import tempfile

    from pipeline import register_alignment as ra

    check("align: short text = name + first sentences",
          ra.build_short_text("Aurora", "A facade. It glows. More detail here.", sentences=2)
          == "Aurora. A facade. It glows.")
    check("align: short text caps length",
          len(ra.build_short_text("N", "x" * 999, max_chars=50)) <= 50)
    check("align: empty description falls back to the name",
          ra.build_short_text("Aurora", "  ") == "Aurora")

    # Synthetic register gap: short = long + systematic offset + noise.
    rng = np.random.default_rng(7)
    long_vecs = rng.normal(size=(60, 16))
    long_vecs /= np.linalg.norm(long_vecs, axis=1, keepdims=True)
    offset = rng.normal(size=16) * 0.5
    short_vecs = long_vecs + offset + rng.normal(size=(60, 16)) * 0.05

    rmap, report = ra.fit_register_map(short_vecs, long_vecs, folds=5)
    check("align: CV beats the raw baseline",
          report["winner"]["cv_cosine"] > report["baseline_cosine"] + 0.05,
          f"{report['baseline_cosine']:.3f} → {report['winner']['cv_cosine']:.3f}")
    mapped = rmap.apply(short_vecs)
    check("align: apply returns unit vectors",
          bool(np.allclose(np.linalg.norm(mapped, axis=1), 1.0)))
    check("align: oof predictions cover every pair",
          report["oof_mapped"].shape == short_vecs.shape)

    with tempfile.TemporaryDirectory() as tmp:
        rmap.support_baseline = np.array([0.4, 0.5, 0.6])
        ra.save_register_map(rmap, Path(tmp))
        loaded = ra.load_register_map(Path(tmp))
        check("align: persistence round-trips (incl. support baseline)",
              loaded is not None
              and np.allclose(loaded.weights, rmap.weights)
              and np.allclose(loaded.intercept, rmap.intercept)
              and loaded.meta == rmap.meta
              and loaded.support_baseline is not None
              and np.allclose(loaded.support_baseline, rmap.support_baseline))
        # Pre-recalibration artifact (no baseline key) still loads.
        np.savez(
            Path(tmp) / ra.REGISTER_MAP_FILENAME,
            weights=rmap.weights, intercept=rmap.intercept, meta="{}",
        )
        legacy = ra.load_register_map(Path(tmp))
        check("align: legacy artifact loads with no baseline",
              legacy is not None and legacy.support_baseline is None)
    check("align: load returns None when absent",
          ra.load_register_map(Path("__nonexistent__")) is None)


def test_corpus_support() -> None:
    """Support percentile: corpus members score high, random noise scores ~0."""
    from backend.corpus import service as corpus

    baseline = np.array([0.2, 0.4, 0.6, 0.8])
    pct = corpus.support_percentiles(baseline, np.array([0.1, 0.5, 0.9]))
    check("support: percentile below/mid/above", list(pct) == [0.0, 0.5, 1.0])
    check("support: empty baseline → NaN",
          bool(np.isnan(corpus.support_percentiles(np.empty(0), np.array([0.5]))).all()))

    # support_scores: mean top-k cosine, with fit-time self-exclusion.
    e = np.eye(4)
    scores = corpus.support_scores(e[:1], e, k=2)
    check("support: mean top-k cosine (self in corpus → 1 and 0)",
          abs(scores[0] - 0.5) < 1e-9, f"{scores[0]:.3f}")
    excluded = corpus.support_scores(e[:1], e, exclude_rows=[0], k=2)
    check("support: self-exclusion removes the perfect match",
          abs(excluded[0] - 0.0) < 1e-9, f"{excluded[0]:.3f}")

    ids, vecs = corpus.load_corpus_vectors()
    if not ids:
        check("support: corpus available", False, "local index missing")
        return
    member = corpus.corpus_support(vecs[:3])
    check("support: corpus members score high (self in top-k)",
          all(s is not None and s > 0.5 for s in member), f"{member}")
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(2, vecs.shape[1]))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    off = corpus.corpus_support(noise)
    check("support: random noise scores ~0",
          all(s is not None and s < 0.05 for s in off), f"{off}")
    check("support: dim mismatch is best-effort None",
          corpus.corpus_support(np.ones((1, 3))) == [None])
    # An explicit (short-register) baseline changes the yardstick: a score
    # below the harsh full-register floor can still rank mid-pack.
    lenient = corpus.corpus_support(vecs[:1], baseline=np.array([0.1, 0.2, 0.3]))
    check("support: explicit baseline overrides the full-register yardstick",
          lenient[0] == 1.0, f"{lenient}")


def test_peek_offline() -> None:
    """The gap preview needs no LLM and no embedding server."""
    from backend.projection import service

    taxonomy = [
        {"id": "root", "topic": "Design Aspects", "isroot": True},
        {"id": "a1", "topic": "Display", "parentid": "root"},
        {"id": "o1", "topic": "LED", "parentid": "a1"},
    ]
    coords = [{"node_id": "o1", "x": 0.52, "y": 0.48}]
    result = service.peek(x=0.5, y=0.5, k=4, taxonomy_nodes=taxonomy, node_coords=coords)
    check("peek: returns k seeds with names",
          len(result["seeds"]) == 4 and all(s.get("Name") for s in result["seeds"]))
    check("peek: nearby explored ideas include the close option",
          "LED" in result["nearby_options"])
    check("peek: parent aspect derived from the click",
          result["parent_aspect"] == "Display", f"got {result['parent_aspect']}")
    # Determinism: the preview must show exactly what generate-at would use.
    again = service.peek(x=0.5, y=0.5, k=4, taxonomy_nodes=taxonomy, node_coords=coords)
    check("peek: deterministic seed set",
          [s["id"] for s in result["seeds"]] == [s["id"] for s in again["seeds"]])


def test_generate_at_helpers() -> None:
    from backend.projection import service

    taxonomy = [
        {"id": "root", "topic": "Design Aspects", "isroot": True},
        {"id": "a1", "topic": "Display", "parentid": "root"},
        {"id": "a2", "topic": "Interaction", "parentid": "root"},
        {"id": "o1", "topic": "LED", "parentid": "a1"},
        {"id": "o2", "topic": "Projection", "parentid": "a1"},
        {"id": "o3", "topic": "Gesture", "parentid": "a2"},
    ]
    coords = [
        {"node_id": "o1", "x": 0.1, "y": 0.1},
        {"node_id": "o2", "x": 0.2, "y": 0.1},
        {"node_id": "o3", "x": 0.8, "y": 0.9},
    ]
    near_display = service._derive_parent_aspect(taxonomy, coords, 0.15, 0.12)
    near_interaction = service._derive_parent_aspect(taxonomy, coords, 0.85, 0.85)
    check("generate-at: parent aspect derived spatially (display side)",
          near_display is not None and near_display["id"] == "a1",
          f"got {near_display}")
    check("generate-at: parent aspect derived spatially (interaction side)",
          near_interaction is not None and near_interaction["id"] == "a2",
          f"got {near_interaction}")
    check("generate-at: no coords → no derived parent (fallback to caller focus)",
          service._derive_parent_aspect(taxonomy, [], 0.5, 0.5) is None)

    nearby = service._format_nearby_options(taxonomy, coords, 0.15, 0.12)
    check("generate-at: nearby options include close, exclude far",
          "LED" in nearby and "Gesture" not in nearby, nearby.replace("\n", " | "))
    check("generate-at: empty region reads as unexplored",
          "unexplored" in service._format_nearby_options(taxonomy, coords, 0.5, 0.6))

    seeds = service.seed_corpus(0.5, 0.5, 5)
    seed_ids = [s["id"] for s in seeds]
    check("generate-at: bracket seeding returns k unique projects with metadata",
          len(seeds) == 5 and len(set(seed_ids)) == 5 and all("Name" in s for s in seeds))
    # Bracketing property: the seed set should not be a single tight cluster —
    # at least two seeds further apart (in 2D) than each is from the click.
    import math as _math
    pts = [(s["x"], s["y"]) for s in seeds]
    spread = max(
        _math.hypot(ax - bx, ay - by) for ax, ay in pts for bx, by in pts
    )
    nearest = min(_math.hypot(px - 0.5, py - 0.5) for px, py in pts)
    check("generate-at: seeds bracket the click (spread > nearest distance)",
          spread > nearest, f"spread={spread:.3f} nearest={nearest:.3f}")


def test_fidelity_metrics() -> None:
    from backend.projection import service

    model = proj.load_model(settings.projection_dir)
    trust = model.meta.get("trustworthiness")
    check("fidelity: trustworthiness in model meta and in [0,1]",
          trust is not None and 0.0 <= trust <= 1.0, f"trust={trust}")

    surface = service.load_surface()
    check("fidelity: trustworthiness served in surface meta",
          "trustworthiness" in surface.get("meta", {}))

    # Self-consistency: scoring corpus vectors at their own true coordinates must
    # yield non-trivial confidence (each point's 2D neighbourhood IS roughly its
    # true neighbourhood — modulo the projection's own distortion).
    ids, vecs = service._load_corpus_vectors()
    if ids:
        pts = {str(p["id"]): (p["x"], p["y"]) for p in surface["points"]}
        sample_ids = [i for i in ids[:8] if i in pts]
        sample_vecs = vecs[: len(sample_ids)]
        coords = np.array([pts[i] for i in sample_ids])
        confs = service._placement_confidence(sample_vecs, coords)
        scored = [c for c in confs if c is not None]
        mean_conf = sum(scored) / len(scored) if scored else 0.0
        check("fidelity: corpus self-confidence is non-trivial",
              len(scored) == len(confs) and mean_conf > 0.15,
              f"mean={mean_conf:.2f}")


# ── HTTP smoke (server running, no embedding server needed) ───────────────────


def _http_get(path: str) -> tuple[int, dict]:
    url = f"http://127.0.0.1:8000{path}"
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 (localhost)
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _http_post(path: str, body: dict) -> tuple[int, dict]:
    url = f"http://127.0.0.1:8000{path}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_http_surface() -> None:
    try:
        status, body = _http_get("/api/projection/surface")
    except Exception as exc:  # noqa: BLE001
        check("http: GET /surface", False, f"server not reachable: {exc}")
        return
    check("http: GET /surface → 200", status == 200)
    check("http: /surface has points + grid", "points" in body and "grid" in body)


def test_live_locate_and_generate() -> None:
    """Requires the embedding (and, for generate-at, the LLM) server to be up."""
    try:
        status, body = _http_post(
            "/api/projection/locate",
            {"items": [
                {"node_id": "t1", "text": "interactive media facade with responsive LED lighting"},
                {"node_id": "t2", "text": "participatory community urban design workshop"},
            ]},
        )
    except Exception as exc:  # noqa: BLE001
        check("live: POST /locate", False, f"server/embedding unavailable: {exc}")
        return

    if status != 200:
        check("live: POST /locate → 200", False, f"status={status} detail={body.get('detail')}")
        return
    pts = body.get("points", [])
    check("live: /locate returns coords for both nodes", len(pts) == 2)
    check("live: /locate coords in [0,1]",
          all(0 <= p["x"] <= 1 and 0 <= p["y"] <= 1 for p in pts))
    # Two semantically different prompts should not collapse to the same point.
    if len(pts) == 2:
        gap = ((pts[0]["x"] - pts[1]["x"]) ** 2 + (pts[0]["y"] - pts[1]["y"]) ** 2) ** 0.5
        check("live: distinct prompts → distinct locations", gap > 1e-3, f"gap={gap:.4f}")

    # generate-at near the centre of the surface.
    try:
        status, body = _http_post(
            "/api/projection/generate-at",
            {
                "x": 0.5, "y": 0.5,
                "focus_node_id": "design-aspects", "focus_node_topic": "Design Aspects",
                "taxonomy_nodes": [{"id": "design-aspects", "topic": "Design Aspects", "isroot": True}],
                "lineage": ["Design Aspects"], "k": 5, "mode": "vllm",
            },
        )
    except Exception as exc:  # noqa: BLE001
        check("live: POST /generate-at", False, f"LLM unavailable: {exc}")
        return
    check("live: /generate-at → 200", status == 200, f"detail={body.get('detail')}")
    if status == 200:
        check("live: /generate-at returns nodes", len(body.get("nodes", [])) > 0)
        check("live: /generate-at returns seed neighbours", len(body.get("seed_neighbours", [])) > 0)
        check("live: generated nodes carry coordinates",
              all("x" in n and "y" in n for n in body.get("nodes", [])))


def main() -> int:
    # Windows consoles default to cp1252 and choke on non-ASCII output.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(description="Design-space projection tests.")
    parser.add_argument("--http", action="store_true", help="Also test GET /surface over HTTP.")
    parser.add_argument("--live", action="store_true", help="Also test /locate + /generate-at (needs servers).")
    args = parser.parse_args()

    print("== Offline: projection math ==")
    test_fit_transform_unit_synthetic()
    test_grid_helpers()
    test_nearest()
    test_surface_payload()

    print("\n== Offline: persisted artifacts + service ==")
    test_persisted_artifacts()
    test_service_offline()
    test_corpus_service()
    test_clipped_detection()
    test_place_by_neighbors()
    test_log_stats()
    test_register_alignment()
    test_corpus_support()
    test_candidate_alignment()
    test_compute_metrics_offline()
    test_prompts_v4()
    test_peek_offline()
    test_generate_at_helpers()
    test_fidelity_metrics()

    if args.http or args.live:
        print("\n== HTTP smoke ==")
        test_http_surface()

    if args.live:
        print("\n== Live (embedding + LLM) ==")
        test_live_locate_and_generate()

    print(f"\n{'='*40}\nPASSED {PASSED}  FAILED {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
