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

    # An out-of-distribution point still lands inside the surface (clipping).
    oob = model.transform(rng.normal(0, 1, (1, 32)) * 50)
    check("synthetic: OOD point clipped to [0,1]", bool((oob >= 0).all() and (oob <= 1).all()))


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
