"""Aggregate the generate-at evaluation log — the drift A/B analysis.

``data/projection/generate_log.jsonl`` accumulates one row per generate-at call
(``prompt_version``, ``seed_strategy``, per-node drift and clipped flags). This
module turns it into per-variant statistics so prompt/seeding changes are judged
on data. Pure stdlib — no I/O — so it is unit-testable offline.
"""

from __future__ import annotations

from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple


def aggregate_annotation_cache(
    records: Iterable[Dict[str, Any]],
    too_broad_share: float = 0.8,
    unprecedented_max: int = 1,
) -> Dict[str, Any]:
    """Distribution summary over cached per-option annotation records.

    Each record is ``{count, project_ids, shortlist_k}`` (one option). Returns
    the count distribution, the mean shortlist-acceptance (``count / shortlist_k``
    — the aboutness→exemplification gap the report tracks, §5.6), and how many
    options hit the granularity flags. Pure stdlib — no I/O — so it can
    regenerate the report's figures deterministically from the current cache.

    Thresholds are passed in (rather than importing ``backend.corpus.annotate``)
    to keep this pipeline module free of a backend dependency; the CLI supplies
    the live values so the numbers match ``diagnostics_for``.
    """
    counts: List[int] = []
    shares: List[float] = []
    saturated = 0
    unprecedented = 0
    for r in records:
        count = int(r.get("count", 0))
        k = int(r.get("shortlist_k", 0))
        counts.append(count)
        if k > 0:
            shares.append(count / k)
            if count >= too_broad_share * k:
                saturated += 1
        if count <= unprecedented_max:
            unprecedented += 1
    n = len(counts)
    return {
        "n_options": n,
        "count_min": min(counts) if counts else 0,
        "count_median": float(median(counts)) if counts else 0.0,
        "count_max": max(counts) if counts else 0,
        "mean_shortlist_acceptance": round(mean(shares), 3) if shares else None,
        "saturated": saturated,
        "unprecedented": unprecedented,
    }


def aggregate_generate_log(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per (prompt_version, seed_strategy, register_aligned, brief_context,
    placement) variant: generation count, node count, drift mean/median over
    NON-clipped nodes, and the clipped rate.

    Clipped placements are extrapolations, not corpus-supported locations, so
    they are excluded from the drift aggregate and reported as a rate instead.
    Legacy rows predate the ``clipped`` flag; for those, a node pinned exactly
    to the surface edge (coordinate 0 or 1) counts as clipped. Rows that
    predate ``register_aligned`` / ``brief_context`` count as false; rows that
    predate ``placement`` (Part 11) group as "umap" — what they actually used.
    """
    groups: Dict[Tuple[Any, Any, bool, bool, str], Dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("prompt_version"),
            row.get("seed_strategy"),
            bool(row.get("register_aligned", False)),
            bool(row.get("brief_context", False)),
            str(row.get("placement") or "umap"),
        )
        group = groups.setdefault(
            key, {"generations": 0, "nodes": 0, "clipped": 0, "drifts": []}
        )
        group["generations"] += 1
        for node in row.get("nodes", []):
            drift = node.get("drift")
            if drift is None:
                continue
            clipped = node.get("clipped")
            if clipped is None:  # legacy-row fallback
                clipped = node.get("x") in (0.0, 1.0) or node.get("y") in (0.0, 1.0)
            group["nodes"] += 1
            if clipped:
                group["clipped"] += 1
            else:
                group["drifts"].append(float(drift))

    stats: List[Dict[str, Any]] = []
    for (prompt_version, seed_strategy, register_aligned, brief_context, placement), group in sorted(
        groups.items(),
        key=lambda item: (str(item[0][0]), str(item[0][1]), item[0][2], item[0][3], item[0][4]),
    ):
        drifts = group["drifts"]
        stats.append(
            {
                "prompt_version": prompt_version,
                "seed_strategy": seed_strategy,
                "register_aligned": register_aligned,
                "brief_context": brief_context,
                "placement": placement,
                "generations": group["generations"],
                "nodes": group["nodes"],
                "drift_mean": mean(drifts) if drifts else None,
                "drift_median": median(drifts) if drifts else None,
                "clipped_rate": (group["clipped"] / group["nodes"]) if group["nodes"] else None,
            }
        )
    return stats
