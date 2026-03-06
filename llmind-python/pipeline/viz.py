"""Cluster visualisation using matplotlib."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


def plot_clusters(points: List[Dict[str, Any]], save_path: Optional[Path] = None) -> None:
    if plt is None:
        sys.stderr.write("matplotlib not installed, skipping plot.\n")
        return

    import matplotlib
    from collections import defaultdict

    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for p in points:
        groups[p["cluster"]].append(p)

    sorted_labels = sorted(groups.keys())
    cmap = matplotlib.colormaps["tab10"].resampled(len(sorted_labels))

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, label in enumerate(sorted_labels):
        pts = groups[label]
        ax.scatter(
            [p["x"] for p in pts],
            [p["y"] for p in pts],
            color=cmap(i),
            alpha=0.7,
            s=50,
            label=f"Cluster {label}",
        )

    ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)
    ax.set_title("Embeddings Clusters")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
