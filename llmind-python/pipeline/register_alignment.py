"""Short→long register alignment for locating short texts in the frozen space.

The design-space projection is fit on full project descriptions (~2.7k chars),
but ``/locate`` inputs are short "Topic. desc" strings. Short and long texts of
the SAME concept embed in systematically different regions of embedding space —
the *register gap* — which surfaces as placement displacement and edge clipping
(ITERATION-PLAN Part 9, diagnostics #3/#4).

The corpus itself provides paired examples of every project in both registers
(its name + first description sentences vs the full indexed text), so a
regularised affine map short→long can be learned from those pairs and applied
to every ``/locate`` embedding *before* the frozen transform. The projection and
all corpus coordinates stay untouched.

Pure numpy — no I/O besides save/load of the fitted artifact
(``data/projection/register_map.npz``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REGISTER_MAP_FILENAME = "register_map.npz"
# Log-spaced grid sized to the spectrum of a unit-vector Gram matrix at n≈200;
# cross-validation picks, the translation candidate is the safe floor.
DEFAULT_ALPHAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
DEFAULT_FOLDS = 5
DEFAULT_SHORT_SENTENCES = 2
DEFAULT_SHORT_MAX_CHARS = 300

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class RegisterMap:
    """A fitted affine correction ``x @ weights + intercept`` (re-normalised).

    ``support_baseline`` is the short-register support yardstick fitted from
    the same pairs (sorted mean top-k cosines of the OUT-OF-FOLD corrected
    short texts to the corpus, self-excluded): the distribution a node-length
    query's support is read against ("as much corpus evidence as a real
    project described at this length" — Part 10 recalibration).
    """

    weights: np.ndarray   # (d, d)
    intercept: np.ndarray  # (d,)
    meta: Dict[str, Any] = field(default_factory=dict)
    support_baseline: Optional[np.ndarray] = None  # sorted, (n_pairs,)

    def apply(self, X: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        out = arr @ self.weights + self.intercept
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def build_short_text(
    name: str,
    description: str,
    *,
    sentences: int = DEFAULT_SHORT_SENTENCES,
    max_chars: int = DEFAULT_SHORT_MAX_CHARS,
) -> str:
    """Short-register exemplar mimicking node locate text ("Name. first sentences")."""
    name = (name or "").strip()
    desc = (description or "").strip()
    if desc:
        short_desc = " ".join(_SENTENCE_SPLIT.split(desc)[: max(1, sentences)]).strip()
        text = f"{name}. {short_desc}" if name else short_desc
    else:
        text = name
    return text[:max_chars].strip()


def _unit(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float((_unit(a) * _unit(b)).sum(axis=1).mean())


def _fit_translation(short: np.ndarray, long: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    d = short.shape[1]
    return np.eye(d), (long - short).mean(axis=0)


def _fit_ridge(
    short: np.ndarray, long: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form multi-output ridge with intercept (via centering)."""
    s_mean = short.mean(axis=0)
    l_mean = long.mean(axis=0)
    sc = short - s_mean
    lc = long - l_mean
    d = short.shape[1]
    weights = np.linalg.solve(sc.T @ sc + alpha * np.eye(d), sc.T @ lc)
    return weights, l_mean - s_mean @ weights


def fit_register_map(
    short: np.ndarray,
    long: np.ndarray,
    *,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    folds: int = DEFAULT_FOLDS,
    seed: int = 42,
) -> Tuple[RegisterMap, Dict[str, Any]]:
    """Fit the short→long correction, model-selected by k-fold cross-validation.

    Candidates: a translation (``W=I``, the safe floor — 209 pairs cannot
    overfit a mean offset) and closed-form ridge over ``alphas``. The winner is
    the candidate with the best CV mean cosine(mapped, long); it is refit on all
    pairs.

    Returns ``(map, report)``. ``report["oof_mapped"]`` holds the winner's
    out-of-fold predictions for every pair — the honest input for downstream
    held-out metrics (2D displacement, clip rate) against the frozen model.
    """
    short = _unit(np.asarray(short, dtype=float))
    long = _unit(np.asarray(long, dtype=float))
    if short.shape != long.shape or short.ndim != 2 or short.shape[0] < folds:
        raise ValueError("fit_register_map expects matching 2D arrays with >= folds rows.")

    n = short.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fold_of = np.empty(n, dtype=int)
    fold_of[order] = np.arange(n) % folds

    candidates: List[Dict[str, Any]] = [{"kind": "translation", "alpha": None}] + [
        {"kind": "ridge", "alpha": float(a)} for a in alphas
    ]

    def _fit(kind: str, alpha: float | None, s: np.ndarray, l: np.ndarray):
        if kind == "translation":
            return _fit_translation(s, l)
        return _fit_ridge(s, l, float(alpha))

    baseline = _mean_cosine(short, long)
    for cand in candidates:
        oof = np.empty_like(short)
        for f in range(folds):
            val = fold_of == f
            weights, intercept = _fit(cand["kind"], cand["alpha"], short[~val], long[~val])
            oof[val] = _unit(short[val] @ weights + intercept)
        cand["cv_cosine"] = _mean_cosine(oof, long)
        cand["oof_mapped"] = oof

    winner = max(candidates, key=lambda c: c["cv_cosine"])
    weights, intercept = _fit(winner["kind"], winner["alpha"], short, long)
    meta = {
        "kind": winner["kind"],
        "alpha": winner["alpha"],
        "cv_cosine": winner["cv_cosine"],
        "baseline_cosine": baseline,
        "n_pairs": n,
        "folds": folds,
    }
    report = {
        "baseline_cosine": baseline,
        "candidates": [
            {k: c[k] for k in ("kind", "alpha", "cv_cosine")} for c in candidates
        ],
        "winner": meta,
        "oof_mapped": winner["oof_mapped"],
    }
    return RegisterMap(weights=weights, intercept=intercept, meta=meta), report


# ── Persistence ─────────────────────────────────────────────────────────────


def save_register_map(rmap: RegisterMap, projection_dir: Path) -> Path:
    projection_dir.mkdir(parents=True, exist_ok=True)
    path = projection_dir / REGISTER_MAP_FILENAME
    np.savez(
        path,
        weights=rmap.weights,
        intercept=rmap.intercept,
        meta=json.dumps(rmap.meta),
        support_baseline=(
            rmap.support_baseline
            if rmap.support_baseline is not None
            else np.empty(0)
        ),
    )
    return path


def load_register_map(projection_dir: Path) -> Optional[RegisterMap]:
    path = projection_dir / REGISTER_MAP_FILENAME
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    # Maps saved before the support recalibration have no baseline array.
    baseline = (
        np.asarray(data["support_baseline"], dtype=float)
        if "support_baseline" in data.files
        else np.empty(0)
    )
    return RegisterMap(
        weights=np.asarray(data["weights"], dtype=float),
        intercept=np.asarray(data["intercept"], dtype=float),
        meta=json.loads(str(data["meta"])),
        support_baseline=baseline if baseline.size else None,
    )
