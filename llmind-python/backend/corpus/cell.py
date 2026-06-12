"""Cross-tab cell generation — ITERATION-PLAN Part 12 B2.

Halskov (2021): an empty option×option cell in the annotated schema is an
exact, nameable gap. This module fills one on request: pole-conditioned
generation seeded with half-matching precedents (projects annotated with one
of the two options), whose kept result becomes a candidate skeleton on the
frontend. The combination is made IN LANGUAGE (the evidence rule: deltas and
gaps brief the LLM; embeddings never construct).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from config import settings
from utils.prompts import GENERATE_CELL_PROMPT
from pipeline import register_alignment as ra
from backend.corpus.llm import budgeted_completion, iter_json_objects
from backend.corpus.service import CorpusServiceError, load_index_meta

# Half-matching exemplars included in the prompt — enough to set the register
# and both poles without crowding the 4k window (Part 12 K2 rules).
MAX_EXEMPLARS = 8
CELL_PROMPT_VERSION = "cell-v1"


def parse_idea(text: str) -> Dict[str, str] | None:
    """First ``{name, desc}``-shaped JSON object in an LLM response. Pure."""
    for obj in iter_json_objects(text):
        name, desc = obj.get("name"), obj.get("desc")
        if isinstance(name, str) and isinstance(desc, str) and name.strip() and desc.strip():
            return {"name": name.strip(), "desc": desc.strip()}
    return None


def _exemplar_lines(exemplar_ids: List[str]) -> List[str]:
    """One-line summaries for the half-matching precedents (id order kept)."""
    meta = load_index_meta()
    lines: List[str] = []
    for pid in exemplar_ids[:MAX_EXEMPLARS]:
        record = meta.get(pid)
        if not record:
            continue
        summary = ra.build_short_text("", record.get("Descriptions") or "", max_chars=160)
        lines.append(f"- {record.get('Name') or '(untitled)'}: {summary}")
    return lines


def generate_cell(
    *,
    aspect_a: str,
    option_a: Dict[str, str],
    aspect_b: str,
    option_b: Dict[str, str],
    exemplar_ids: List[str],
) -> Dict[str, Any]:
    """One project concept committing to BOTH options of an empty cell.

    Returns ``{name, desc, cell: [optionA, optionB], exemplars_used}``. The
    row is logged to ``generate_log.jsonl`` (its own ``prompt_version``, so it
    aggregates as a separate variant in log-stats).
    """
    lines = _exemplar_lines(exemplar_ids)
    prompt = (
        GENERATE_CELL_PROMPT.replace("{{ASPECT_A}}", aspect_a.strip() or "Aspect A")
        .replace("{{OPTION_A_NAME}}", option_a["name"].strip())
        .replace("{{OPTION_A_DESC}}", (option_a.get("desc") or "").strip() or "(no description)")
        .replace("{{ASPECT_B}}", aspect_b.strip() or "Aspect B")
        .replace("{{OPTION_B_NAME}}", option_b["name"].strip())
        .replace("{{OPTION_B_DESC}}", (option_b.get("desc") or "").strip() or "(no description)")
        .replace("{{EXEMPLARS}}", "\n".join(lines) or "(none annotated yet)")
    )
    content, reasoning = budgeted_completion(prompt)
    # Thinking models that hit the cap mid-deliberation leave the concept in
    # the reasoning tail, if anywhere.
    idea = parse_idea(content) or (parse_idea(reasoning[-600:]) if not content.strip() else None)
    if idea is None:
        raise CorpusServiceError(
            "The model returned no parseable concept for this cell — try again."
        )

    # Same evaluation log as generate-at; distinct prompt_version segments it.
    from backend.projection.service import _log_generation

    _log_generation(
        {
            "ts": time.time(),
            "prompt_version": CELL_PROMPT_VERSION,
            "seed_strategy": "half-matching",
            "kind": "cell",
            "cell": [option_a["name"], option_b["name"]],
            "aspects": [aspect_a, aspect_b],
            "exemplars": len(lines),
            "mode": settings.vllm_model,
            "nodes": [],
        }
    )
    return {
        "name": idea["name"],
        "desc": idea["desc"],
        "cell": [option_a["name"], option_b["name"]],
        "exemplars_used": len(lines),
    }
