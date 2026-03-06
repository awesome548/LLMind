from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def read_json_array(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Unable to read JSON from {path}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"Input file must contain a JSON array: {path}")
    return [dict(item) for item in data if isinstance(item, dict)]


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Support Pydantic BaseModel instances via .model_dump()
    if hasattr(data, "model_dump"):
        serialisable = data.model_dump()
    else:
        serialisable = data
    path.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")


_JSON_FENCE_RE = re.compile(r"```json\s*(?P<body>.*?)\s*```", re.IGNORECASE | re.DOTALL)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_json_between_markers(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from a markdown ```json fenced block, else best-effort braces.
    Returns a dict or None.
    """
    m = _JSON_FENCE_RE.search(text)
    candidate = m.group("body") if m else None
    if not candidate:
        m2 = _BRACE_RE.search(text)
        candidate = m2.group(0) if m2 else None
    if not candidate:
        return None

    try:
        return json.loads(candidate)
    except Exception:
        candidate2 = candidate.strip().strip("`")
        try:
            return json.loads(candidate2)
        except Exception:
            return None