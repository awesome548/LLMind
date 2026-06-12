"""Budgeted local-LLM calls — thinking-model aware (Part 12 K2 rules).

The local stack's window is small (4096 tokens, prompt + thinking + answer)
and the serving model may be thinking-only (Qwen3.6 ignores /no_think and
``chat_template_kwargs.enable_thinking`` — verified live). So every judgment/
generation call sizes ``max_tokens`` to whatever the prompt leaves, lets the
deliberation finish, and surfaces the reasoning so callers can salvage a
verdict when the cap is still hit.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator

from openai import BadRequestError

from config import settings
from utils.clients import build_vllm_client

# Total context window assumed for the local model (LM Studio load setting).
LOCAL_CTX = 4096
# Safety margin for the token estimate's error + chat-template wrapper tokens.
_MARGIN = 96
# Never ask for less than this — a smaller response can't carry a verdict.
_MIN_BUDGET = 256


def estimate_tokens(text: str) -> int:
    """Charset-aware prompt-token estimate. Pure.

    ASCII prose runs ~3-4.5 chars/token; CJK and other non-ASCII text runs
    ≥1 token per character — a flat chars//3 badly undershoots on chunks with
    Chinese project names/descriptions, which made prompt+max_tokens overflow
    the window (the option-7 'Context size has been exceeded' failure).
    """
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars // 3 + round((len(text) - ascii_chars) * 1.2)


def iter_json_objects(text: str) -> Iterator[Dict[str, Any]]:
    """Every parseable top-level JSON object in a text, in order. Pure.

    Brace-balanced scanning (string- and escape-aware), so objects whose
    values themselves contain braces parse correctly — a flat ``\\{[^{}]*\\}``
    regex truncates those, silently dropping the LLM's answer.
    """
    raw = text or ""
    i = 0
    while (start := raw.find("{", i)) != -1:
        depth = 0
        in_string = False
        escaped = False
        for j in range(start, len(raw)):
            ch = raw[j]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            elif ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(raw[start : j + 1])
                        if isinstance(obj, dict):
                            yield obj
                    except ValueError:
                        pass
                    break
        else:
            # Unbalanced to the end of the text — nothing more to find.
            return
        i = max(j + 1, start + 1)


def budgeted_completion(prompt: str) -> tuple[str, str]:
    """One deterministic chat call with window-aware ``max_tokens``.

    Returns ``(content, reasoning)`` — ``reasoning`` is the thinking text when
    the server surfaces it (LM Studio's ``reasoning_content``), else ``""``.
    The token estimate is only an estimate: when the server still rejects the
    request as over-window, the budget halves and the call retries (cheap —
    the rejected call never ran) down to a floor.
    """
    client = build_vllm_client(settings.vllm_base_url)
    budget = max(_MIN_BUDGET, LOCAL_CTX - estimate_tokens(prompt) - _MARGIN)
    while True:
        try:
            completion = client.chat.completions.create(
                model=settings.vllm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=budget,
            )
            break
        except BadRequestError as exc:
            if "context" in str(exc).lower() and budget > _MIN_BUDGET:
                budget = max(_MIN_BUDGET, budget // 2)
                continue
            raise
    message = completion.choices[0].message
    extra = getattr(message, "model_extra", None) or {}
    return message.content or "", str(extra.get("reasoning_content") or "")
