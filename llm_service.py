from google import genai
from dotenv import load_dotenv
import os
import re
import json
from typing import Optional, List, Dict, Tuple, Any

class LLMClient:
    def __init__(self):
        load_dotenv()
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate_content(self, model, contents, config):
        return self.client.models.generate_content(model=model, contents=contents, config=config)

# =============================
# Service functions
# =============================


def get_response_from_llm(
    user_text: str,
    client: LLMClient,
    model: str,
    system_message: str = "You are a helpful assistant.",
    msg_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Send (system + user) to LLM, return (text, updated_history).
    """
    from google import genai
    from google.genai import types
    if msg_history is None:
        msg_history = []
    else:
        msg_history.append({"role": "user", "content": user_text})
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
        system_instruction=system_message,
        # contents=[user_text],
    )
    response = client.generate_content(model=model, contents=user_text, config=config)
    return response.text, []


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