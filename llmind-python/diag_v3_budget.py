"""Throwaway: real LED chunk with thinking allowed to finish (budgeted max_tokens)."""
import json, time
import numpy as np
from backend.corpus.annotate import JUDGE_BATCH, SHORTLIST_K, parse_membership
from backend.corpus.service import embed_texts, load_corpus_vectors, load_index_meta
from config import settings
from pipeline import register_alignment as ra
from utils.clients import build_vllm_client
from utils.prompts import ANNOTATE_OPTION_PROMPT

name = "LED wall panels"
desc = "Outdoor-grade LED panels with scalable resolution and high brightness for building facades."
ids, corpus_unit = load_corpus_vectors()
meta = load_index_meta()
vec = embed_texts([f"{name}. {desc}"])
rmap = ra.load_register_map(settings.projection_dir)
if rmap is not None:
    vec = rmap.apply(vec)
top = np.argsort(corpus_unit @ vec[0])[-SHORTLIST_K:][::-1]

def summary(pid):
    r = meta.get(pid, {})
    d = ra.build_short_text("", r.get("Descriptions") or "", max_chars=220)
    det = " ".join((r.get("Details") or "").split())[:200]
    return f"{d} [Details: {det}]" if det else d

chunk = [{"name": meta.get(ids[i], {}).get("Name", "?"), "summary": summary(ids[i])} for i in top[:JUDGE_BATCH]]
projects = "\n".join(f"{i + 1}. {r['name']}: {r['summary']}" for i, r in enumerate(chunk))
prompt = (
    ANNOTATE_OPTION_PROMPT.replace("{{OPTION_NAME}}", name)
    .replace("{{OPTION_DESC}}", desc)
    .replace("{{PROJECTS}}", projects)
)
est_prompt = len(prompt) // 3
budget = max(256, 4096 - est_prompt - 96)
print(f"prompt chars {len(prompt)}, est tokens {est_prompt}, max_tokens {budget}")
client = build_vllm_client(settings.vllm_base_url)
t0 = time.time()
completion = client.chat.completions.create(
    model=settings.vllm_model,
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    max_tokens=budget,
)
msg = completion.choices[0].message
extra = getattr(msg, "model_extra", None) or {}
reasoning = str(extra.get("reasoning_content") or "")
print(f"elapsed {time.time()-t0:.0f}s  finish={completion.choices[0].finish_reason}  usage={completion.usage.prompt_tokens}+{completion.usage.completion_tokens}")
print("content:", json.dumps(msg.content))
print("parsed:", parse_membership(msg.content or "", len(chunk)))
print("reasoning tail (300):", json.dumps(reasoning[-300:]))
