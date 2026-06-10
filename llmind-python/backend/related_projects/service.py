from __future__ import annotations

import json
import re
import copy
from typing import Any, Iterable

from pydantic import BaseModel, Field

from config import settings
from utils.clients import build_openai_client, build_vllm_client
from utils.modes import BackendMode
from utils.supabase import get_supabase_client
from utils.prompts import USER_PROMPT_TEMPLATE, SYSTEM_PROMPT
from utils.json import extract_message_json

class ServiceError(RuntimeError):
    """Raised when an external dependency or model response fails."""


class NodeOption(BaseModel):
    id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    # One-sentence description; embedded (with the topic) to position the option
    # in the design space and to seed later retrieval. Required with no default —
    # structured-output strict mode drops defaulted fields from ``required``
    # (see CLAUDE.md schema rules) — but unconstrained, so an empty string
    # degrades placement quality instead of failing the whole generation.
    desc: str


class NodeGenerationPayload(BaseModel):
    parent_id: str = Field(min_length=1)
    options: list[NodeOption] = Field(default_factory=list)


CURATED_FALLBACK_PROJECTS: list[dict[str, Any]] = [
    {
        "id": None,
        "Id": None,
        "Name": "Relevant projects will appear here",
        "Descriptions": "",
        "Details": "",
        "Image": None,
    }
]


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _extract_json_from_markdown(text: str) -> NodeGenerationPayload:
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    payload_text = json_match.group(1).strip() if json_match else text.strip()
    try:
        payload_obj = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ServiceError("Failed to parse model response as JSON.") from exc
    return NodeGenerationPayload.model_validate(payload_obj)


def _make_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    schema_copy = copy.deepcopy(schema)

    def _visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            node.setdefault("additionalProperties", False)
            node.setdefault("properties", {})
        for child in node.get("properties", {}).values():
            _visit(child)
        for child in node.get("$defs", {}).values():
            _visit(child)
        if "items" in node:
            _visit(node["items"])

    _visit(schema_copy)
    return schema_copy


def _resolve_system_message() -> str:
    if isinstance(SYSTEM_PROMPT, dict):
        return str(SYSTEM_PROMPT.get("system", "")).strip()
    return str(SYSTEM_PROMPT).strip()


def _generate_node_payload(
    *,
    model: str,
    user_prompt: str,
    mode: BackendMode,
    reasoning_effort: str = "medium",
    base_url: str | None = None,
) -> NodeGenerationPayload:
    if mode == BackendMode.vllm:
        resolved_base_url = base_url or settings.vllm_base_url
        client = build_vllm_client(resolved_base_url)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "node_generation_payload",
                "strict": True,
                "schema": _make_strict_schema(NodeGenerationPayload.model_json_schema()),
            },
        }
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _resolve_system_message()},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
        )
        raw = extract_message_json(completion.choices[0].message)
        return NodeGenerationPayload.model_validate_json(raw)

    client = build_openai_client()
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": _resolve_system_message()},
            {"role": "user", "content": user_prompt},
        ],
        response_format=NodeGenerationPayload,
        reasoning_effort=reasoning_effort,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is not None:
        return parsed
    raw_text = completion.choices[0].message.content or ""
    return _extract_json_from_markdown(raw_text)


def _extract_related_project(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    image_value = metadata.get("Image") or row.get("image")
    row_id = metadata.get("id") or row.get("id") or metadata.get("Id")

    return {
        "id": _safe_str(row_id) or None,
        "Id": _safe_str(metadata.get("Id")) or None,
        "Name": _safe_str(metadata.get("Name") or row.get("name")) or "(untitled)",
        "Descriptions": _safe_str(metadata.get("Descriptions") or row.get("description")),
        "Details": _safe_str(metadata.get("Details") or row.get("detail") or row.get("content")),
        "Image": _safe_str(image_value).strip() or None,
    }


def _format_taxonomy(nodes: Iterable[dict[str, Any]]) -> str:
    node_list = [dict(node) for node in nodes]
    if not node_list:
        return ""

    node_map = {node.get("id"): node for node in node_list if node.get("id")}
    root_node = next((node for node in node_list if node.get("isroot") is True), None)
    if root_node is None:
        root_node = next((node for node in node_list if not node.get("parentid")), None)
    if root_node is None:
        return ""

    def _format_node(node_id: str, indent: int = 0) -> str:
        node = node_map.get(node_id)
        if not node:
            return ""

        prefix = "  " * indent
        topic = _safe_str(node.get("topic"))
        line = f"{prefix}- {topic} ({node_id})\n"

        children = [
            child for child in node_list if _safe_str(child.get("parentid")) == node_id
        ]
        children_text = "".join(
            _format_node(_safe_str(child.get("id")), indent + 1) for child in children
        )
        return f"{line}{children_text}"

    return _format_node(_safe_str(root_node.get("id")))


def _format_taxonomy_focused(nodes: Iterable[dict[str, Any]], focus_node_id: str) -> str:
    """Compact, *complete* taxonomy view for node generation.

    Shows the root + every aspect (the 2-level skeleton) and expands the focus
    aspect's existing options. Unlike a char-truncated full tree, this is always
    well-formed and bounded regardless of total tree size, so the model reliably
    sees the structure to follow and the existing options to avoid duplicating.
    """
    node_list = [dict(n) for n in nodes]
    if not node_list:
        return ""

    by_id = {_safe_str(n.get("id")): n for n in node_list if n.get("id")}
    children_of: dict[str, list[dict[str, Any]]] = {}
    for n in node_list:
        children_of.setdefault(_safe_str(n.get("parentid")), []).append(n)

    root = next((n for n in node_list if n.get("isroot") is True), None) or next(
        (n for n in node_list if not n.get("parentid")), None
    )
    if not root:
        return ""
    root_id = _safe_str(root.get("id"))

    # The aspect to expand: the focus itself if it's an aspect, else the focus's
    # parent aspect (so an option focus still shows its sibling options).
    focus = by_id.get(focus_node_id)
    expand_id = focus_node_id
    if focus is not None and _safe_str(focus.get("parentid")) not in ("", root_id):
        expand_id = _safe_str(focus.get("parentid"))

    lines = [f"- {_safe_str(root.get('topic'))} ({root_id})"]
    for aspect in children_of.get(root_id, []):
        aid = _safe_str(aspect.get("id"))
        lines.append(f"  - {_safe_str(aspect.get('topic'))} ({aid})")
        if aid == expand_id:
            for opt in children_of.get(aid, []):
                lines.append(f"    - {_safe_str(opt.get('topic'))} ({_safe_str(opt.get('id'))})")
    return "\n".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars].rstrip() + "…"


def _format_projects_for_prompt(projects: list[dict[str, Any]]) -> str:
    if not projects:
        return "No related projects found."

    lines = []
    for index, project in enumerate(projects, start=1):
        name = _safe_str(project.get("Name")).strip() or "(untitled)"
        description = _safe_str(project.get("Descriptions")).strip() or _safe_str(
            project.get("Details")
        ).strip()
        # Descriptions can be very long (whole project write-ups) — cap each so the
        # prompt stays within a small context window.
        description = _truncate(description, settings.prompt_max_project_chars)
        description_line = f"\n  Description: {description}" if description else ""
        lines.append(f"{index}. {name}{description_line}")

    return "\n\n".join(lines)


def _fetch_related_projects_local(
    *, query: str, limit: int, similarity_threshold: float | None
) -> list[dict[str, Any]]:
    """Search the offline npz index, embedding the query with the local model.

    Touches neither Supabase nor OpenAI — used when ``VECTOR_STORE=local``.
    """
    from utils import local_store

    threshold = (
        settings.supabase_similarity_threshold
        if similarity_threshold is None
        else similarity_threshold
    )
    try:
        client = build_vllm_client(settings.vllm_base_url)
        response = client.embeddings.create(
            model=settings.vllm_embed_model, input=[query]
        )
        query_embedding = response.data[0].embedding if response.data else None
        if not isinstance(query_embedding, list):
            raise ServiceError("Failed to generate a local embedding for the query.")
        rows = local_store.search(query_embedding, k=limit, threshold=threshold)
        return [_extract_related_project({"metadata": row}) for row in rows]
    except ServiceError:
        raise
    except Exception as exc:
        raise ServiceError("Failed to search the local vector index.") from exc


def fetch_related_projects(
    *,
    query: str,
    limit: int = 5,
    similarity_threshold: float | None = None,
    embedding_model: str | None = None,
    match_function: str | None = None,
) -> list[dict[str, Any]]:
    trimmed_query = query.strip()
    if not trimmed_query:
        return []

    if settings.vector_store == "local":
        return _fetch_related_projects_local(
            query=trimmed_query, limit=limit, similarity_threshold=similarity_threshold
        )

    resolved_embedding_model = embedding_model or settings.openai_embed_model
    resolved_match_function = match_function or settings.supabase_match_function
    resolved_similarity_threshold = (
        settings.supabase_similarity_threshold
        if similarity_threshold is None
        else similarity_threshold
    )

    openai_client = build_openai_client()
    supabase_client = get_supabase_client()

    try:
        embedding_response = openai_client.embeddings.create(
            model=resolved_embedding_model,
            input=[trimmed_query],
        )
        query_embedding = embedding_response.data[0].embedding if embedding_response.data else None
        if not isinstance(query_embedding, list):
            raise ServiceError("Failed to generate a valid embedding for the query.")

        rpc_response = supabase_client.rpc(
            resolved_match_function,
            {
                "query_embedding": query_embedding,
                "match_count": limit,
                "similarity_threshold": resolved_similarity_threshold,
            },
        ).execute()

        rows = list(getattr(rpc_response, "data", None) or [])
        return [_extract_related_project(row) for row in rows[:limit] if isinstance(row, dict)]
    except ServiceError:
        raise
    except Exception as exc:
        raise ServiceError("Failed to fetch related projects from Supabase.") from exc


def build_related_query_text(
    *,
    topic: str,
    lineage: list[str] | None = None,
    description: str | None = None,
) -> str:
    cleaned_topic = topic.strip()
    lineage_parts = lineage or []
    lineage_text = " > ".join(
        part.strip() for part in lineage_parts[1:] if _safe_str(part).strip()
    ).strip()
    description_text = _safe_str(description).strip()
    query_source = " | ".join(
        part for part in [lineage_text, description_text, cleaned_topic] if part
    )
    return (query_source or cleaned_topic or "test").strip()


def search_related_projects(
    *,
    topic: str,
    lineage: list[str] | None = None,
    description: str | None = None,
    should_query_supabase: bool = True,
    limit: int = 5,
    similarity_threshold: float | None = None,
    embedding_model: str | None = None,
    match_function: str | None = None,
) -> list[dict[str, Any]]:
    if not should_query_supabase:
        return [dict(project) for project in CURATED_FALLBACK_PROJECTS]

    query_text = build_related_query_text(
        topic=topic,
        lineage=lineage,
        description=description,
    )

    try:
        projects = fetch_related_projects(
            query=query_text,
            limit=limit,
            similarity_threshold=similarity_threshold,
            embedding_model=embedding_model,
            match_function=match_function,
        )
        if projects:
            return projects
        return [dict(project) for project in CURATED_FALLBACK_PROJECTS]
    except ServiceError:
        return [dict(project) for project in CURATED_FALLBACK_PROJECTS]


def generate_nodes_from_related_projects(
    *,
    focus_node_id: str,
    focus_node_topic: str,
    taxonomy_nodes: list[dict[str, Any]],
    related_projects: list[dict[str, Any]] | None = None,
    lineage: list[str] | None = None,
    description: str | None = None,
    should_query_supabase: bool = True,
    model: str | None = None,
    mode: BackendMode = BackendMode.openai,
    base_url: str | None = None,
    reasoning_effort: str = "medium",
    prompt_template: str | None = None,
    extra_template_fields: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Generate child options for a focus node, seeded by related projects.

    ``prompt_template`` defaults to the aspect-exploration prompt; the design
    space passes its own location-conditioned template (with
    ``extra_template_fields`` for placeholders the default prompt lacks, e.g.
    ``{{NEARBY_OPTIONS}}``).
    """
    if not focus_node_id.strip() or not focus_node_topic.strip():
        raise ServiceError("Focus node id and topic are required.")

    resolved_projects = (
        [dict(project) for project in related_projects]
        if related_projects is not None
        else search_related_projects(
            topic=focus_node_topic,
            lineage=lineage,
            description=description,
            should_query_supabase=should_query_supabase,
            limit=settings.supabase_match_count,
        )
    )

    # Focused, complete view (root + aspects + focus options) — bounded and
    # well-formed so the model follows the structure; safety-capped just in case.
    formatted_taxonomy = _truncate(
        _format_taxonomy_focused(taxonomy_nodes, focus_node_id),
        settings.prompt_max_taxonomy_chars,
    )
    related_projects_section = _format_projects_for_prompt(resolved_projects)

    template = prompt_template or USER_PROMPT_TEMPLATE
    user_prompt = (
        template.replace("{{TAXONOMY}}", formatted_taxonomy)
        .replace("{{SELECTED_NODE_TOPIC}}", focus_node_topic)
        .replace("{{SELECTED_NODE_ID}}", focus_node_id)
        .replace("{{RELATED_PROJECTS}}", related_projects_section)
    )
    for placeholder, value in (extra_template_fields or {}).items():
        user_prompt = user_prompt.replace(f"{{{{{placeholder}}}}}", value)

    resolved_model = model or settings.openai_node_model

    try:
        parsed = _generate_node_payload(
            model=resolved_model,
            user_prompt=user_prompt,
            mode=mode,
            reasoning_effort=reasoning_effort,
            base_url=base_url,
        )

        # The parent is always the focus node we asked about — a local model's
        # echoed parent_id is unreliable and a mismatch makes the frontend drop
        # every generated node ("no matching parent"). Force the known parent.
        options = {opt.id: opt.topic for opt in parsed.options}
        node_array = [
            {
                "node_id": opt.id,
                "topic": opt.topic,
                "desc": opt.desc.strip(),
                "parent_node": focus_node_id,
            }
            for opt in parsed.options
        ]

        return {
            "parent_id": focus_node_id,
            "options": options,
            "nodes": node_array,
            "related_projects": resolved_projects,
        }
    except ServiceError:
        raise
    except Exception as exc:
        raise ServiceError("Failed to generate nodes from related projects.") from exc
