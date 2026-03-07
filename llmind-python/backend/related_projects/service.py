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

class ServiceError(RuntimeError):
    """Raised when an external dependency or model response fails."""


class NodeGenerationPayload(BaseModel):
    parent_id: str = Field(min_length=1)
    options: dict[str, str] = Field(default_factory=dict)


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
) -> NodeGenerationPayload:
    if mode == BackendMode.vllm:
        resolved_base_url = settings.vllm_base_url
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
        raw = completion.choices[0].message.content or ""
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


def _format_projects_for_prompt(projects: list[dict[str, Any]]) -> str:
    if not projects:
        return "No related projects found."

    lines = []
    for index, project in enumerate(projects, start=1):
        name = _safe_str(project.get("Name")).strip() or "(untitled)"
        description = _safe_str(project.get("Descriptions")).strip() or _safe_str(
            project.get("Details")
        ).strip()
        description_line = f"\n  Description: {description}" if description else ""
        lines.append(f"{index}. {name}{description_line}")

    return "\n\n".join(lines)


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
) -> dict[str, Any]:
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

    formatted_taxonomy = _format_taxonomy(taxonomy_nodes)
    related_projects_section = _format_projects_for_prompt(resolved_projects)

    user_prompt = (
        USER_PROMPT_TEMPLATE.replace("{{TAXONOMY}}", formatted_taxonomy)
        .replace("{{SELECTED_NODE_TOPIC}}", focus_node_topic)
        .replace("{{SELECTED_NODE_ID}}", focus_node_id)
        .replace("{{RELATED_PROJECTS}}", related_projects_section)
    )

    resolved_model = model or settings.openai_node_model

    try:
        parsed = _generate_node_payload(
            model=resolved_model,
            user_prompt=user_prompt,
            mode=mode,
            reasoning_effort=reasoning_effort,
        )

        options = {str(key): str(value) for key, value in parsed.options.items()}
        node_array = [
            {
                "node_id": node_id,
                "topic": topic,
                "parent_node": parsed.parent_id,
            }
            for node_id, topic in options.items()
        ]

        return {
            "parent_id": parsed.parent_id,
            "options": options,
            "nodes": node_array,
            "related_projects": resolved_projects,
        }
    except ServiceError:
        raise
    except Exception as exc:
        raise ServiceError("Failed to generate nodes from related projects.") from exc
