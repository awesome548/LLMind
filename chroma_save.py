# build_and_store_embeddings.py

from __future__ import annotations
import json, os
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

load_dotenv()

HERE = Path(__file__).resolve().parent
INPUT_PATH = HERE / "data" / "media_architecture.json"
OUTPUT_PATH = HERE / "data" / "cleaned_media_architecture.json"

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "mab_projects")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "text-embedding-3-small")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

def to_doc_text(item: dict) -> str:
    return (item.get("Description") or "").strip()

def extract_project_id(url: str) -> str:
    """Extract project ID from URL after 'project/'"""
    if "project/" in url:
        return url.split("project/")[-1]
    # Fallback to UUID if URL doesn't match expected format
    return str(uuid4())

def main() -> None:
    raw = INPUT_PATH.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be an array of objects.")

    # Clean drop html_main + extract project id from url
    cleaned = []
    for item in data:
        item = dict(item)
        # Extract project ID from URL before removing it
        project_id = extract_project_id(item.get("url", ""))
        item.pop("url", None)
        item.pop("html_main", None)
        cleaned.append({"id": project_id, **item})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved cleaned data to {OUTPUT_PATH}")

    # Prepare docs
    ids_all = [it["id"] for it in cleaned]
    docs_all = [to_doc_text(it) for it in cleaned]
    metas_all = [{"Name": it.get("Name"), "Description": it.get("Description")} for it in cleaned]

    # ---- IMPORTANT: filter out invalid/empty documents ----
    filtered = [(i, d, m) for i, d, m in zip(ids_all, docs_all, metas_all) if isinstance(d, str) and d.strip()]
    if not filtered:
        print("No non-empty descriptions to embed. Nothing to upsert.")
        return

    ids, documents, metadatas = map(list, zip(*filtered))
    skipped = len(ids_all) - len(ids)
    if skipped:
        print(f"Skipped {skipped} item(s) with empty or invalid Description.")

    # Chroma persistent client + embedding function (Chroma will call OpenAI for you)
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name=OPENAI_MODEL,
        ),
    )

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f'Upserted {len(ids)} items into collection "{COLLECTION_NAME}" (db: {CHROMA_DB_PATH}).')

if __name__ == "__main__":
    main()