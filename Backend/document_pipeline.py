"""PDF to OKF Markdown to Qdrant ingestion helpers."""

import uuid
from pathlib import Path

import yaml
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from access_policy import ROLES, authorized_qdrant_filter

COLLECTION_NAME = "company_documents"
CHUNK_SIZE = 500
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class DocumentValidationError(ValueError):
    """Safe validation failures caused by an uploaded or authored document."""


def pdf_to_text(pdf_path: Path) -> str:
    import pymupdf

    try:
        with pymupdf.open(pdf_path) as pdf:
            text = "\n".join(page.get_text() for page in pdf).strip()
    except Exception as exc:
        raise DocumentValidationError("The uploaded PDF is malformed or unreadable") from exc
    if not text:
        raise DocumentValidationError("The uploaded PDF contains no extractable text")
    return text


def make_okf_markdown(text: str, *, title: str, access_level: str,
                      source: str, document_id: str, uploaded_by: int) -> str:
    if access_level not in ROLES:
        raise DocumentValidationError("Invalid access level")
    frontmatter = yaml.safe_dump({
        "type": "Company Document",
        "title": title,
        "description": f"Uploaded company document for the {access_level} access tier.",
        "access_level": access_level,
        "source": source,
        "document_id": document_id,
        "uploaded_by": uploaded_by,
    }, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{frontmatter}\n---\n\n# {title}\n\n{text}\n"


def parse_okf(markdown: str) -> tuple[dict, str]:
    parts = markdown.split("---", 2)
    if len(parts) != 3 or parts[0].strip():
        raise DocumentValidationError("Invalid OKF Markdown frontmatter")
    try:
        metadata = yaml.safe_load(parts[1])
    except yaml.YAMLError as exc:
        raise DocumentValidationError("Invalid OKF Markdown frontmatter") from exc
    if (not isinstance(metadata, dict)
            or not isinstance(metadata.get("access_level"), str)
            or metadata.get("access_level") not in ROLES
            or not isinstance(metadata.get("source"), str)
            or not metadata["source"].strip()):
        raise DocumentValidationError("Invalid OKF access metadata")
    return metadata, parts[2].strip()


def chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    if size <= 0:
        raise DocumentValidationError("Chunk size must be positive")
    return [text[index:index + size] for index in range(0, len(text), size)]


def ingest_markdown(markdown: str, *, qdrant, model) -> int:
    """Idempotently upsert document chunks. Clean up partial writes on failure."""
    metadata, text = parse_okf(markdown)
    document_id = str(metadata.get("document_id", "")).strip()
    if not document_id:
        # Existing checked-in OKF documents use a stable source-derived ID.
        document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(metadata.get("source", ""))))
    chunks = chunk_text(text)
    if not chunks:
        raise DocumentValidationError("Document has no text to index")
    vectors = model.encode(chunks)
    if len(vectors) != len(chunks):
        raise DocumentValidationError("Embedding model returned an incomplete vector set")
    point_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{index}")) for index in range(len(chunks))]
    points = [PointStruct(id=point_ids[index], vector=vector.tolist(), payload={
        "text": chunk,
        "access_level": metadata["access_level"],
        "title": str(metadata.get("title", "Untitled")),
        "source": str(metadata.get("source", "unknown")),
        "document_id": document_id,
        "chunk_index": index,
        "uploaded_by": metadata.get("uploaded_by"),
        "ingestion_status": "processing",
    }) for index, (chunk, vector) in enumerate(zip(chunks, vectors))]
    has_document_id = bool(metadata.get("document_id"))
    indexed_by = "document_id" if has_document_id else "source"
    indexed_value = document_id if has_document_id else str(metadata.get("source", ""))
    condition = Filter(must=[FieldCondition(key=indexed_by, match=MatchValue(value=indexed_value))])
    try:
        # Bust query caches before any existing stable-ID chunks become processing.
        from retrieval_cache import invalidate_retrieval_cache
        invalidate_retrieval_cache()
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        # Remove obsolete chunks after the replacement chunks are confirmed.
        new_id_set = set(point_ids)
        old_ids = []
        offset = None
        while True:
            page, offset = qdrant.scroll(collection_name=COLLECTION_NAME, scroll_filter=condition,
                with_payload=False, with_vectors=False, limit=1000, offset=offset)
            old_ids.extend(point.id for point in page if str(point.id) not in new_id_set)
            if offset is None:
                break
        if old_ids:
            qdrant.delete(collection_name=COLLECTION_NAME, points_selector=old_ids, wait=True)
        qdrant.set_payload(collection_name=COLLECTION_NAME,
            payload={"ingestion_status": "searchable"}, points=condition, wait=True)
    except Exception:
        # The owning upload records failure; ensure partial points cannot leak.
        try:
            qdrant.delete(collection_name=COLLECTION_NAME, points_selector=condition, wait=True)
        except Exception:
            pass
        raise
    return len(points)


def retrieve_authorized(question: str, *, role: str, qdrant, model, limit: int):
    vector = model.encode(question).tolist()
    return qdrant.query_points(collection_name=COLLECTION_NAME, query=vector,
        query_filter=authorized_qdrant_filter(role),
        limit=limit).points


def filter_authorized_results(results, role: str):
    """Defense in depth: discard malformed or out-of-policy returned payloads."""
    from access_policy import allowed_access_levels
    permitted = set(allowed_access_levels(role))
    return [point for point in results
        if isinstance(getattr(point, "payload", None), dict)
        and isinstance(point.payload.get("access_level"), str)
        and point.payload["access_level"] in permitted
        and point.payload.get("ingestion_status") == "searchable"
        and isinstance(point.payload.get("text"), str)]
