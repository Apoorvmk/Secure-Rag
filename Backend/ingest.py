"""Ingest the repository's existing OKF Markdown files into Qdrant."""

import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from document_pipeline import COLLECTION_NAME, EMBEDDING_MODEL_NAME, ingest_markdown

BACKEND_DIR = Path(__file__).resolve().parent
OKF_DIR = BACKEND_DIR.parent / "documents" / "okf"


def main():
    load_dotenv(BACKEND_DIR / ".env")
    url, key = os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY")
    if not url or not key:
        raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY in Backend/.env")
    qdrant = QdrantClient(url=url, api_key=key)
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    qdrant.create_payload_index(collection_name=COLLECTION_NAME, field_name="access_level",
                                field_schema=PayloadSchemaType.KEYWORD)
    qdrant.create_payload_index(collection_name=COLLECTION_NAME, field_name="document_id",
                                field_schema=PayloadSchemaType.KEYWORD)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    for file_path in sorted(OKF_DIR.glob("*.md")):
        if file_path.name == "index.md":
            continue
        count = ingest_markdown(file_path.read_text(encoding="utf-8"), qdrant=qdrant, model=model)
        print(f"{file_path.name}: indexed {count} chunks")


if __name__ == "__main__":
    main()
