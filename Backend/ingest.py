
"""Ingest existing OKF Markdown files into Qdrant."""

import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from document_pipeline import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    ingest_markdown,
)

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
OKF_DIR = BASE_DIR / "documents" / "okf"

# Load .env from project root
load_dotenv(BASE_DIR / ".env")


def main():
    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY")

    if not url or not key:
        raise RuntimeError(
            "QDRANT_URL or QDRANT_API_KEY missing from root .env"
        )

    if not OKF_DIR.exists():
        raise FileNotFoundError(f"OKF folder not found: {OKF_DIR}")

    files = [
        f for f in sorted(OKF_DIR.glob("*.md"))
        if f.name != "index.md"
    ]

    if not files:
        raise RuntimeError(f"No Markdown files found in {OKF_DIR}")

    print(f"Found {len(files)} OKF documents.")

    qdrant = QdrantClient(url=url, api_key=key)

    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            ),
        )

    for field in [
        "access_level",
        "document_id",
        "ingestion_status",
        "source"
    ]:
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )

    from sentence_transformers import SentenceTransformer

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    total_chunks = 0

    for file_path in files:
        content = file_path.read_text(encoding="utf-8")

        count = ingest_markdown(
            content,
            qdrant=qdrant,
            model=model
        )

        total_chunks += count
        print(f"{file_path.name}: indexed {count} chunks")

    print(f"\nTotal indexed chunks: {total_chunks}")


if __name__ == "__main__":
    main()
