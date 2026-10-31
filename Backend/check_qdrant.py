
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import os

# Load the root .env file
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

collection = "company_documents"

if not client.collection_exists(collection):
    print("Collection does not exist!")
    raise SystemExit

count = client.count(
    collection_name=collection,
    exact=True
).count

print("Total indexed chunks:", count)

if count == 0:
    print("No documents are indexed.")
    raise SystemExit

offset = None
documents = {}

while True:
    points, offset = client.scroll(
        collection_name=collection,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )

    for point in points:
        payload = point.payload or {}
        doc_id = payload.get("document_id") or payload.get("source", "unknown")

        documents.setdefault(doc_id, {
            "title": payload.get("title", "Unknown"),
            "source": payload.get("source", "Unknown"),
            "access_level": payload.get("access_level", "Unknown"),
            "chunks": 0
        })

        documents[doc_id]["chunks"] += 1

    if offset is None:
        break

print("\nIndexed documents:")

for doc_id, details in documents.items():
    print("\nDocument ID:", doc_id)
    print("Title:", details["title"])
    print("Source:", details["source"])
    print("Access:", details["access_level"])
    print("Chunks:", details["chunks"])
