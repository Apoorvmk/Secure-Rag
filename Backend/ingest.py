from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from dotenv import load_dotenv
import os
import uuid
import yaml
from pathlib import Path

load_dotenv()

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

model = SentenceTransformer("all-MiniLM-L6-v2")

OKF_DIR = Path(__file__).resolve().parent.parent / "documents" / "okf"

CHUNK_SIZE = 500


for file_path in OKF_DIR.glob("*.md"):

    if file_path.name == "index.md":
        continue

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Read YAML frontmatter
    parts = content.split("---", 2)

    metadata = yaml.safe_load(parts[1])
    text = parts[2].strip()

    # Split into chunks
    chunks = [
        text[i:i + CHUNK_SIZE]
        for i in range(0, len(text), CHUNK_SIZE)
    ]

    embeddings = model.encode(chunks)

    points = []

    for chunk, embedding in zip(chunks, embeddings):

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                    "access_level": metadata["access_level"],
                    "title": metadata["title"],
                    "source": metadata["source"]
                }
            )
        )

    qdrant.upsert(
        collection_name="company_documents",
        points=points
    )

    print(
        f"{file_path.name}: "
        f"uploaded {len(chunks)} chunks "
        f"(access: {metadata['access_level']})"
    )