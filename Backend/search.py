from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

load_dotenv()

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

model = SentenceTransformer("all-MiniLM-L6-v2")

question = "How many days of leave do employees get?"

query_vector = model.encode(question).tolist()

results = qdrant.query_points(
    collection_name="company_documents",
    query=query_vector,
    limit=TOP_K
).points

for result in results:
    print("SCORE:", result.score)
    print("TEXT:", result.payload["text"])
    print("--------------------")