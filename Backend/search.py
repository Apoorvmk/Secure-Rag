"""Authorized local search utility. A role must be selected explicitly."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from access_policy import TOP_K, ROLES
from document_pipeline import filter_authorized_results, retrieve_authorized

BACKEND_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="How many days of leave do employees get?")
    parser.add_argument("--role", required=True, choices=sorted(ROLES),
        help="authorization role of the operator running this local utility")
    args = parser.parse_args()
    load_dotenv(BACKEND_DIR / ".env")
    url, key = os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY")
    if not url or not key:
        parser.error("Set QDRANT_URL and QDRANT_API_KEY in Backend/.env")
    qdrant = QdrantClient(url=url, api_key=key)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    results = retrieve_authorized(args.question, role=args.role, qdrant=qdrant, model=model, limit=TOP_K)
    for result in filter_authorized_results(results, args.role):
        payload = result.payload
        print(f"SCORE: {result.score}\nTEXT: {payload['text']}\n--------------------")


if __name__ == "__main__":
    main()
