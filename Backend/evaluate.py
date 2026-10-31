"""Reproducible retrieval evaluation; default mode is local and synthetic."""

import argparse
import json
from pathlib import Path
import sys
import time

from access_policy import ROLES, allowed_access_levels
from rag_pipeline import bm25_rank, tokenize


def reference_evidence_coverage(reference: str, retrieved: list[str]) -> float | None:
    """Lexical evidence coverage proxy, not a semantic faithfulness judgment."""
    terms = {term for term in tokenize(reference) if len(term) > 2}
    if not terms:
        return None
    evidence = {term for text in retrieved for term in tokenize(text)}
    return len(terms & evidence) / len(terms)


def answer_evidence_support(answer: str, retrieved: list[str]) -> float | None:
    """Token coverage of a supplied candidate answer by retrieved evidence; a proxy only."""
    return reference_evidence_coverage(answer, retrieved)


def run_local(dataset: dict, k: int) -> dict:
    corpus = dataset.get("corpus", [])
    queries = dataset.get("queries", [])
    if not isinstance(corpus, list) or not isinstance(queries, list) or not queries:
        raise ValueError("Dataset requires non-empty 'queries' and a 'corpus' list")
    recalls, reciprocal_ranks, evidence_scores, answer_scores, latencies = [], [], [], [], []
    details = []
    for case in queries:
        role = case.get("role")
        if role not in ROLES:
            raise ValueError(f"Unknown role in evaluation case: {role}")
        allowed = set(allowed_access_levels(role))
        authorized = [row for row in corpus if isinstance(row, dict)
                      and row.get("access_level") in allowed
                      and isinstance(row.get("id"), str)
                      and isinstance(row.get("text"), str)]
        started = time.perf_counter()
        ranked = bm25_rank(case["question"], authorized, limit=k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        ranked_ids = [row["id"] for row in ranked]
        expected = set(case.get("relevant_ids", [])) & {row["id"] for row in authorized}
        hits = [index for index, doc_id in enumerate(ranked_ids, 1) if doc_id in expected]
        recall = len(set(ranked_ids) & expected) / len(expected) if expected else None
        rr = 1 / hits[0] if hits else 0.0
        if expected:
            recalls.append(recall)
            reciprocal_ranks.append(rr)
        coverage = reference_evidence_coverage(case.get("reference_answer", ""),
            [row["text"] for row in ranked])
        if coverage is not None:
            evidence_scores.append(coverage)
        answer_support = answer_evidence_support(case.get("candidate_answer", ""),
            [row["text"] for row in ranked])
        if answer_support is not None:
            answer_scores.append(answer_support)
        details.append({"question": case["question"], "role": role, "ranked_ids": ranked_ids,
                        "expected_ids": sorted(expected), "recall_at_k": recall,
                        "reciprocal_rank": rr, "query_latency_ms": round(elapsed_ms, 3),
                        "reference_answer_evidence_coverage_proxy": coverage,
                        "candidate_answer_evidence_support_proxy": answer_support})
    return {
        "mode": "local_bm25_synthetic_or_user_supplied_corpus",
        "k": k,
        "queries": len(details),
        "queries_with_authorized_relevant_documents": len(recalls),
        "recall_at_k": sum(recalls) / len(recalls) if recalls else None,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None,
        "mean_query_latency_ms": sum(latencies) / len(latencies),
        "mean_stage_latency_ms": {"bm25_search": sum(latencies) / len(latencies)},
        "mean_reference_answer_evidence_coverage_proxy": (
            sum(evidence_scores) / len(evidence_scores) if evidence_scores else None),
        "mean_candidate_answer_evidence_support_proxy": (
            sum(answer_scores) / len(answer_scores) if answer_scores else None),
        "details": details,
    }


def run_live(dataset: dict, k: int) -> dict:
    """Evaluate the configured Qdrant + embedding retrieval path (no LLM calls)."""
    if not isinstance(dataset.get("queries"), list) or not dataset["queries"]:
        raise ValueError("Dataset requires a non-empty 'queries' array")
    backend = Path(__file__).resolve().parent
    sys.path.insert(0, str(backend))
    import main
    from observability import stage_delta, stage_snapshot

    before_stages = stage_snapshot()
    recalls, reciprocal_ranks, evidence_scores, answer_scores, latencies = [], [], [], [], []
    details = []
    for case in dataset["queries"]:
        role = case.get("role")
        if role not in ROLES:
            raise ValueError(f"Unknown role in evaluation case: {role}")
        started = time.perf_counter()
        results = main.cached_hybrid_retrieve(case["question"], role)
        elapsed_ms = (time.perf_counter() - started) * 1000
        latencies.append(elapsed_ms)
        ranked_ids = list(dict.fromkeys(point.payload.get("document_id") for point in results
                                        if isinstance(point.payload.get("document_id"), str)))[:k]
        expected = set(case.get("relevant_ids", []))
        hits = [index for index, doc_id in enumerate(ranked_ids, 1) if doc_id in expected]
        recall = len(set(ranked_ids) & expected) / len(expected) if expected else None
        rr = 1 / hits[0] if hits else 0.0
        if expected:
            recalls.append(recall)
            reciprocal_ranks.append(rr)
        texts = [point.payload["text"] for point in results]
        coverage = reference_evidence_coverage(case.get("reference_answer", ""), texts)
        if coverage is not None:
            evidence_scores.append(coverage)
        answer_support = answer_evidence_support(case.get("candidate_answer", ""), texts)
        if answer_support is not None:
            answer_scores.append(answer_support)
        details.append({"question": case["question"], "role": role, "ranked_ids": ranked_ids,
                        "expected_ids": sorted(expected), "recall_at_k": recall,
                        "reciprocal_rank": rr, "query_latency_ms": round(elapsed_ms, 3),
                        "reference_answer_evidence_coverage_proxy": coverage,
                        "candidate_answer_evidence_support_proxy": answer_support})
    count = len(details)
    measured_stages = stage_delta(before_stages, stage_snapshot())
    mean_stages = {stage.rsplit(":", 1)[0]: value["mean_ms"]
                   for stage, value in measured_stages.items() if stage.endswith(":success")}
    return {"mode": "live_qdrant_hybrid_no_llm", "k": k, "queries": count,
            "queries_with_authorized_relevant_documents": len(recalls),
            "recall_at_k": sum(recalls) / len(recalls) if recalls else None,
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None,
            "mean_query_latency_ms": sum(latencies) / count,
            "mean_stage_latency_ms": mean_stages,
            "mean_reference_answer_evidence_coverage_proxy": (
                sum(evidence_scores) / len(evidence_scores) if evidence_scores else None),
            "mean_candidate_answer_evidence_support_proxy": (
                sum(answer_scores) / len(answer_scores) if answer_scores else None),
            "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Backend/eval/sample_dataset.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--live", action="store_true", help="Use configured Qdrant and embedding service")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    result = run_live(dataset, args.k) if args.live else run_local(dataset, args.k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
