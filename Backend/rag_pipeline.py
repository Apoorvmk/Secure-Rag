"""Authorized lexical/vector retrieval and rank fusion helpers."""

from collections import Counter
from math import isfinite, log
import re
from types import SimpleNamespace
import time

from access_policy import authorized_qdrant_filter
from document_pipeline import COLLECTION_NAME, filter_authorized_results
from observability import record_stage

TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
STOPWORDS = frozenset("a an and are as at be by does for from get how i in is it of on or the they this to what when where who why with you your".split())
RRF_K = 60
MAX_LEXICAL_CANDIDATES = 5000


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)
            if token.casefold() not in STOPWORDS]


def authorized_filter(role: str):
    """One server-side filter shared by semantic and lexical retrieval."""
    return authorized_qdrant_filter(role)


def bm25_rank(question: str, points: list, limit: int = 30) -> list:
    """Rank an already-authorized corpus with BM25 (Okapi, k1=1.5, b=.75)."""
    terms = tokenize(question)
    if not terms or not points:
        return []
    docs = [tokenize(point.get("text", "") if isinstance(point, dict)
                     else point.payload.get("text", "")) for point in points]
    average_length = sum(map(len, docs)) / max(len(docs), 1)
    document_frequency = Counter(term for doc in docs for term in set(doc))
    scored = []
    for point, tokens in zip(points, docs):
        frequencies = Counter(tokens)
        score = 0.0
        for term in terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            idf = log(1 + (len(docs) - document_frequency[term] + 0.5) /
                      (document_frequency[term] + 0.5))
            denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * len(tokens) / max(average_length, 1))
            score += idf * frequency * 2.5 / denominator
        if score > 0:
            scored.append((score, point))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [point for _, point in scored[:limit]]


def reciprocal_rank_fusion(*ranked_lists: list, limit: int = 8, k: int = RRF_K) -> list:
    """Fuse ranks by stable point ID; keep original validated payloads."""
    fused = {}
    points = {}
    for ranked in ranked_lists:
        for rank, point in enumerate(ranked, 1):
            identifier = str(getattr(point, "id", ""))
            if not identifier:
                continue
            fused[identifier] = fused.get(identifier, 0.0) + 1 / (k + rank)
            points[identifier] = point
    ordered = sorted(fused, key=lambda identifier: fused[identifier], reverse=True)
    output = []
    semantic_scores = {}
    for ranked in ranked_lists:
        for point in ranked:
            identifier = str(getattr(point, "id", ""))
            value = getattr(point, "semantic_score", None)
            if value is None:
                value = getattr(point, "score", None)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            semantic_scores[identifier] = max(semantic_scores.get(identifier, numeric), numeric)
    for identifier in ordered[:limit]:
        try:
            semantic_score = float(semantic_scores.get(identifier))
        except (TypeError, ValueError):
            semantic_score = None
        output.append(SimpleNamespace(id=identifier, payload=points[identifier].payload,
                                      score=fused[identifier], semantic_score=semantic_score))
    return output


def retrieve_hybrid(question: str, *, role: str, qdrant, model, limit: int = 8,
                    reranker=None) -> list:
    """Retrieve with pre-filtered Qdrant vector and authorized BM25 candidates."""
    filt = authorized_filter(role)
    stage_started = time.perf_counter()
    try:
        query_vector = model.encode(question).tolist()
    except Exception:
        record_stage("query_embedding", time.perf_counter() - stage_started, "error")
        raise
    record_stage("query_embedding", time.perf_counter() - stage_started)

    stage_started = time.perf_counter()
    try:
        semantic = qdrant.query_points(collection_name=COLLECTION_NAME, query=query_vector,
                                       query_filter=filt, limit=max(limit * 4, 20)).points
    except Exception:
        record_stage("vector_search", time.perf_counter() - stage_started, "error")
        raise
    record_stage("vector_search", time.perf_counter() - stage_started)
    semantic = filter_authorized_results(semantic, role)

    lexical_corpus = []
    offset = None
    stage_started = time.perf_counter()
    try:
        while len(lexical_corpus) < MAX_LEXICAL_CANDIDATES:
            page, offset = qdrant.scroll(collection_name=COLLECTION_NAME, scroll_filter=filt,
                with_payload=True, with_vectors=False,
                limit=min(1000, MAX_LEXICAL_CANDIDATES - len(lexical_corpus)), offset=offset)
            lexical_corpus.extend(filter_authorized_results(page, role))
            if offset is None:
                break
        lexical = bm25_rank(question, lexical_corpus, limit=max(limit * 4, 20))
    except Exception:
        record_stage("bm25_search", time.perf_counter() - stage_started, "error")
        raise
    record_stage("bm25_search", time.perf_counter() - stage_started)
    fused = reciprocal_rank_fusion(semantic, lexical, limit=max(limit * 4, 20))

    if reranker and fused:
        stage_started = time.perf_counter()
        try:
            pairs = [(question, point.payload["text"]) for point in fused]
            scores = list(reranker.predict(pairs))
            if len(scores) != len(fused) or any(not isfinite(float(score)) for score in scores):
                raise ValueError("Cross-encoder returned invalid scores")
            fused = [point for _, point in sorted(zip(scores, fused), key=lambda pair: float(pair[0]), reverse=True)]
            record_stage("rerank", time.perf_counter() - stage_started)
        except Exception:
            # Reranking is an enhancement; authorized fused retrieval remains valid.
            record_stage("rerank", time.perf_counter() - stage_started, "error")
    return filter_authorized_results(fused[:limit], role)


def rewrite_query(question: str, previous_questions: list[str], llm, *, force: bool = False) -> str | None:
    """Produce a short standalone retrieval query; return None on any uncertainty."""
    contextual = bool(previous_questions and re.search(
        r"\b(it|they|that|those|there|this|these|same|what about|and)\b", question, re.I))
    if not force and not contextual:
        return None
    relevant_history = previous_questions[-3:] if contextual else []
    try:
        response = llm.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": (
                    "Rewrite the latest question as a standalone search query using only the prior user questions "
                    "to resolve references. Do not answer, add facts, or include instructions. Return only the query."
                )},
                {"role": "user", "content": "Prior questions:\n" + "\n".join(relevant_history) +
                 "\nLatest question:\n" + question},
            ],
        )
        rewritten = response.choices[0].message.content
        if not isinstance(rewritten, str):
            return None
        rewritten = " ".join(rewritten.split())
        if not rewritten or len(rewritten) > 500 or rewritten.casefold() == question.casefold():
            return None
        return rewritten
    except Exception:
        return None


def evidence_is_sufficient(question: str, results: list, *, min_query_coverage: float = 0.2) -> bool:
    """Conservative lexical evidence check; empty/weak evidence triggers correction/refusal."""
    terms = set(tokenize(question))
    terms = {term for term in terms if len(term) > 2}
    if not results:
        return False
    semantic_scores = [getattr(point, "semantic_score", None) for point in results[:3]]
    if any(isinstance(score, (float, int)) and score >= 0.3 for score in semantic_scores):
        return True
    if not terms:
        return False
    evidence = set()
    for point in results[:3]:
        payload = getattr(point, "payload", {}) or {}
        evidence.update(tokenize(payload.get("text", "")))
    return len(terms & evidence) / len(terms) >= min_query_coverage
