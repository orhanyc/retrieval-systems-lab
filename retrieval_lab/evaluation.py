from __future__ import annotations

import math
import time
from statistics import mean

from .generation import GroundedAnswerGenerator
from .models import EvaluationResult, QueryCase


class RetrievalEvaluator:
    def __init__(self, generator: GroundedAnswerGenerator | None = None) -> None:
        self.generator = generator or GroundedAnswerGenerator()

    def evaluate(self, method: str, retriever, cases: list[QueryCase], k: int = 5, reranker=None, configuration: dict | None = None) -> EvaluationResult:
        recalls, reciprocal_ranks, ndcgs = [], [], []
        answer_scores, grounded, citations, latencies = [], [], [], []
        for case in cases:
            started = time.perf_counter()
            results = retriever.search(case.query, max(k, 15 if reranker else k))
            if reranker:
                results = reranker.rerank(case.query, results, k)
            else:
                results = results[:k]
            answer = self.generator.generate(case.query, results)
            latencies.append((time.perf_counter() - started) * 1000)
            documents = [result.chunk.document_id for result in results]
            relevant = set(case.relevant_documents)
            retrieved_relevant = [index for index, document in enumerate(documents, start=1) if document in relevant]
            recalls.append(len(set(documents) & relevant) / max(1, len(relevant)))
            reciprocal_ranks.append(1 / retrieved_relevant[0] if retrieved_relevant else 0)
            ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
            actual = sum(1 / math.log2(rank + 1) for rank in retrieved_relevant)
            ndcgs.append(actual / max(ideal, 1e-9))
            normalized_answer = answer.text.lower()
            answer_scores.append(sum(term.lower() in normalized_answer for term in case.expected_answer_terms) / max(1, len(case.expected_answer_terms)))
            grounded.append(float(answer.grounded and not answer.abstained))
            citations.append(float(self.generator.validate_citations(answer, results)))
        return EvaluationResult(method, mean(recalls), mean(reciprocal_ranks), mean(ndcgs), mean(answer_scores), mean(grounded), mean(citations), mean(latencies), len(cases), dict(configuration or {}))


def sample_queries() -> list[QueryCase]:
    return [
        QueryCase("q1", "How do point-in-time joins prevent leakage?", ("features",), ("latest", "before", "label")),
        QueryCase("q2", "What metrics evaluate ranked retrieval?", ("evaluation",), ("recall", "reciprocal", "ndcg")),
        QueryCase("q3", "How should a challenger model be released safely?", ("canary",), ("traffic", "promote", "roll back")),
        QueryCase("q4", "Why combine dense and lexical search?", ("hybrid",), ("bm25", "dense", "fusion")),
        QueryCase("q5", "What should production RAG monitoring track?", ("observability",), ("latency", "citation", "groundedness")),
        QueryCase("q6", "How does reranking improve retrieval?", ("reranking",), ("query", "passage", "precision")),
    ]
