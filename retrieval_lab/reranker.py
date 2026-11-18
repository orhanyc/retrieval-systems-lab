from __future__ import annotations

from .models import SearchResult
from .retrievers import tokens


class PairwiseReranker:
    def __init__(self, title_weight: float = 0.25, phrase_weight: float = 0.35) -> None:
        self.title_weight = title_weight
        self.phrase_weight = phrase_weight

    def rerank(self, query: str, results: list[SearchResult], limit: int | None = None) -> list[SearchResult]:
        query_terms = set(tokens(query))
        query_phrase = " ".join(tokens(query)[:4])
        scored = []
        for result in results:
            body_terms = set(tokens(result.chunk.text))
            title_terms = set(tokens(result.chunk.title))
            overlap = len(query_terms & body_terms) / max(1, len(query_terms))
            title_overlap = len(query_terms & title_terms) / max(1, len(query_terms))
            phrase = self.phrase_weight if query_phrase and query_phrase in result.chunk.text.lower() else 0
            rerank_score = result.score + overlap + self.title_weight * title_overlap + phrase
            components = {**result.component_scores, "reranker": rerank_score}
            scored.append((result, rerank_score, components))
        ranked = sorted(scored, key=lambda item: (-item[1], item[0].chunk.chunk_id))[: limit or len(scored)]
        return [SearchResult(item.chunk, score, rank, item.method + "+rerank", components) for rank, (item, score, components) in enumerate(ranked, start=1)]
