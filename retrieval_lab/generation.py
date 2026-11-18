from __future__ import annotations

import re

from .models import Answer, SearchResult


class GroundedAnswerGenerator:
    def __init__(self, minimum_score: float = 0.01, maximum_sentences: int = 3) -> None:
        self.minimum_score = minimum_score
        self.maximum_sentences = maximum_sentences

    def generate(self, query: str, results: list[SearchResult]) -> Answer:
        eligible = [result for result in results if result.score >= self.minimum_score]
        if not eligible:
            return Answer("Insufficient evidence in the indexed corpus.", [], False, 0, True)
        query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        candidates = []
        for result in eligible:
            for sentence in re.split(r"(?<=[.!?])\s+", result.chunk.text):
                terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                overlap = len(query_terms & terms) / max(1, len(query_terms))
                candidates.append((overlap + result.score, sentence.strip(), result.chunk.chunk_id))
        selected = sorted(candidates, reverse=True)[: self.maximum_sentences]
        if not selected or selected[0][0] <= 0:
            return Answer("Insufficient evidence in the indexed corpus.", [], False, 0, True)
        citations = []
        sentences = []
        for _, sentence, chunk_id in selected:
            if sentence and sentence not in sentences:
                sentences.append(sentence)
                citations.append(chunk_id)
        confidence = min(1, sum(score for score, _, _ in selected) / max(1, len(selected)))
        return Answer(" ".join(sentences), list(dict.fromkeys(citations)), True, confidence, False)

    def validate_citations(self, answer: Answer, results: list[SearchResult]) -> bool:
        available = {result.chunk.chunk_id for result in results}
        return bool(answer.citations) and all(citation in available for citation in answer.citations)
