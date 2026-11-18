from __future__ import annotations

from dataclasses import dataclass

from .models import SearchResult


@dataclass(frozen=True)
class ResultPolicy:
    allowed_sources: tuple[str, ...] = ()
    required_version: str | None = None
    minimum_score: float = 0
    maximum_per_document: int = 2


class ResultFilter:
    def apply(self, results: list[SearchResult], policy: ResultPolicy, limit: int | None = None) -> list[SearchResult]:
        selected = []
        per_document: dict[str, int] = {}
        for result in results:
            if result.score < policy.minimum_score:
                continue
            if policy.allowed_sources and result.chunk.source not in policy.allowed_sources:
                continue
            if policy.required_version and result.chunk.version != policy.required_version:
                continue
            count = per_document.get(result.chunk.document_id, 0)
            if count >= policy.maximum_per_document:
                continue
            per_document[result.chunk.document_id] = count + 1
            selected.append(result)
            if limit is not None and len(selected) >= limit:
                break
        return [
            SearchResult(item.chunk, item.score, rank, item.method, item.component_scores)
            for rank, item in enumerate(selected, start=1)
        ]

    def explain(self, original: list[SearchResult], filtered: list[SearchResult]) -> dict:
        original_ids = {result.chunk.chunk_id for result in original}
        filtered_ids = {result.chunk.chunk_id for result in filtered}
        return {
            "original_results": len(original),
            "filtered_results": len(filtered),
            "removed_chunk_ids": sorted(original_ids - filtered_ids),
            "retained_chunk_ids": [result.chunk.chunk_id for result in filtered],
        }
