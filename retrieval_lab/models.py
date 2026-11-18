from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    text: str
    source: str
    version: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    title: str
    text: str
    position: int
    start: int
    end: int
    source: str
    version: str


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    rank: int
    method: str
    component_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return value


@dataclass(frozen=True)
class QueryCase:
    query_id: str
    query: str
    relevant_documents: tuple[str, ...]
    expected_answer_terms: tuple[str, ...]


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[str]
    grounded: bool
    confidence: float
    abstained: bool


@dataclass
class EvaluationResult:
    method: str
    recall_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float
    answer_correctness: float
    groundedness: float
    citation_correctness: float
    mean_latency_ms: float
    queries: int
    configuration: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
