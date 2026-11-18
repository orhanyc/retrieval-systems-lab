from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict

from .models import Chunk, SearchResult


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return words + [f"{left}_{right}" for left, right in zip(words, words[1:])]


class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequency: Counter[str] = Counter()
        self.average_length = 0.0

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self.term_frequencies = [Counter(tokens(chunk.title + " " + chunk.text)) for chunk in chunks]
        self.document_frequency = Counter()
        for frequencies in self.term_frequencies:
            self.document_frequency.update(frequencies.keys())
        self.average_length = sum(sum(frequencies.values()) for frequencies in self.term_frequencies) / max(1, len(chunks))

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        query_terms = Counter(tokens(query))
        scored = []
        for chunk, frequencies in zip(self.chunks, self.term_frequencies):
            length = sum(frequencies.values())
            score = 0.0
            for term, query_frequency in query_terms.items():
                frequency = frequencies[term]
                if not frequency:
                    continue
                document_frequency = self.document_frequency[term]
                inverse = math.log(1 + (len(self.chunks) - document_frequency + 0.5) / (document_frequency + 0.5))
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / max(1, self.average_length))
                score += query_frequency * inverse * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scored.append((chunk, score))
        ranked = sorted(scored, key=lambda item: (-item[1], item[0].chunk_id))[:limit]
        return [SearchResult(chunk, score, rank, self.name, {self.name: score}) for rank, (chunk, score) in enumerate(ranked, start=1)]


class HashingDenseRetriever:
    name = "dense"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []
        self.synonyms = {
            "leakage": ["future", "point", "time"],
            "semantic": ["dense", "meaning"],
            "lexical": ["bm25", "keyword"],
            "rollback": ["revert", "canary"],
            "evidence": ["citation", "grounded"],
        }

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = list(chunks)
        self.vectors = [self._vector(chunk.title + " " + chunk.text) for chunk in chunks]

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        query_vector = self._vector(query)
        scored = [(chunk, self._dot(query_vector, vector)) for chunk, vector in zip(self.chunks, self.vectors)]
        ranked = sorted((item for item in scored if item[1] > 0), key=lambda item: (-item[1], item[0].chunk_id))[:limit]
        return [SearchResult(chunk, score, rank, self.name, {self.name: score}) for rank, (chunk, score) in enumerate(ranked, start=1)]

    def _vector(self, text: str) -> list[float]:
        expanded = tokens(text)
        for term in list(expanded):
            expanded.extend(self.synonyms.get(term, []))
        vector = [0.0] * self.dimensions
        for term in expanded:
            digest = hashlib.blake2b(term.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            sign = 1 if (value >> 8) & 1 else -1
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1
        return [value / norm for value in vector]

    def _dot(self, left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))


class HybridRetriever:
    name = "hybrid"

    def __init__(self, sparse: BM25Retriever, dense: HashingDenseRetriever, fusion_constant: int = 60) -> None:
        self.sparse = sparse
        self.dense = dense
        self.fusion_constant = fusion_constant

    def index(self, chunks: list[Chunk]) -> None:
        self.sparse.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        candidates = max(limit * 3, 20)
        lists = [self.sparse.search(query, candidates), self.dense.search(query, candidates)]
        scores: dict[str, float] = defaultdict(float)
        components: dict[str, dict[str, float]] = defaultdict(dict)
        chunks: dict[str, Chunk] = {}
        for results in lists:
            for result in results:
                chunks[result.chunk.chunk_id] = result.chunk
                scores[result.chunk.chunk_id] += 1 / (self.fusion_constant + result.rank)
                components[result.chunk.chunk_id].update(result.component_scores)
        ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:limit]
        return [SearchResult(chunks[chunk_id], scores[chunk_id], rank, self.name, components[chunk_id]) for rank, chunk_id in enumerate(ranked, start=1)]
