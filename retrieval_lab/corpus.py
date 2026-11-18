from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .models import Chunk, Document


class Corpus:
    def __init__(self, documents: list[Document]) -> None:
        identifiers = [document.document_id for document in documents]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("document identifiers must be unique")
        self.documents = sorted(documents, key=lambda item: item.document_id)

    @property
    def version_hash(self) -> str:
        payload = [
            {"id": document.document_id, "version": document.version, "text": document.text}
            for document in self.documents
        ]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "Corpus":
        documents = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                documents.append(Document(**json.loads(line)))
        return cls(documents)


class Chunker:
    def __init__(self, size: int = 90, overlap: int = 20) -> None:
        if size <= 0 or overlap < 0 or overlap >= size:
            raise ValueError("invalid chunk size or overlap")
        self.size = size
        self.overlap = overlap

    def split(self, corpus: Corpus) -> list[Chunk]:
        chunks = []
        for document in corpus.documents:
            words = list(re.finditer(r"\S+", document.text))
            step = self.size - self.overlap
            for position, start_word in enumerate(range(0, len(words), step)):
                selected = words[start_word:start_word + self.size]
                if not selected:
                    continue
                start, end = selected[0].start(), selected[-1].end()
                digest = hashlib.sha1(f"{document.document_id}:{start}:{end}:{document.version}".encode()).hexdigest()[:14]
                chunks.append(Chunk(digest, document.document_id, document.title, document.text[start:end], position, start, end, document.source, document.version))
                if start_word + self.size >= len(words):
                    break
        return chunks


def sample_corpus() -> Corpus:
    documents = [
        Document("features", "Point-in-Time Features", "Point-in-time feature joins select the latest feature value available before each label timestamp. This prevents future information from leaking into training data and keeps offline evaluation honest.", "generated://ml-guide", "1"),
        Document("drift", "Model Drift", "Data drift describes changes in input distributions. Prediction drift tracks changes in model scores. Population stability index and distribution distances can trigger investigation before model quality degrades.", "generated://ml-guide", "1"),
        Document("canary", "Canary Deployment", "A canary deployment sends a small deterministic traffic percentage to a challenger model. Teams compare quality, latency and error rates, then promote the challenger or roll back safely.", "generated://ml-guide", "1"),
        Document("rag", "Grounded Generation", "Retrieval augmented generation supplies relevant passages to a generator. Answers should cite the retrieved source, remain supported by evidence and abstain when context is insufficient.", "generated://rag-guide", "1"),
        Document("hybrid", "Hybrid Retrieval", "Hybrid search combines lexical BM25 rankings with semantic dense retrieval. Reciprocal rank fusion is robust because it merges ranked lists without requiring comparable score scales.", "generated://rag-guide", "1"),
        Document("evaluation", "Retrieval Evaluation", "Recall at k measures whether relevant documents appear in the retrieved set. Mean reciprocal rank rewards an early first relevant result. NDCG supports graded relevance and discounts lower ranks.", "generated://rag-guide", "1"),
        Document("reranking", "Cross-Encoder Reranking", "A reranker scores query and passage pairs after first-stage retrieval. It improves precision at the top of the list while keeping compute bounded by reranking only a small candidate set.", "generated://rag-guide", "1"),
        Document("observability", "RAG Observability", "Production retrieval systems track query latency, empty result rate, citation coverage, answer groundedness, index version and cost. Trace identifiers connect retrieval decisions to generated responses.", "generated://rag-guide", "1"),
    ]
    return Corpus(documents)
