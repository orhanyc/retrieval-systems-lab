from __future__ import annotations

from .corpus import Chunker, Corpus
from .generation import GroundedAnswerGenerator
from .reranker import PairwiseReranker
from .retrievers import BM25Retriever, HashingDenseRetriever, HybridRetriever


class RetrievalPipeline:
    def __init__(self, corpus: Corpus, chunk_size: int = 90, overlap: int = 20) -> None:
        self.corpus = corpus
        self.chunker = Chunker(chunk_size, overlap)
        self.chunks = self.chunker.split(corpus)
        self.bm25 = BM25Retriever()
        self.dense = HashingDenseRetriever()
        self.hybrid = HybridRetriever(self.bm25, self.dense)
        self.hybrid.index(self.chunks)
        self.reranker = PairwiseReranker()
        self.generator = GroundedAnswerGenerator()

    def search(self, query: str, method: str = "hybrid", top_k: int = 5, rerank: bool = True):
        retriever = self._retriever(method)
        candidates = retriever.search(query, max(top_k, 15 if rerank else top_k))
        return self.reranker.rerank(query, candidates, top_k) if rerank else candidates[:top_k]

    def answer(self, query: str, method: str = "hybrid", top_k: int = 5, rerank: bool = True) -> dict:
        results = self.search(query, method, top_k, rerank)
        answer = self.generator.generate(query, results)
        return {"query": query, "answer": answer.__dict__, "results": [result.to_dict() for result in results], "corpus_version": self.corpus.version_hash}

    def _retriever(self, method: str):
        retrievers = {"bm25": self.bm25, "dense": self.dense, "hybrid": self.hybrid}
        if method not in retrievers:
            raise ValueError(f"unknown retrieval method: {method}")
        return retrievers[method]
