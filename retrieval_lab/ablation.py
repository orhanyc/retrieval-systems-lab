from __future__ import annotations

from dataclasses import asdict, dataclass

from .corpus import Chunker, Corpus
from .evaluation import RetrievalEvaluator
from .reranker import PairwiseReranker
from .retrievers import BM25Retriever, HashingDenseRetriever, HybridRetriever


@dataclass(frozen=True)
class AblationRun:
    chunk_size: int
    overlap: int
    method: str
    reranker: bool
    recall_at_k: float
    ndcg_at_k: float
    latency_ms: float
    chunks: int


class AblationRunner:
    def __init__(self, corpus: Corpus, cases) -> None:
        self.corpus = corpus
        self.cases = list(cases)
        self.evaluator = RetrievalEvaluator()

    def run(self, chunk_sizes: list[int] | None = None, overlaps: list[int] | None = None) -> list[AblationRun]:
        runs = []
        for size in chunk_sizes or [50, 90, 140]:
            for overlap in overlaps or [0, 15]:
                if overlap >= size:
                    continue
                chunks = Chunker(size, overlap).split(self.corpus)
                sparse = BM25Retriever()
                dense = HashingDenseRetriever()
                hybrid = HybridRetriever(sparse, dense)
                hybrid.index(chunks)
                for method, retriever in (("bm25", sparse), ("dense", dense), ("hybrid", hybrid)):
                    for use_reranker in (False, True) if method == "hybrid" else (False,):
                        result = self.evaluator.evaluate(
                            method + ("+rerank" if use_reranker else ""), retriever, self.cases,
                            k=5, reranker=PairwiseReranker() if use_reranker else None,
                            configuration={"chunk_size": size, "overlap": overlap},
                        )
                        runs.append(AblationRun(size, overlap, method, use_reranker, result.recall_at_k, result.ndcg_at_k, result.mean_latency_ms, len(chunks)))
        return runs

    def pareto_frontier(self, runs: list[AblationRun]) -> list[AblationRun]:
        frontier = []
        for candidate in runs:
            dominated = any(
                other.ndcg_at_k >= candidate.ndcg_at_k
                and other.latency_ms <= candidate.latency_ms
                and (other.ndcg_at_k > candidate.ndcg_at_k or other.latency_ms < candidate.latency_ms)
                for other in runs
            )
            if not dominated:
                frontier.append(candidate)
        return sorted(frontier, key=lambda item: (-item.ndcg_at_k, item.latency_ms))

    def report(self, runs: list[AblationRun]) -> dict:
        best = max(runs, key=lambda item: (item.ndcg_at_k, -item.latency_ms))
        return {"runs": [asdict(run) for run in runs], "best": asdict(best), "pareto_frontier": [asdict(run) for run in self.pareto_frontier(runs)]}
