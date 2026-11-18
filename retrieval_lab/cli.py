from __future__ import annotations

import json

from .corpus import sample_corpus
from .evaluation import RetrievalEvaluator, sample_queries
from .experiments import ExperimentRegistry
from .pipeline import RetrievalPipeline


def main() -> None:
    corpus = sample_corpus()
    pipeline = RetrievalPipeline(corpus)
    evaluator = RetrievalEvaluator(pipeline.generator)
    registry = ExperimentRegistry()
    results = []
    for method in ("bm25", "dense", "hybrid"):
        retriever = pipeline._retriever(method)
        evaluation = evaluator.evaluate(method, retriever, sample_queries(), k=5, reranker=pipeline.reranker if method == "hybrid" else None, configuration={"chunk_size": 90, "overlap": 20, "reranker": method == "hybrid"})
        registry.record(evaluation, corpus.version_hash)
        results.append(evaluation.to_dict())
    example = pipeline.answer("How does hybrid retrieval combine rankings?")
    print(json.dumps({"evaluations": results, "example": example}, indent=2, default=str))


if __name__ == "__main__":
    main()
