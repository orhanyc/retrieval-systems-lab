from __future__ import annotations

import random
import re
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class RobustnessCase:
    original: str
    perturbed: str
    perturbation: str


@dataclass(frozen=True)
class RobustnessResult:
    perturbation: str
    cases: int
    top_document_agreement: float
    result_overlap: float


class QueryPerturber:
    def __init__(self, seed: int = 42) -> None:
        self.random = random.Random(seed)
        self.replacements = {
            "evaluate": "measure",
            "safe": "reliable",
            "combine": "merge",
            "model": "system",
            "search": "retrieval",
            "track": "monitor",
        }

    def generate(self, query: str) -> list[RobustnessCase]:
        cases = []
        words = query.split()
        if len(words) > 3:
            removed = words[:]
            removed.pop(self.random.randrange(len(removed)))
            cases.append(RobustnessCase(query, " ".join(removed), "word_deletion"))
        swapped = words[:]
        if len(swapped) > 3:
            index = self.random.randrange(len(swapped) - 1)
            swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
            cases.append(RobustnessCase(query, " ".join(swapped), "word_swap"))
        replaced = [self.replacements.get(re.sub(r"[^a-z]", "", word.lower()), word) for word in words]
        if replaced != words:
            cases.append(RobustnessCase(query, " ".join(replaced), "synonym"))
        cases.append(RobustnessCase(query, query.lower().replace("?", ""), "normalization"))
        return cases


class RobustnessEvaluator:
    def __init__(self, perturber: QueryPerturber | None = None) -> None:
        self.perturber = perturber or QueryPerturber()

    def evaluate(self, retriever, queries: list[str], k: int = 5) -> list[RobustnessResult]:
        grouped: dict[str, list[tuple[bool, float]]] = {}
        for query in queries:
            baseline = retriever.search(query, k)
            baseline_ids = [result.chunk.document_id for result in baseline]
            for case in self.perturber.generate(query):
                results = retriever.search(case.perturbed, k)
                result_ids = [result.chunk.document_id for result in results]
                top_agreement = bool(baseline_ids and result_ids and baseline_ids[0] == result_ids[0])
                overlap = len(set(baseline_ids) & set(result_ids)) / max(1, len(set(baseline_ids) | set(result_ids)))
                grouped.setdefault(case.perturbation, []).append((top_agreement, overlap))
        return [
            RobustnessResult(name, len(values), sum(top for top, _ in values) / len(values), sum(overlap for _, overlap in values) / len(values))
            for name, values in sorted(grouped.items())
        ]

    def report(self, retriever, queries: list[str], k: int = 5) -> dict:
        results = self.evaluate(retriever, queries, k)
        return {"results": [asdict(result) for result in results], "mean_top_document_agreement": sum(result.top_document_agreement for result in results) / max(1, len(results))}
