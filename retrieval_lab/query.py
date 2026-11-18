from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPlan:
    original: str
    normalized: str
    expansions: list[str]
    subqueries: list[str]
    filters: dict[str, str]


class QueryPlanner:
    SYNONYMS = {
        "rag": ["retrieval augmented generation", "grounded generation"],
        "search": ["retrieval", "ranking"],
        "metrics": ["evaluation", "recall", "ndcg"],
        "safe": ["rollback", "canary", "reliability"],
        "semantic": ["dense", "embedding"],
    }

    def plan(self, query: str) -> QueryPlan:
        normalized = " ".join(re.findall(r"[a-z0-9]+", query.lower()))
        words = normalized.split()
        expansions = []
        for word in words:
            expansions.extend(self.SYNONYMS.get(word, []))
        filters = self._filters(query)
        subqueries = self._decompose(normalized)
        return QueryPlan(query, normalized, list(dict.fromkeys(expansions)), subqueries, filters)

    def expanded_query(self, plan: QueryPlan) -> str:
        return " ".join([plan.normalized, *plan.expansions])

    def _decompose(self, normalized: str) -> list[str]:
        separators = (" and ", " versus ", " vs ", " compared with ")
        for separator in separators:
            if separator in normalized:
                parts = [part.strip() for part in normalized.split(separator) if part.strip()]
                if len(parts) > 1:
                    return parts
        return [normalized]

    def _filters(self, query: str) -> dict[str, str]:
        filters = {}
        source = re.search(r"source:([\w:/.-]+)", query, re.I)
        version = re.search(r"version:([\w.-]+)", query, re.I)
        if source:
            filters["source"] = source.group(1)
        if version:
            filters["version"] = version.group(1)
        return filters
