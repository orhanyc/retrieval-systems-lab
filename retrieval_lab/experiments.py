from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
from uuid import uuid4

from .models import EvaluationResult


class ExperimentRegistry:
    def __init__(self, path: str | Path = "artifacts/experiments.jsonl") -> None:
        self.path = Path(path)

    def record(self, result: EvaluationResult, corpus_hash: str) -> dict:
        entry = {"experiment_id": str(uuid4()), "timestamp": datetime.now(timezone.utc).isoformat(), "corpus_hash": corpus_hash, "result": asdict(result)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def list(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def leaderboard(self, metric: str = "ndcg_at_k") -> list[dict]:
        entries = self.list()
        return sorted(entries, key=lambda entry: entry["result"].get(metric, 0), reverse=True)
