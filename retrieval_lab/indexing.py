from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any

from .models import Chunk


@dataclass(frozen=True)
class IndexVersion:
    version_id: str
    created_at: str
    corpus_hash: str
    configuration: dict[str, Any]
    chunk_count: int
    checksum: str
    status: str


class IndexRegistry:
    def __init__(self) -> None:
        self.versions: list[IndexVersion] = []
        self.active_version: str | None = None
        self.lock = RLock()

    def register(self, corpus_hash: str, chunks: list[Chunk], configuration: dict[str, Any]) -> IndexVersion:
        checksum = self._chunk_checksum(chunks)
        version_id = hashlib.sha256(f"{corpus_hash}:{checksum}:{json.dumps(configuration, sort_keys=True)}".encode()).hexdigest()[:16]
        with self.lock:
            existing = next((version for version in self.versions if version.version_id == version_id), None)
            if existing:
                return existing
            version = IndexVersion(version_id, datetime.now(timezone.utc).isoformat(), corpus_hash, dict(configuration), len(chunks), checksum, "candidate")
            self.versions.append(version)
            return version

    def activate(self, version_id: str) -> IndexVersion:
        with self.lock:
            target = self.get(version_id)
            updated = []
            for version in self.versions:
                status = "active" if version.version_id == version_id else "archived" if version.status == "active" else version.status
                updated.append(IndexVersion(**{**asdict(version), "status": status}))
            self.versions = updated
            self.active_version = version_id
            return self.get(version_id)

    def get(self, version_id: str) -> IndexVersion:
        version = next((item for item in self.versions if item.version_id == version_id), None)
        if version is None:
            raise KeyError(f"unknown index version: {version_id}")
        return version

    def active(self) -> IndexVersion | None:
        return self.get(self.active_version) if self.active_version else None

    def compare(self, left_id: str, right_id: str) -> dict[str, Any]:
        left, right = self.get(left_id), self.get(right_id)
        changed_configuration = {
            key: {"left": left.configuration.get(key), "right": right.configuration.get(key)}
            for key in sorted(set(left.configuration) | set(right.configuration))
            if left.configuration.get(key) != right.configuration.get(key)
        }
        return {
            "left": asdict(left),
            "right": asdict(right),
            "same_corpus": left.corpus_hash == right.corpus_hash,
            "same_chunks": left.checksum == right.checksum,
            "chunk_count_delta": right.chunk_count - left.chunk_count,
            "configuration_changes": changed_configuration,
        }

    def verify(self, version_id: str, chunks: list[Chunk]) -> bool:
        return self.get(version_id).checksum == self._chunk_checksum(chunks)

    def list(self, status: str | None = None) -> list[IndexVersion]:
        with self.lock:
            return [version for version in self.versions if status is None or version.status == status]

    def _chunk_checksum(self, chunks: list[Chunk]) -> str:
        payload = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "start": chunk.start,
                "end": chunk.end,
                "text": chunk.text,
            }
            for chunk in sorted(chunks, key=lambda item: item.chunk_id)
        ]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class AtomicIndexSwap:
    def __init__(self, registry: IndexRegistry) -> None:
        self.registry = registry
        self.indexes: dict[str, object] = {}
        self.lock = RLock()

    def stage(self, version: IndexVersion, index: object) -> None:
        with self.lock:
            self.indexes[version.version_id] = index

    def promote(self, version_id: str) -> object:
        with self.lock:
            if version_id not in self.indexes:
                raise KeyError("index was not staged")
            self.registry.activate(version_id)
            return self.indexes[version_id]

    def current(self) -> object | None:
        with self.lock:
            return self.indexes.get(self.registry.active_version) if self.registry.active_version else None
