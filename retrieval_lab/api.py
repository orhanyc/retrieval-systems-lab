from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .corpus import sample_corpus
from .pipeline import RetrievalPipeline


class RetrievalHandler(BaseHTTPRequestHandler):
    pipeline = RetrievalPipeline(sample_corpus())

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/health":
                self._send(200, {"status": "ok", "documents": len(self.pipeline.corpus.documents), "chunks": len(self.pipeline.chunks), "corpus_version": self.pipeline.corpus.version_hash})
            elif parsed.path == "/search":
                phrase = query.get("q", [""])[0]
                if not phrase:
                    raise ValueError("q is required")
                method = query.get("method", ["hybrid"])[0]
                limit = int(query.get("top_k", ["5"])[0])
                results = self.pipeline.search(phrase, method, limit)
                self._send(200, {"results": [result.to_dict() for result in results]})
            elif parsed.path == "/answer":
                phrase = query.get("q", [""])[0]
                if not phrase:
                    raise ValueError("q is required")
                self._send(200, self.pipeline.answer(phrase, query.get("method", ["hybrid"])[0]))
            else:
                self._send(404, {"error": "not found"})
        except ValueError as error:
            self._send(400, {"error": str(error)})

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def serve() -> None:
    host = os.getenv("RETRIEVAL_HOST", "0.0.0.0")
    port = int(os.getenv("RETRIEVAL_PORT", "8080"))
    print(f"Retrieval Systems Lab listening on http://{host}:{port}")
    ThreadingHTTPServer((host, port), RetrievalHandler).serve_forever()
