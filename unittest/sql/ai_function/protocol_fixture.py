#!/usr/bin/env python3
"""Local protocol fixture, not the organizer's Mock AI Service.

All requests use synthetic data. Keep authorization values out of the log.
OpenAI embeddings deliberately return valid data in reverse index order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def vector(text, dimensions):
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [int.from_bytes(digest[2 * (i % 16):2 * (i % 16) + 2], "big") / 8.0 for i in range(dimensions)]


def rerank_score(text, query):
    match = re.search(r"score:([-+]?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else float(query.casefold() in text.casefold())


def normal_response(path, body):
    if path.endswith("/chat/completions"):
        messages = body.get("messages")
        assert isinstance(messages, list) and messages
        prompt = messages[-1]["content"]
        assert isinstance(prompt, str)
        return {"choices": [{"message": {"role": "assistant", "content": "echo:" + prompt}}]}
    if path.endswith("/api/generate"):
        assert body.get("stream") is False, "Ollama generation must be explicitly non-streaming"
        assert isinstance(body.get("prompt"), str)
        return {"model": body["model"], "response": "echo:" + body["prompt"], "done": True}
    if path.endswith("/embeddings") or path.endswith("/api/embed"):
        values = body.get("input")
        assert isinstance(values, list) and values and all(isinstance(v, str) for v in values)
        dimensions = body.get("dimensions", 768 if body["model"] == "text-embedding-v1" else 7)
        assert type(dimensions) is int and dimensions > 0
        vectors = [vector(value, dimensions) for value in values]
        if path.endswith("/api/embed"):
            return {"model": body["model"], "embeddings": vectors}
        return {"object": "list", "data": [
            {"object": "embedding", "index": index, "embedding": vectors[index]}
            for index in reversed(range(len(values)))
        ]}
    if path.endswith("/rerank"):
        documents = body.get("documents")
        assert isinstance(documents, list) and all(isinstance(v, str) for v in documents)
        query = body["query"]
        scores = [rerank_score(text, query) for text in documents]
        # Equal-score records are intentionally in reverse local index order.
        indices = sorted(range(len(documents)), key=lambda i: (-scores[i], -i))
        return {"results": [{"index": i, "relevance_score": scores[i]} for i in indices]}
    raise ValueError("unknown fixture path " + path)


def respond(path, body):
    """Faults belong to this independent fixture, never to product code."""
    result = normal_response(path, body)
    if not path.startswith('/fault/'):
        return result
    fault = path.split('/')[2]
    if fault == 'root_array':
        return []
    if path.endswith('/chat/completions'):
        if fault == 'missing':
            return {}
        result['choices'][0]['message']['content'] = {'empty': '', 'nonstring': 23}[fault]
    elif path.endswith('/embeddings'):
        data = result['data']
        if fault == 'missing':
            return {}
        if fault == 'count':
            data.pop()
        elif fault == 'duplicate':
            data.append(data[0])
        elif fault == 'index':
            data[0]['index'] = -1
        elif fault == 'index_type':
            data[0]['index'] = '0'
        elif fault == 'empty':
            data[0]['embedding'] = []
        elif fault == 'element':
            data[0]['embedding'][0] = 'not-a-number'
        elif fault == 'boolean':
            data[0]['embedding'][0] = True
        elif fault == 'dimension':
            data[0]['embedding'].append(1.0)
        elif fault == 'mixed_dimension':
            if len(data) > 1:
                data[0]['embedding'].append(1.0)
    elif path.endswith('/rerank'):
        data = result['results']
        if fault == 'missing':
            data[0].pop('index')
        elif fault == 'count':
            data.pop()
        elif fault == 'index':
            data[0]['index'] = len(data)
        elif fault == 'duplicate':
            data[0]['index'] = data[-1]['index']
        elif fault == 'score':
            data[0]['relevance_score'] = True
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *values):
            pass

        def do_POST(self):
            start = time.monotonic()
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = None
            path = urlsplit(self.path).path
            status = 200
            try:
                body = json.loads(raw)
                assert isinstance(body, dict)
                assert isinstance(body.get("model"), str) and body["model"]
                if path.startswith('/fault/http403/'):
                    status, result = 403, {'fixture_error': 'synthetic forbidden'}
                else:
                    result = respond(path, body)
            except Exception as exc:
                status = 400
                result = {"fixture_error": str(exc)}
            payload = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            with lock, args.log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "at_unix": time.time(), "path": path, "body": body,
                    "has_authorization": bool(self.headers.get("Authorization")),
                    "content_type": self.headers.get("Content-Type"),
                    "status": status, "response": result,
                    "response_sha256": hashlib.sha256(payload).hexdigest(),
                    "elapsed_s": time.monotonic() - start,
                }, ensure_ascii=False) + "\n")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Local protocol fixture listening on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
