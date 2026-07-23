"""Browser UI for the search engines.

Run:   uv run search_web.py
Open:  http://localhost:8000
"""

import json
import logging
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, ".")

from lib.search_utils import load_movies

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("rag.web")

PORT = 8000
HTML_PATH = Path(__file__).parent / "web_ui.html"

_engines: dict = {}

DATASETS = {
    "movies": {
        "data": "data/movies.json",
        "embeddings": "cache/movie_embeddings.npy",
        "chunk_embeddings": "cache/chunk_embeddings.npy",
        "chunk_metadata": "cache/chunk_metadata.json",
        "cache_prefix": "cache/",
    },
    "produce": {
        "data": "data/produce.json",
        "embeddings": "cache/produce_embeddings.npy",
        "chunk_embeddings": "cache/produce_chunk_embeddings.npy",
        "chunk_metadata": "cache/produce_chunk_metadata.json",
        "cache_prefix": "cache/produce_",
        "chunk_size": 3,
        "chunk_overlap": 1,
        "aggregation": "top2",
    },
}


def get_keyword_engine(dataset: str):
    key = (dataset, "keyword")
    if key not in _engines:
        from lib.inverted_index import InvertedIndex

        cfg = DATASETS[dataset]
        index = InvertedIndex(
            data_path=cfg["data"], cache_prefix=cfg["cache_prefix"]
        )
        index.load()
        _engines[key] = index
    return _engines[key]


def get_semantic_engine(dataset: str):
    key = (dataset, "semantic")
    if key not in _engines:
        from lib.semantic_search import SemanticSearch

        cfg = DATASETS[dataset]
        s = SemanticSearch(embeddings_path=cfg["embeddings"])
        s.load_or_create_embeddings(load_movies(cfg["data"]))
        _engines[key] = s
    return _engines[key]


def get_chunked_engine(dataset: str):
    key = (dataset, "chunked")
    if key not in _engines:
        from lib.semantic_search import ChunkedSemanticSearch

        cfg = DATASETS[dataset]
        s = ChunkedSemanticSearch(
            embeddings_path=cfg["embeddings"],
            chunk_embeddings_path=cfg["chunk_embeddings"],
            chunk_metadata_path=cfg["chunk_metadata"],
            chunk_size=cfg.get("chunk_size", 4),
            chunk_overlap=cfg.get("chunk_overlap", 1),
            aggregation=cfg.get("aggregation", "max"),
        )
        s.load_or_create_chunk_embeddings(load_movies(cfg["data"]))
        _engines[key] = s
    return _engines[key]


def get_hybrid_engine(dataset: str):
    key = (dataset, "hybrid")
    if key not in _engines:
        from lib.hybrid_search import HybridSearch

        cfg = DATASETS[dataset]
        if dataset != "movies":
            raise ValueError(
                "hybrid search currently supports only the movies dataset"
            )
        _engines[key] = HybridSearch(load_movies(cfg["data"]))
    return _engines[key]


def matched_words(text: str, query_tokens: set[str]) -> list[str]:
    from lib.keyword_search import preprocess_text, stemmer

    hits = set()
    for word in text.split():
        clean = preprocess_text(word)
        if clean and stemmer.stem(clean) in query_tokens:
            hits.add(clean)
    return sorted(hits)


def search(
    dataset: str, mode: str, query: str, limit: int, rerank: str | None = None
) -> list[dict]:
    if mode == "keyword":
        from lib.keyword_search import tokenize_text

        query_tokens = set(tokenize_text(query))
        hits = get_keyword_engine(dataset).bm25_search(query, limit)
        max_score = max((score for _, score in hits), default=1.0) or 1.0
        return [
            {
                "title": movie["title"],
                "score": round(score / max_score, 4),
                "raw_score": f"BM25 {score:.2f}",
                "snippet": movie["description"][:200],
                "description": movie["description"],
                "matched_chunk": None,
                "highlight": matched_words(
                    f"{movie['title']} {movie['description']}", query_tokens
                ),
            }
            for movie, score in hits
        ]
    if mode == "semantic":
        return [
            {
                "title": r["title"],
                "score": round(float(r["score"]), 4),
                "raw_score": f"cosine {float(r['score']):.4f}",
                "snippet": r["description"][:200],
                "description": r["description"],
                "matched_chunk": None,
                "highlight": [],
            }
            for r in get_semantic_engine(dataset).search(query, limit)
        ]
    if mode == "chunked":
        engine = get_chunked_engine(dataset)
        return [
            {
                "title": r["title"],
                "score": round(float(r["score"]), 4),
                "raw_score": f"cosine {float(r['score']):.4f}",
                "snippet": r["document"],
                "description": engine.documents[r["metadata"]["movie_idx"]][
                    "description"
                ],
                "matched_chunk": r["metadata"]["matched_chunk"],
                "chunk_idx": r["metadata"]["chunk_idx"],
                "total_chunks": r["metadata"]["total_chunks"],
                "highlight": [],
            }
            for r in engine.search_chunks(query, limit)
        ]
    if mode in ("hybrid_weighted", "hybrid_rrf"):
        engine = get_hybrid_engine(dataset)
        if mode == "hybrid_weighted":
            results = engine.weighted_search(query, alpha=0.5, limit=limit)
        elif rerank:
            from lib.llm import RERANKERS

            if rerank not in RERANKERS:
                raise ValueError(f"unknown rerank method: {rerank}")
            results = engine.rrf_search(query, k=60, limit=limit * 5)
            results = RERANKERS[rerank](query, results)[:limit]
        else:
            results = engine.rrf_search(query, k=60, limit=limit)
        return [
            {
                "title": r["title"],
                "score": round(float(r["score"]), 4),
                "raw_score": (
                    f"rerank {r['rerank_score']:.1f}/10, rrf {r['score']:.3f}"
                    if "rerank_score" in r
                    else f"rerank rank {r['rerank_rank']}, rrf {r['score']:.3f}"
                    if "rerank_rank" in r
                    else f"cross-enc {r['cross_encoder_score']:.3f}, rrf {r['score']:.3f}"
                    if "cross_encoder_score" in r
                    else f"bm25 {r['bm25_score']:.3f} + semantic {r['semantic_score']:.3f}"
                    if "bm25_score" in r
                    else f"bm25 rank {r['bm25_rank'] or '-'}, semantic rank {r['semantic_rank'] or '-'}"
                    if "bm25_rank" in r
                    else "hybrid"
                ),
                "snippet": r.get("document", r.get("description", ""))[:200],
                "description": r.get("description", r.get("document", "")),
                "matched_chunk": r.get("metadata", {}).get("matched_chunk"),
                "highlight": [],
            }
            for r in results
        ]
    raise ValueError(f"unknown mode: {mode}")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PATH.read_bytes())
            return

        if parsed.path == "/api/search":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0].strip()
            mode = params.get("mode", ["semantic"])[0]
            dataset = params.get("dataset", ["movies"])[0]
            if dataset not in DATASETS:
                dataset = "movies"
            limit = int(params.get("limit", ["5"])[0])
            enhance = params.get("enhance", [None])[0]
            rerank = params.get("rerank", [None])[0]
            t0 = time.perf_counter()
            logger.info(
                "search: dataset=%s mode=%s query=%r limit=%d enhance=%s rerank=%s",
                dataset, mode, query, limit, enhance, rerank,
            )
            try:
                if not query:
                    raise ValueError("empty query")
                enhanced = None
                if enhance:
                    from lib.llm import ENHANCERS, enhance_query

                    if enhance not in ENHANCERS:
                        raise ValueError(f"unknown enhance method: {enhance}")
                    corrected = enhance_query(query, enhance)
                    enhanced = {
                        "method": enhance,
                        "original": query,
                        "query": corrected,
                    }
                    query = corrected
                results = search(dataset, mode, query, limit, rerank)
                payload = {"results": results, "enhanced": enhanced}
                status = 200
                logger.info(
                    "done: %d results in %.2fs",
                    len(results), time.perf_counter() - t0,
                )
            except Exception as e:
                payload = {"error": str(e)}
                status = 400
                logger.exception("search failed: %s", e)
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} - {fmt % args}")


if __name__ == "__main__":
    print(f"Search UI running at  http://localhost:{PORT}")
    print("First search per mode is slow (engine loads once, then stays warm).")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
