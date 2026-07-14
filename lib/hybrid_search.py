import os

from .inverted_index import InvertedIndex
from .semantic_search import ChunkedSemanticSearch


def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = 0.5
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        fetch = limit * 500
        bm25_results = self._bm25_search(query, fetch)
        semantic_results = self.semantic_search.search_chunks(query, fetch)

        bm25_normalized = normalize_scores([score for _, score in bm25_results])
        semantic_normalized = normalize_scores([r["score"] for r in semantic_results])

        combined: dict[int, dict] = {}
        for (movie, _), norm in zip(bm25_results, bm25_normalized):
            combined[movie["id"]] = {
                "doc": movie,
                "bm25_score": norm,
                "semantic_score": 0.0,
            }
        for result, norm in zip(semantic_results, semantic_normalized):
            entry = combined.setdefault(
                result["id"],
                {
                    "doc": self.semantic_search.document_map[result["id"]],
                    "bm25_score": 0.0,
                    "semantic_score": 0.0,
                },
            )
            entry["semantic_score"] = norm

        for entry in combined.values():
            entry["hybrid_score"] = hybrid_score(
                entry["bm25_score"], entry["semantic_score"], alpha
            )

        ranked = sorted(
            combined.values(), key=lambda e: e["hybrid_score"], reverse=True
        )
        return [
            {
                "id": entry["doc"]["id"],
                "title": entry["doc"]["title"],
                "description": entry["doc"]["description"],
                "score": entry["hybrid_score"],
                "bm25_score": entry["bm25_score"],
                "semantic_score": entry["semantic_score"],
            }
            for entry in ranked[:limit]
        ]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        fetch = limit * 500
        bm25_results = self._bm25_search(query, fetch)
        semantic_results = self.semantic_search.search_chunks(query, fetch)

        combined: dict[int, dict] = {}
        for rank, (movie, _) in enumerate(bm25_results, 1):
            combined[movie["id"]] = {
                "doc": movie,
                "bm25_rank": rank,
                "semantic_rank": None,
            }
        for rank, result in enumerate(semantic_results, 1):
            entry = combined.setdefault(
                result["id"],
                {
                    "doc": self.semantic_search.document_map[result["id"]],
                    "bm25_rank": None,
                    "semantic_rank": None,
                },
            )
            entry["semantic_rank"] = rank

        for entry in combined.values():
            score = 0.0
            if entry["bm25_rank"] is not None:
                score += rrf_score(entry["bm25_rank"], k)
            if entry["semantic_rank"] is not None:
                score += rrf_score(entry["semantic_rank"], k)
            entry["rrf_score"] = score

        ranked = sorted(combined.values(), key=lambda e: e["rrf_score"], reverse=True)
        return [
            {
                "id": entry["doc"]["id"],
                "title": entry["doc"]["title"],
                "description": entry["doc"]["description"],
                "score": entry["rrf_score"],
                "bm25_rank": entry["bm25_rank"],
                "semantic_rank": entry["semantic_rank"],
            }
            for entry in ranked[:limit]
        ]
