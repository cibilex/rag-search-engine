import json
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import format_search_result, load_movies

EMBEDDINGS_PATH = "cache/movie_embeddings.npy"
CHUNK_EMBEDDINGS_PATH = "cache/chunk_embeddings.npy"
CHUNK_METADATA_PATH = "cache/chunk_metadata.json"


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticSearch:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embeddings_path: str = EMBEDDINGS_PATH,
    ):
        self.model = SentenceTransformer(model_name)
        self.embeddings_path = embeddings_path
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        movie_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int) -> list[dict]:
        if self.embeddings is None:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        scored = [
            (cosine_similarity(query_embedding, doc_embedding), doc)
            for doc_embedding, doc in zip(self.embeddings, self.documents)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "score": score,
                "title": doc["title"],
                "description": doc["description"],
            }
            for score, doc in scored[:limit]
        ]

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or whitespace-only")
        return self.model.encode([text])[0]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embeddings_path: str = EMBEDDINGS_PATH,
        chunk_embeddings_path: str = CHUNK_EMBEDDINGS_PATH,
        chunk_metadata_path: str = CHUNK_METADATA_PATH,
        chunk_size: int = 4,
        chunk_overlap: int = 1,
        aggregation: str = "max",
    ) -> None:
        super().__init__(model_name, embeddings_path)
        self.chunk_embeddings_path = chunk_embeddings_path
        self.chunk_metadata_path = chunk_metadata_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.aggregation = aggregation
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        chunk_metadata = []
        for movie_idx, doc in enumerate(documents):
            description = doc.get("description", "")
            if not description.strip():
                continue
            chunks = semantic_chunk_text(
                description, self.chunk_size, self.chunk_overlap
            )
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(f"{doc['title']}: {chunk}")
                chunk_metadata.append(
                    {
                        "movie_idx": movie_idx,
                        "chunk_idx": chunk_idx,
                        "total_chunks": len(chunks),
                    }
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)},
                f,
                indent=2,
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(
            self.chunk_metadata_path
        ):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path) as f:
                self.chunk_metadata = json.load(f)["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        query_embedding = self.generate_embedding(query)

        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            meta = self.chunk_metadata[i]
            chunk_scores.append(
                {
                    "chunk_idx": meta["chunk_idx"],
                    "movie_idx": meta["movie_idx"],
                    "score": cosine_similarity(query_embedding, chunk_embedding),
                }
            )

        per_movie: dict[int, list[dict]] = {}
        for chunk_score in chunk_scores:
            per_movie.setdefault(chunk_score["movie_idx"], []).append(chunk_score)

        aggregated = []
        for scores in per_movie.values():
            scores.sort(key=lambda item: item["score"], reverse=True)
            best = scores[0]
            if self.aggregation == "top2":
                top = scores[:2]
                agg_score = sum(item["score"] for item in top) / len(top)
            else:  # "max"
                agg_score = best["score"]
            aggregated.append({**best, "score": agg_score})

        ranked = sorted(aggregated, key=lambda item: item["score"], reverse=True)[
            :limit
        ]

        results = []
        for item in ranked:
            doc = self.documents[item["movie_idx"]]
            chunks = semantic_chunk_text(
                doc["description"], self.chunk_size, self.chunk_overlap
            )
            matched_chunk = chunks[item["chunk_idx"]]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:100],
                    score=item["score"],
                    metadata={
                        "chunk_idx": item["chunk_idx"],
                        "movie_idx": item["movie_idx"],
                        "total_chunks": len(chunks),
                        "matched_chunk": matched_chunk,
                    },
                )
            )
        return results


def embed_text(text: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query: str) -> None:
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def search_command(query: str, limit: int = 5) -> list[dict]:
    search = SemanticSearch()
    documents = load_movies()
    search.load_or_create_embeddings(documents)
    return search.search(query, limit)


def chunk_text(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")
    words = text.split()
    stride = chunk_size - overlap
    return [
        " ".join(words[i : i + chunk_size]) for i in range(0, len(words), stride)
    ]


def semantic_chunk_text(
    text: str, max_chunk_size: int = 4, overlap: int = 0
) -> list[str]:
    if overlap >= max_chunk_size:
        raise ValueError("Overlap must be smaller than max chunk size")
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]
    stride = max_chunk_size - overlap
    chunks = []
    for i in range(0, len(sentences), stride):
        chunks.append(" ".join(sentences[i : i + max_chunk_size]))
        if i + max_chunk_size >= len(sentences):
            break
    return chunks


def semantic_chunk_command(
    text: str, max_chunk_size: int = 4, overlap: int = 0
) -> None:
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(semantic_chunk_text(text, max_chunk_size, overlap), 1):
        print(f"{i}. {chunk}")


def chunk_command(text: str, chunk_size: int = 200, overlap: int = 0) -> None:
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunk_text(text, chunk_size, overlap), 1):
        print(f"{i}. {chunk}")


def embed_chunks_command() -> None:
    search = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked_command(query: str, limit: int = 5) -> list[dict]:
    search = ChunkedSemanticSearch()
    documents = load_movies()
    search.load_or_create_chunk_embeddings(documents)
    return search.search_chunks(query, limit)


def verify_embeddings() -> None:
    search = SemanticSearch()
    documents = load_movies()
    embeddings = search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def verify_model() -> None:
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")
