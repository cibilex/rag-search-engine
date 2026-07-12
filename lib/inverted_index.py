import math
import os
import pickle
from collections import Counter

from .constants import BM25_B, BM25_K1
from .keyword_search import tokenize_text
from .search_utils import load_movies

INDEX_PATH = "cache/index.pkl"
DOCMAP_PATH = "cache/docmap.pkl"
TERM_FREQ_PATH = "cache/term_frequencies.pkl"
DOC_LENGTHS_PATH = "cache/doc_lengths.pkl"


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter()
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, set()))

    def get_tf(self, doc_id: int, term: str) -> int:
        counter = self.term_frequencies.get(doc_id)
        if counter is None:
            return 0
        return counter[term]

    def get_bm25_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(term))
        return math.log(
            (total_doc_count - term_match_doc_count + 0.5)
            / (term_match_doc_count + 0.5)
            + 1
        )

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1.0
        if avg_doc_length > 0:
            doc_length = self.doc_lengths.get(doc_id, 0)
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query: str, limit: int) -> list[tuple[dict, float]]:
        tokens = tokenize_text(query)
        scores: dict[int, float] = {}
        for token in tokens:
            for doc_id in self.get_documents(token):
                scores[doc_id] = scores.get(doc_id, 0.0) + self.bm25(doc_id, token)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [(self.docmap[doc_id], score) for doc_id, score in ranked[:limit]]

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")

    def save(self) -> None:
        os.makedirs("cache", exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(TERM_FREQ_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(DOC_LENGTHS_PATH, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        if (
            not os.path.exists(INDEX_PATH)
            or not os.path.exists(DOCMAP_PATH)
            or not os.path.exists(TERM_FREQ_PATH)
            or not os.path.exists(DOC_LENGTHS_PATH)
        ):
            raise FileNotFoundError(
                "Index files not found. Run the 'build' command first."
            )
        with open(INDEX_PATH, "rb") as f:
            self.index = pickle.load(f)
        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = pickle.load(f)
        with open(TERM_FREQ_PATH, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(DOC_LENGTHS_PATH, "rb") as f:
            self.doc_lengths = pickle.load(f)
