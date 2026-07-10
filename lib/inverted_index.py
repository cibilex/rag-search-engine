import math
import os
import pickle
from collections import Counter

from .constants import BM25_K1
from .keyword_search import tokenize_text
from .search_utils import load_movies

INDEX_PATH = "cache/index.pkl"
DOCMAP_PATH = "cache/docmap.pkl"
TERM_FREQ_PATH = "cache/term_frequencies.pkl"


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
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

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1) -> float:
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1)

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

    def load(self) -> None:
        if (
            not os.path.exists(INDEX_PATH)
            or not os.path.exists(DOCMAP_PATH)
            or not os.path.exists(TERM_FREQ_PATH)
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
