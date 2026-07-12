import math
import string
import sys
from functools import lru_cache

from nltk.stem import PorterStemmer

from .constants import BM25_B, BM25_K1
from .search_utils import DEFAULT_SEARCH_LIMIT, load_stopwords

stemmer = PorterStemmer()


def tokenize_term(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError(f"Term '{term}' must tokenize to exactly one token, got {tokens}")
    return tokens[0]


def tf_command(doc_id: int, term: str) -> None:
    from .inverted_index import InvertedIndex

    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    token = tokenize_term(term)
    print(index.get_tf(doc_id, token))


def compute_idf(index, token: str) -> float:
    total_doc_count = len(index.docmap)
    term_match_doc_count = len(index.get_documents(token))
    return math.log((total_doc_count + 1) / (term_match_doc_count + 1))


def idf_command(term: str) -> float:
    from .inverted_index import InvertedIndex

    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    token = tokenize_term(term)
    return compute_idf(index, token)


def tfidf_command(doc_id: int, term: str) -> float:
    from .inverted_index import InvertedIndex

    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    token = tokenize_term(term)
    tf = index.get_tf(doc_id, token)
    idf = compute_idf(index, token)
    return tf * idf


def bm25_idf_command(term: str) -> float:
    from .inverted_index import InvertedIndex

    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    token = tokenize_term(term)
    return index.get_bm25_idf(token)


def bm25_tf_command(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    from .inverted_index import InvertedIndex

    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    token = tokenize_term(term)
    return index.get_bm25_tf(doc_id, token, k1, b)


def bm25_search_command(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[tuple[dict, float]]:
    from .inverted_index import InvertedIndex

    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    return index.bm25_search(query, limit)


def build_command() -> None:
    from .inverted_index import InvertedIndex

    index = InvertedIndex()
    index.build()
    index.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    from .inverted_index import InvertedIndex

    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    query_tokens = tokenize_text(query)
    seen_ids = set()
    results = []
    for token in query_tokens:
        for doc_id in index.get_documents(token):
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            results.append(index.docmap[doc_id])
            if len(results) >= limit:
                return results

    return results


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


@lru_cache(maxsize=1)
def get_stop_words() -> frozenset[str]:
    return frozenset(preprocess_text(word) for word in load_stopwords())


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    stop_words = get_stop_words()
    valid_tokens = []
    for token in tokens:
        if token and token not in stop_words:
            valid_tokens.append(stemmer.stem(token))
    return valid_tokens
