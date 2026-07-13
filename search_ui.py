"""Interactive UI for exploring the search engines.

Run:  uv run search_ui.py
Engines load lazily (first use) and stay in memory — repeated searches are fast.
"""

import sys

sys.path.insert(0, ".")

from lib.search_utils import load_movies

BANNER = """
==========================================
   RAG SEARCH ENGINE — interactive UI
==========================================
Search modes:
  1. Keyword search (BM25)
  2. Semantic search (whole document)
  3. Semantic search (chunked)
  q. Quit
"""

# lazily-initialized engines (load once, reuse)
_engines: dict = {}


def get_keyword_engine():
    if "keyword" not in _engines:
        from lib.inverted_index import InvertedIndex

        print("  [loading BM25 index...]")
        index = InvertedIndex()
        index.load()
        _engines["keyword"] = index
    return _engines["keyword"]


def get_semantic_engine():
    if "semantic" not in _engines:
        from lib.semantic_search import SemanticSearch

        print("  [loading embedding model + movie embeddings...]")
        s = SemanticSearch()
        s.load_or_create_embeddings(load_movies())
        _engines["semantic"] = s
    return _engines["semantic"]


def get_chunked_engine():
    if "chunked" not in _engines:
        from lib.semantic_search import ChunkedSemanticSearch

        print("  [loading embedding model + 72k chunk embeddings...]")
        s = ChunkedSemanticSearch()
        s.load_or_create_chunk_embeddings(load_movies())
        _engines["chunked"] = s
    return _engines["chunked"]


def run_keyword(query: str, limit: int) -> None:
    index = get_keyword_engine()
    results = index.bm25_search(query, limit)
    if not results:
        print("  no results (BM25 needs shared tokens — try different words)")
    for i, (movie, score) in enumerate(results, 1):
        print(f"\n{i}. {movie['title']} (BM25 score: {score:.2f})")
        print(f"   {movie['description'][:120]}...")


def run_semantic(query: str, limit: int) -> None:
    s = get_semantic_engine()
    for i, res in enumerate(s.search(query, limit), 1):
        print(f"\n{i}. {res['title']} (cosine: {res['score']:.4f})")
        print(f"   {res['description'][:120]}...")


def run_chunked(query: str, limit: int) -> None:
    s = get_chunked_engine()
    for i, res in enumerate(s.search_chunks(query, limit), 1):
        print(f"\n{i}. {res['title']} (cosine: {res['score']:.4f})")
        print(f"   {res['document']}...")
        print(f"   matched chunk: {res['metadata']['matched_chunk'][:150]}...")


MODES = {
    "1": ("Keyword (BM25)", run_keyword),
    "2": ("Semantic (whole doc)", run_semantic),
    "3": ("Semantic (chunked)", run_chunked),
}


def main() -> None:
    print(BANNER)
    while True:
        choice = input("Select mode [1/2/3/q]: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            print("bye!")
            return
        if choice not in MODES:
            print("  invalid choice")
            continue

        name, runner = MODES[choice]
        print(f"\n--- {name} --- (empty query to go back)")
        while True:
            query = input("\nquery> ").strip()
            if not query:
                break
            limit_raw = input("limit [5]> ").strip()
            limit = int(limit_raw) if limit_raw.isdigit() else 5
            try:
                runner(query, limit)
            except Exception as e:
                print(f"  error: {e}")


if __name__ == "__main__":
    main()
