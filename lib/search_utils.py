import json

DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 4


def format_search_result(
    doc_id, title: str, document: str, score: float, metadata: dict | None = None
) -> dict:
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata or {},
    }


def load_movies(path: str = "data/movies.json") -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords(path: str = "data/stopwords.txt") -> list[str]:
    with open(path) as f:
        content = f.read()
    return content.splitlines()
