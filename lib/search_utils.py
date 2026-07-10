import json

DEFAULT_SEARCH_LIMIT = 5


def load_movies(path: str = "data/movies.json") -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords(path: str = "data/stopwords.txt") -> list[str]:
    with open(path) as f:
        content = f.read()
    return content.splitlines()
