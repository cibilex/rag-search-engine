import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies


def load_golden_dataset(path: str = "data/golden_dataset.json") -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["test_cases"]


def precision_at_k(retrieved: list[str], relevant: list[str]) -> float:
    relevant_set = set(relevant)
    hits = sum(1 for title in retrieved if title in relevant_set)
    return hits / len(retrieved)


def recall_at_k(retrieved: list[str], relevant: list[str]) -> float:
    relevant_set = set(relevant)
    hits = sum(1 for title in retrieved if title in relevant_set)
    return hits / len(relevant)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    test_cases = load_golden_dataset()
    searcher = HybridSearch(load_movies())

    print(f"k={limit}\n")
    for case in test_cases:
        query = case["query"]
        relevant = case["relevant_docs"]

        results = searcher.rrf_search(query, k=60, limit=limit)
        retrieved = [r["title"] for r in results]

        precision = precision_at_k(retrieved, relevant)
        recall = recall_at_k(retrieved, relevant)
        f1 = f1_score(precision, recall)

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved)}")
        print(f"  - Relevant: {', '.join(relevant)}")
        print()


if __name__ == "__main__":
    main()
