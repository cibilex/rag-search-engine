import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.semantic_search import (
    embed_query_text,
    embed_text,
    search_command,
    verify_embeddings,
    verify_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embedding model loads correctly")

    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser(
        "verify_embeddings", help="Build or load movie embeddings and verify them"
    )

    embed_query_parser = subparsers.add_parser(
        "embed_query", help="Generate an embedding for a search query"
    )
    embed_query_parser.add_argument("query", type=str, help="Query to embed")

    search_parser = subparsers.add_parser(
        "search", help="Search movies by meaning using cosine similarity"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            start = time.perf_counter()
            results = search_command(args.query, args.limit)
            elapsed = time.perf_counter() - start
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']} (score: {res['score']:.4f})")
                print(f"  {res['description'][:100]} ...")
                print()
            print(f"Execution time: {elapsed:.2f}s")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
