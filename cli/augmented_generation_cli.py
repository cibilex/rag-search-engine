import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hybrid_search import HybridSearch
from lib.llm import answer_question, answer_with_citations, generate_answer, summarize_results
from lib.search_utils import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize search results across multiple documents"
    )
    summarize_parser.add_argument("query", type=str, help="Search query to summarize")
    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results"
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Answer a query with cited sources"
    )
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results"
    )

    question_parser = subparsers.add_parser(
        "question", help="Answer a natural-language question about movies"
    )
    question_parser.add_argument("question", type=str, help="Question to answer")
    question_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            searcher = HybridSearch(load_movies())
            results = searcher.rrf_search(query, k=60, limit=5)

            print("Search Results:")
            for res in results:
                print(f"- {res['title']}")
            print()

            answer = generate_answer(query, results)
            print("RAG Response:")
            print(answer)
        case "summarize":
            query = args.query
            searcher = HybridSearch(load_movies())
            results = searcher.rrf_search(query, k=60, limit=args.limit)

            print("Search Results:")
            for res in results:
                print(f"  - {res['title']}")
            print()

            summary = summarize_results(query, results)
            print("LLM Summary:")
            print(summary)
        case "citations":
            query = args.query
            searcher = HybridSearch(load_movies())
            results = searcher.rrf_search(query, k=60, limit=args.limit)

            print("Search Results:")
            for res in results:
                print(f"  - {res['title']}")
            print()

            answer = answer_with_citations(query, results)
            print("LLM Answer:")
            print(answer)
        case "question":
            question = args.question
            searcher = HybridSearch(load_movies())
            results = searcher.rrf_search(question, k=60, limit=args.limit)

            print("Search Results:")
            for res in results:
                print(f"  - {res['title']}")
            print()

            answer = answer_question(question, results)
            print("Answer:")
            print(answer)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
