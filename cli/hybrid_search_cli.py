import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.hybrid_search import HybridSearch, normalize_scores
from lib.search_utils import load_movies

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("rag.hybrid_cli")
logger.setLevel(logging.DEBUG)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Min-max normalize a list of scores"
    )
    normalize_parser.add_argument(
        "scores", type=float, nargs="*", help="Scores to normalize"
    )

    weighted_parser = subparsers.add_parser(
        "weighted-search", help="Hybrid search blending BM25 and semantic scores"
    )
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument(
        "--alpha", type=float, default=0.5, help="BM25 weight (0=semantic, 1=keyword)"
    )
    weighted_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results"
    )

    rrf_parser = subparsers.add_parser(
        "rrf-search", help="Hybrid search using Reciprocal Rank Fusion"
    )
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument(
        "-k", type=int, default=60, help="RRF damping constant"
    )
    rrf_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results"
    )
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="LLM re-ranking method",
    )
    rrf_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Rate results 0-3 relevance using an LLM",
    )

    args = parser.parse_args()

    match args.command:
        case "weighted-search":
            searcher = HybridSearch(load_movies())
            results = searcher.weighted_search(args.query, args.alpha, args.limit)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']} (hybrid: {res['score']:.4f})")
                print(
                    f"   bm25: {res['bm25_score']:.4f}, semantic: {res['semantic_score']:.4f}"
                )
                print(f"   {res['description'][:100]}...")
        case "rrf-search":
            query = args.query
            logger.debug("original query: %r", query)
            if args.enhance:
                from lib.llm import enhance_query

                query = enhance_query(query, args.enhance)
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{query}'\n")
            logger.debug("query after enhancement: %r", query)
            searcher = HybridSearch(load_movies())
            results = searcher.rrf_search(query, args.k, args.limit * 5 if args.rerank_method else args.limit)
            logger.debug(
                "RRF results (%d): %s",
                len(results),
                [r["title"] for r in results],
            )
            if args.rerank_method:
                from lib.llm import RERANKERS

                print(
                    f"Re-ranking top {args.limit} results "
                    f"using {args.rerank_method} method..."
                )
                results = RERANKERS[args.rerank_method](query, results)[: args.limit]
                logger.debug(
                    "results after re-ranking (%d): %s",
                    len(results),
                    [r["title"] for r in results],
                )
            print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")
            for i, res in enumerate(results, 1):
                bm25_rank = res["bm25_rank"] if res["bm25_rank"] is not None else "-"
                sem_rank = (
                    res["semantic_rank"] if res["semantic_rank"] is not None else "-"
                )
                print(f"{i}. {res['title']}")
                if "rerank_score" in res:
                    print(f"   Re-rank Score: {res['rerank_score']:.3f}/10")
                if "rerank_rank" in res:
                    print(f"   Re-rank Rank: {res['rerank_rank']}")
                if "cross_encoder_score" in res:
                    print(
                        f"   Cross Encoder Score: {res['cross_encoder_score']:.3f}"
                    )
                print(f"  RRF Score: {res['score']:.3f}")
                print(f"  BM25 Rank: {bm25_rank}, Semantic Rank: {sem_rank}")
                print(f"  {res['description'][:100]}...")
                print()
            if args.evaluate:
                from lib.llm import evaluate_results

                scores = evaluate_results(query, results)
                for i, (res, score) in enumerate(zip(results, scores), 1):
                    print(f"{i}. {res['title']}: {score}/3")
        case "normalize":
            for score in normalize_scores(args.scores):
                print(f"* {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
