import json
import logging
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger("rag.llm")


def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_TOKEN")
    if not api_key:
        raise RuntimeError("DEEPSEEK_TOKEN environment variable not set")
    return OpenAI(base_url="https://api.deepseek.com", api_key=api_key)


def correct_spelling(query: str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
    return _complete(prompt)


def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
    return _complete(prompt)


def expand_query(query: str) -> str:
    prompt = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""
    return f"{query} {_complete(prompt)}"


ENHANCERS = {
    "spell": correct_spelling,
    "rewrite": rewrite_query,
    "expand": expand_query,
}


def enhance_query(query: str, method: str) -> str:
    enhanced = ENHANCERS[method](query)
    logger.info("query enhanced (%s): %r -> %r", method, query, enhanced)
    return enhanced


def score_document(query: str, doc: dict, retries: int = 3) -> float:
    prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document") or doc.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
    for _ in range(retries):
        try:
            return float(_complete(prompt))
        except ValueError:
            continue
    return 0.0


def rerank_individual(query: str, docs: list[dict], delay: float = 0.0) -> list[dict]:
    logger.info("re-ranking %d docs (individual) for query %r", len(docs), query)
    scored = []
    for doc in docs:
        scored.append({**doc, "rerank_score": score_document(query, doc)})
        if delay:
            time.sleep(delay)
    return sorted(scored, key=lambda d: d["rerank_score"], reverse=True)


_cross_encoder = None


def rerank_cross_encoder(query: str, docs: list[dict]) -> list[dict]:
    global _cross_encoder
    logger.info("re-ranking %d docs (cross_encoder) for query %r", len(docs), query)
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    pairs = [
        [query, f"{d.get('title', '')} - {d.get('document') or d.get('description', '')}"]
        for d in docs
    ]
    scores = _cross_encoder.predict(pairs)
    scored = [{**d, "cross_encoder_score": float(s)} for d, s in zip(docs, scores)]
    return sorted(scored, key=lambda d: d["cross_encoder_score"], reverse=True)


def rerank_batch(query: str, docs: list[dict], retries: int = 3) -> list[dict]:
    logger.info("re-ranking %d docs (batch) for query %r", len(docs), query)
    doc_list_str = "\n".join(
        f"{d['id']}: {d.get('title', '')} - "
        f"{(d.get('document') or d.get('description', ''))[:500]}"
        for d in docs
    )
    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return the movie IDs in order of relevance, best match first.

Your response must be a raw JSON array of integers.
Do not wrap the JSON in Markdown. Do not use a ```json code block.
Do not include any explanatory text.

For example:
[75, 12, 34, 2, 1]

Ranking:"""
    for _ in range(retries):
        try:
            ids = json.loads(_complete(prompt))
            order = {int(doc_id): i for i, doc_id in enumerate(ids)}
        except (ValueError, TypeError):
            continue
        ranked = sorted(docs, key=lambda d: order.get(d["id"], len(order)))
        return [{**d, "rerank_rank": i} for i, d in enumerate(ranked, 1)]
    raise RuntimeError("batch rerank failed: LLM kept returning invalid JSON")


RERANKERS = {
    "individual": rerank_individual,
    "batch": rerank_batch,
    "cross_encoder": rerank_cross_encoder,
}


def evaluate_results(query: str, docs: list[dict], retries: int = 3) -> list[int]:
    logger.info("evaluating %d results for query %r", len(docs), query)
    formatted_results = [
        f"{i}. {d.get('title', '')} - {(d.get('document') or d.get('description', ''))[:300]}"
        for i, d in enumerate(docs, 1)
    ]
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
    for _ in range(retries):
        try:
            scores = json.loads(_complete(prompt))
            if isinstance(scores, list) and len(scores) == len(docs):
                return [int(s) for s in scores]
        except (ValueError, TypeError):
            continue
    raise RuntimeError("evaluate failed: LLM kept returning invalid JSON")


def summarize_results(query: str, docs: list[dict]) -> str:
    logger.info("summarizing %d results for query %r", len(docs), query)
    doc_list_str = "\n\n".join(
        f"{d.get('title', '')}\n{d.get('document') or d.get('description', '')}"
        for d in docs
    )
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Webflyx users. Webflyx is a movie streaming service.

Query: {query}

Search results:
{doc_list_str}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
    return _complete(prompt)


def answer_with_citations(query: str, docs: list[dict]) -> str:
    logger.info("generating cited answer for query %r from %d docs", query, len(docs))
    documents = "\n\n".join(
        f"[{i}] {d.get('title', '')}\n{d.get('document') or d.get('description', '')}"
        for i, d in enumerate(docs, 1)
    )
    prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Webflyx, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{documents}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
    return _complete(prompt)


def answer_question(question: str, docs: list[dict]) -> str:
    logger.info("answering question %r from %d docs", question, len(docs))
    context = "\n\n".join(
        f"{d.get('title', '')}\n{d.get('document') or d.get('description', '')}"
        for d in docs
    )
    prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

Question: {question}

Documents:
{context}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
    return _complete(prompt)


def generate_answer(query: str, docs: list[dict]) -> str:
    logger.info("generating RAG answer for query %r from %d docs", query, len(docs))
    doc_list_str = "\n\n".join(
        f"{d.get('title', '')}\n{d.get('document') or d.get('description', '')}"
        for d in docs
    )
    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{doc_list_str}

Answer:"""
    return _complete(prompt)


def _complete(prompt: str) -> str:
    response = get_client().chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip().strip('"')
