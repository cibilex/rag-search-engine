"""Demo: why whole-doc embeddings miss content past the model's max sequence length.

Run from repo root:  uv run truncation_demo.py
Disposable — delete after (not part of the course code).
"""

import sys

sys.path.insert(0, ".")

from lib.search_utils import load_movies
from lib.semantic_search import SemanticSearch, cosine_similarity

QUERY = "Earth is inside a locker at an alien train station"
TARGET = "Men in Black II"
SCENE = (
    "Kay kicks the door open, revealing to Jay and Frank a giant room full of aliens; "
    "their door is actually a door to a locker that contains Earth, among hundreds of "
    "lockers at an alien type of Grand Central Terminal."
)

docs = load_movies()
mib = next(d for d in docs if d["title"] == TARGET)
words = mib["description"].split()

print(f"Query: {QUERY!r}")
print(f"Target: {TARGET} — description is {len(words)} words\n")

print("1) Content exists in the raw data:")
locker_pos = next(i for i, w in enumerate(words) if "locker" in w)
print(f"   'locker' first appears at word ~{locker_pos} of {len(words)}")
print("   model embeds only first ~200 words -> that content never got embedded\n")

print("2) Where whole-doc search ranks the target (loading model + embeddings)...")
s = SemanticSearch()
s.load_or_create_embeddings(docs)
q = s.generate_embedding(QUERY)
scored = sorted(
    ((cosine_similarity(q, e), d["title"]) for e, d in zip(s.embeddings, s.documents)),
    reverse=True,
)
rank = next(i for i, (_, t) in enumerate(scored, 1) if t == TARGET)
target_score = next(sc for sc, t in scored if t == TARGET)
print(f"   rank: {rank} / {len(docs)}   score: {target_score:.4f}")
print(f"   top hit: {scored[0][1]} ({scored[0][0]:.4f})\n")

print("3) Same query vs the ending, embedded separately (what chunking enables):")
tail_score = cosine_similarity(q, s.generate_embedding(" ".join(words[-80:])))
scene_score = cosine_similarity(q, s.generate_embedding(SCENE))
print(f"   vs 80-word tail blob:      {tail_score:.4f}")
print(f"   vs exact scene sentences:  {scene_score:.4f}")
print(f"\nConclusion: whole-doc {target_score:.4f} -> scene chunk {scene_score:.4f}.")
print("Same model, same query — only difference is WHAT got embedded.")
