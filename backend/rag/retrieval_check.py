"""Eyeball retrieval quality against hand-written clinical queries.

    python -m rag.retrieval_check
    python -m rag.retrieval_check --query "radial meniscal tear management" -k 3

Not a test. Tuning retrieval depth and chunk size is a judgement call that a unit
test cannot make for you; reading ten real queries' top hits once tells you more
than any assertion about scores. Reach for this after changing the chunker, the
embedding model, or the corpus.
"""

from __future__ import annotations

import argparse
import sys

from rag.chunker import strip_context_header
from rag.config import TOP_K

DEFAULT_QUERIES: tuple[str, ...] = (
    "What defines Kellgren-Lawrence grade 3 osteoarthritis?",
    "Which radiographic features distinguish grade 1 from grade 2?",
    "How reliable is KL grading between observers?",
    "What is a radial meniscal tear and why does it matter?",
    "Which meniscal tears are repairable?",
    "What does meniscal extrusion indicate?",
    "How are cartilage lesions graded by depth?",
    "What does a full-thickness cartilage defect imply for management?",
    "Is joint effusion a specific finding?",
    "Can a normal radiograph coexist with significant soft-tissue injury?",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="run a single query instead of the default set")
    parser.add_argument("-k", type=int, default=TOP_K, help="hits per query")
    parser.add_argument(
        "--chars", type=int, default=160, help="passage characters to print"
    )
    args = parser.parse_args(argv)

    from rag.embedder import get_embedder
    from rag.store import get_store

    embedder = get_embedder()
    store = get_store()

    try:
        if store.count() == 0:
            print("Index is empty. Run `python -m rag.ingest` first.")
            return 1

        queries = [args.query] if args.query else list(DEFAULT_QUERIES)
        for query in queries:
            print(f"\n=== {query}")
            hits = store.search(embedder.embed_query(query), k=args.k)
            if not hits:
                print("  (no hits)")
                continue
            for rank, hit in enumerate(hits, start=1):
                # Not a naive split on the first ": " — article titles
                # routinely contain one.
                body = strip_context_header(hit.text)
                print(
                    f"  {rank}. {hit.score:.3f}  [{hit.source_type}]  "
                    f"{hit.source[:70]}"
                )
                print(f"     {body[:args.chars]}...")
    finally:
        store.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
