"""Apply the schema and load the synthetic data into Postgres + pgvector.

Usage: DATABASE_URL=postgresql://postgres:postgres@localhost:5433/matching \\
       python scripts/seed_db.py [--reset]

--reset drops the public and auth schemas first (use only on a throwaway database).
Embeddings use MATCHING_EMBEDDER (default here: tfidf, the offline embedder).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from matching.dataset import embedding_corpus, load_campaigns, load_creators  # noqa: E402
from matching.embedding import get_embedder  # noqa: E402
from matching.pgdb import apply_schema, seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reset", action="store_true", help="drop and recreate the schema")
    args = parser.parse_args()
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        sys.exit("DATABASE_URL is not set")
    applied = apply_schema(dsn, reset=args.reset)
    print("schema applied" if applied else "schema already present; upserting data")
    embedder = get_embedder(embedding_corpus(load_creators(), load_campaigns()),
                            os.getenv("MATCHING_EMBEDDER", "tfidf"))
    matcher = seed(dsn, embedder)
    print(f"seeded {len(matcher.index)} creators with {embedder.name} embeddings")


if __name__ == "__main__":
    main()
