"""Apply the schema and load the synthetic dataset into Postgres + pgvector.

Used by scripts/seed_db.py and the integration test. Requires `psycopg[binary]`.
"""
from __future__ import annotations

from pathlib import Path

from .dataset import embedding_corpus, load_brands, load_campaigns, load_creators
from .embedding import DIM, Embedder, TfidfEmbedder
from .index import PgVectorIndex
from .matcher import Matcher

ROOT = Path(__file__).resolve().parents[2]
SHIM = ROOT / "db" / "supabase_shim.sql"
MIGRATIONS = ROOT / "supabase" / "migrations"

RESET_SQL = """
drop schema if exists public cascade;
create schema public;
drop schema if exists auth cascade;
"""


def apply_schema(dsn: str, reset: bool = False) -> bool:
    """Run the local Supabase shim, then every migration in filename order.

    Skipped (returns False) when the schema already exists, since the migrations are
    plain `create` statements. `reset=True` drops the `public` and `auth` schemas
    first: point it only at a throwaway database.
    """
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        if reset:
            conn.execute(RESET_SQL)
        elif conn.execute("select to_regclass('public.creators')").fetchone()[0] is not None:
            return False
        conn.execute(SHIM.read_text())
        for path in sorted(MIGRATIONS.glob("*.sql")):
            conn.execute(path.read_text())
    return True


def column_dim(dsn: str, table: str, column: str) -> int:
    """Declared dimension of a pgvector column (pgvector stores it as the type modifier)."""
    import psycopg

    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            """select atttypmod from pg_attribute
               where attrelid = %s::regclass and attname = %s""", (table, column)).fetchone()
    return int(row[0])


def seed(dsn: str, embedder: Embedder | None = None) -> Matcher:
    """Load brands, campaigns (with brief embeddings) and creators (with embeddings)."""
    import psycopg

    brands, campaigns, creators = load_brands(), load_campaigns(), load_creators()
    embedder = embedder or TfidfEmbedder(embedding_corpus(creators, campaigns))
    for table, column in (("creator_embeddings", "embedding"), ("campaigns", "brief_embedding")):
        if (d := column_dim(dsn, table, column)) != DIM:
            raise RuntimeError(f"{table}.{column} is vector({d}); the embedder produces {DIM}")

    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.executemany(
            """insert into brands (id, name, industry) values (%s, %s, %s)
               on conflict (id) do update set name = excluded.name, industry = excluded.industry""",
            [(b.id, b.name, b.industry) for b in brands])

    index = PgVectorIndex(dsn, embedder.name)
    matcher = Matcher(embedder, index)
    index.upsert_campaigns(campaigns, [matcher.embed_brief(c) for c in campaigns])
    index.upsert_many(creators, embedder.encode([c.text for c in creators]))
    return matcher
