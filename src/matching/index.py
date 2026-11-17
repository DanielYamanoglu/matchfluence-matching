"""Vector indexes with one interface: in-memory numpy for the demo, pgvector for Postgres."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from .models import Campaign, Creator

MIN_AUDIENCE_SHARE = 0.20


@dataclass(frozen=True)
class CandidateFilter:
    """Hard constraints applied before ranking (mirrors the SQL in match_creators)."""

    target_countries: tuple[str, ...] = ()
    min_followers: int = 0
    max_followers: int = 2_000_000_000
    max_rate_usd: float = float("inf")
    min_audience_share: float = MIN_AUDIENCE_SHARE

    @classmethod
    def from_campaign(cls, c: Campaign) -> "CandidateFilter":
        return cls(tuple(c.target_countries), c.min_followers, c.max_followers, c.budget_usd)

    def allows(self, cr: Creator) -> bool:
        if not self.min_followers <= cr.followers <= self.max_followers:
            return False
        if cr.rate_usd > self.max_rate_usd:
            return False
        if self.target_countries:
            return any(
                cr.audience_countries.get(c, 0.0) >= self.min_audience_share
                for c in self.target_countries
            )
        return True


@dataclass(frozen=True)
class Hit:
    creator_id: str
    similarity: float


class VectorIndex(Protocol):
    def upsert(self, creator: Creator, vector: np.ndarray) -> None: ...

    def search_campaign(self, campaign: Campaign, query: np.ndarray, k: int) -> list[Hit]:
        """Top-k creators for `campaign` (query = its brief embedding) after its hard filters."""
        ...

    def get_many(self, ids: Sequence[str]) -> list[Creator]: ...

    def __len__(self) -> int: ...


@dataclass
class InMemoryIndex:
    """Brute-force cosine search over a numpy matrix (vectors are L2-normalised)."""

    _ids: list[str] = field(default_factory=list)
    _creators: dict[str, Creator] = field(default_factory=dict)
    _vectors: np.ndarray | None = None
    _pos: dict[str, int] = field(default_factory=dict)

    def upsert(self, creator: Creator, vector: np.ndarray) -> None:
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        self._creators[creator.id] = creator
        if creator.id in self._pos:
            self._vectors[self._pos[creator.id]] = v[0]
            return
        self._pos[creator.id] = len(self._ids)
        self._ids.append(creator.id)
        self._vectors = v if self._vectors is None else np.vstack([self._vectors, v])

    def search(self, query: np.ndarray, k: int, flt: CandidateFilter) -> list[Hit]:
        if self._vectors is None:
            return []
        sims = self._vectors @ np.asarray(query, dtype=np.float32).ravel()
        allowed = np.fromiter(
            (flt.allows(self._creators[i]) for i in self._ids), dtype=bool, count=len(self._ids)
        )
        sims = np.where(allowed, sims, -np.inf)
        top = np.argsort(-sims, kind="stable")[:k]
        return [Hit(self._ids[i], float(sims[i])) for i in top if np.isfinite(sims[i])]

    def search_campaign(self, campaign: Campaign, query: np.ndarray, k: int) -> list[Hit]:
        return self.search(query, k, CandidateFilter.from_campaign(campaign))

    def get_many(self, ids: Sequence[str]) -> list[Creator]:
        return [self._creators[i] for i in ids]

    def __len__(self) -> int:
        return len(self._ids)


class PgVectorIndex:
    """Same interface backed by Postgres + pgvector (schema in supabase/migrations).

    Retrieval calls the SQL function `match_creators()`, so the hard filters live in one
    place on this path. Requires `psycopg[binary]`; the database is seeded by
    scripts/seed_db.py. Exercised by tests/test_pgvector_integration.py.
    """

    def __init__(self, dsn: str, model: str):
        import psycopg  # optional dependency

        self.model = model
        self._conn = psycopg.connect(dsn, autocommit=True)

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def vec(v: np.ndarray) -> str:
        # repr of each float32 value round-trips exactly into pgvector's float4 storage
        return "[" + ",".join(map(repr, np.asarray(v, dtype=np.float32).ravel().tolist())) + "]"

    def stored_models(self) -> set[str]:
        with self._conn.cursor() as cur:
            cur.execute("select distinct model from creator_embeddings")
            return {r[0] for r in cur.fetchall()}

    def upsert(self, creator: Creator, vector: np.ndarray) -> None:
        self.upsert_many([creator], [vector])

    def upsert_many(self, creators: Sequence[Creator], vectors: Sequence[np.ndarray]) -> None:
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.executemany(
                """insert into creators (id, handle, platform, bio, tags, audience_countries,
                       followers, engagement_rate, rate_usd, past_performance)
                   values (%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s)
                   on conflict (id) do update set handle = excluded.handle,
                       platform = excluded.platform, bio = excluded.bio, tags = excluded.tags,
                       audience_countries = excluded.audience_countries,
                       followers = excluded.followers, engagement_rate = excluded.engagement_rate,
                       rate_usd = excluded.rate_usd, past_performance = excluded.past_performance""",
                [(c.id, c.handle, c.platform, c.bio, c.tags, json.dumps(c.audience_countries),
                  c.followers, c.engagement_rate, c.rate_usd, c.past_performance)
                 for c in creators],
            )
            cur.executemany(
                """insert into creator_embeddings (creator_id, model, embedding)
                   values (%s, %s, %s::vector)
                   on conflict (creator_id) do update set embedding = excluded.embedding,
                       model = excluded.model, updated_at = now()""",
                [(c.id, self.model, self.vec(v)) for c, v in zip(creators, vectors)],
            )

    def upsert_campaigns(self, campaigns: Sequence[Campaign], vectors: Sequence[np.ndarray]) -> None:
        """Store campaigns with their brief embeddings (their brands must already exist)."""
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.executemany(
                """insert into campaigns (id, brand_id, title, brief, target_countries,
                       min_followers, max_followers, budget_usd, brief_embedding)
                   values (%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
                   on conflict (id) do update set brand_id = excluded.brand_id,
                       title = excluded.title, brief = excluded.brief,
                       target_countries = excluded.target_countries,
                       min_followers = excluded.min_followers,
                       max_followers = excluded.max_followers,
                       budget_usd = excluded.budget_usd,
                       brief_embedding = excluded.brief_embedding""",
                [(c.id, c.brand_id, c.title, c.brief, c.target_countries, c.min_followers,
                  c.max_followers, c.budget_usd, self.vec(v)) for c, v in zip(campaigns, vectors)],
            )

    def search_campaign(self, campaign: Campaign, query: np.ndarray, k: int) -> list[Hit]:
        # Write the brief embedding first so match_creators() filters on the campaign as given.
        self.upsert_campaigns([campaign], [query])
        with self._conn.cursor() as cur:
            cur.execute("select creator_id, similarity from match_creators(%s::text, %s::int, %s::numeric)",
                        (campaign.id, k, MIN_AUDIENCE_SHARE))
            return [Hit(r[0], float(r[1])) for r in cur.fetchall()]

    def get_many(self, ids: Sequence[str]) -> list[Creator]:
        with self._conn.cursor() as cur:
            cur.execute(
                """select id, handle, platform, bio, tags, audience_countries, followers,
                          engagement_rate::float, rate_usd::float, past_performance::float
                   from creators where id = any(%s::text[])""", (list(ids),))
            rows = {r[0]: Creator(
                id=r[0], handle=r[1], platform=r[2], bio=r[3], tags=list(r[4]),
                audience_countries=r[5], followers=r[6], engagement_rate=r[7],
                rate_usd=r[8], past_performance=r[9]) for r in cur.fetchall()}
        return [rows[i] for i in ids]

    def __len__(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("select count(*) from creators")
            return int(cur.fetchone()[0])
