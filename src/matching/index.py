"""Vector indexes with one interface: in-memory numpy for the demo, pgvector for Postgres."""
from __future__ import annotations

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

    def search(self, query: np.ndarray, k: int, flt: CandidateFilter) -> list[Hit]: ...

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

    def get_many(self, ids: Sequence[str]) -> list[Creator]:
        return [self._creators[i] for i in ids]

    def __len__(self) -> int:
        return len(self._ids)


class PgVectorIndex:
    """Same interface backed by Postgres + pgvector (schema in supabase/migrations).

    Not exercised by the test suite (needs a live database); the SQL mirrors
    `match_creators` and `CandidateFilter`. Requires `psycopg[binary]`.
    """

    MODEL_ID = "demo"

    def __init__(self, dsn: str):
        import psycopg  # optional dependency

        self._conn = psycopg.connect(dsn, autocommit=True)

    @staticmethod
    def _vec(v: np.ndarray) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in np.asarray(v).ravel()) + "]"

    def upsert(self, creator: Creator, vector: np.ndarray) -> None:
        import json

        with self._conn.cursor() as cur:
            cur.execute(
                """insert into creators (id, handle, platform, bio, tags, audience_countries,
                       followers, engagement_rate, rate_usd, past_performance)
                   values (%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s)
                   on conflict (id) do update set bio = excluded.bio, tags = excluded.tags,
                       audience_countries = excluded.audience_countries,
                       followers = excluded.followers, engagement_rate = excluded.engagement_rate,
                       rate_usd = excluded.rate_usd, past_performance = excluded.past_performance""",
                (creator.id, creator.handle, creator.platform, creator.bio, creator.tags,
                 json.dumps(creator.audience_countries), creator.followers,
                 creator.engagement_rate, creator.rate_usd, creator.past_performance),
            )
            cur.execute(
                """insert into creator_embeddings (creator_id, model, embedding)
                   values (%s, %s, %s::vector)
                   on conflict (creator_id) do update set embedding = excluded.embedding,
                       model = excluded.model, updated_at = now()""",
                (creator.id, self.MODEL_ID, self._vec(vector)),
            )

    def search(self, query: np.ndarray, k: int, flt: CandidateFilter) -> list[Hit]:
        sql = """
            select cr.id, 1 - (ce.embedding <=> %(q)s::vector) as similarity
            from creators cr join creator_embeddings ce on ce.creator_id = cr.id
            where cr.followers between %(lo)s and %(hi)s
              and cr.rate_usd <= %(rate)s
              and (cardinality(%(countries)s::text[]) = 0 or exists (
                    select 1 from unnest(%(countries)s::text[]) tc
                    where coalesce((cr.audience_countries ->> tc)::numeric, 0) >= %(share)s))
            order by ce.embedding <=> %(q)s::vector
            limit %(k)s"""
        params = {
            "q": self._vec(query), "lo": flt.min_followers, "hi": flt.max_followers,
            "rate": None if flt.max_rate_usd == float("inf") else flt.max_rate_usd,
            "countries": list(flt.target_countries), "share": flt.min_audience_share, "k": k,
        }
        if params["rate"] is None:
            params["rate"] = 10**9
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [Hit(str(r[0]), float(r[1])) for r in cur.fetchall()]

    def get_many(self, ids: Sequence[str]) -> list[Creator]:
        with self._conn.cursor() as cur:
            cur.execute(
                """select id::text, handle, platform, bio, tags, audience_countries, followers,
                          engagement_rate::float, rate_usd::float, past_performance::float
                   from creators where id = any(%s::uuid[])""", (list(ids),))
            rows = {str(r[0]): Creator(
                id=str(r[0]), handle=r[1], platform=r[2], bio=r[3], tags=list(r[4]),
                audience_countries=r[5], followers=r[6], engagement_rate=r[7],
                rate_usd=r[8], past_performance=r[9]) for r in cur.fetchall()}
        return [rows[i] for i in ids]

    def __len__(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("select count(*) from creators")
            return int(cur.fetchone()[0])
