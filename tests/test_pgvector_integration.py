"""Postgres + pgvector path vs the in-memory index, on the full synthetic dataset.

Runs only when TEST_DATABASE_URL points at a THROWAWAY database: the fixture drops and
recreates the `public` and `auth` schemas, applies the migrations and seeds the data.
    docker compose up -d --wait
    TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5433/matching pytest -q
"""
import os

import pytest

DSN = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DSN, reason="TEST_DATABASE_URL not set")

from matching.api import build_state  # noqa: E402
from matching.dataset import embedding_corpus, load_campaigns, load_creators  # noqa: E402
from matching.embedding import TfidfEmbedder  # noqa: E402
from matching.index import CandidateFilter, InMemoryIndex  # noqa: E402
from matching.matcher import OVERFETCH, Matcher  # noqa: E402

SIM_TOL = 1e-6  # float32 dot product (NumPy) vs pgvector cosine distance; observed < 1e-7
K = 10
CREATORS, CAMPAIGNS = load_creators(), load_campaigns()
BY_ID = {c.id: c for c in CREATORS}


@pytest.fixture(scope="module")
def matchers():
    pytest.importorskip("psycopg")
    from matching.pgdb import apply_schema, seed

    embedder = TfidfEmbedder(embedding_corpus(CREATORS, CAMPAIGNS))
    assert apply_schema(DSN, reset=True)
    pg = seed(DSN, embedder)
    mem = Matcher(embedder, InMemoryIndex())
    mem.add_creators(CREATORS)
    yield mem, pg
    pg.index.close()


def filter_variants():
    """Campaigns with hard filters chosen to bind: narrow follower band, low budget,
    several countries, and none at all (stored under new ids via the upsert path)."""
    base = CAMPAIGNS[0]
    return [
        base.model_copy(update={"id": "cp_t_band", "min_followers": 50_000,
                                "max_followers": 120_000}),
        base.model_copy(update={"id": "cp_t_cheap", "budget_usd": 300.0}),
        base.model_copy(update={"id": "cp_t_multi", "target_countries": ["DE", "TR", "BR"]}),
        base.model_copy(update={"id": "cp_t_none", "target_countries": [], "min_followers": 0,
                                "max_followers": 2_000_000_000, "budget_usd": 1e6}),
    ]


def assert_same_hits(mem_hits, pg_hits):
    # Exact id order, ties included: TF-IDF yields many exactly equal similarities, and both
    # sides break them by creator id (stable sort over id-ordered rows vs `order by ..., cr.id`).
    assert [h.creator_id for h in pg_hits] == [h.creator_id for h in mem_hits]
    assert [h.similarity for h in pg_hits] == pytest.approx(
        [h.similarity for h in mem_hits], abs=SIM_TOL)


@pytest.mark.parametrize("campaign", CAMPAIGNS + filter_variants(), ids=lambda c: c.id)
def test_match_creators_equals_in_memory_retrieval(matchers, campaign):
    mem, pg = matchers
    q = mem.embed_brief(campaign)
    mem_hits = mem.index.search_campaign(campaign, q, K * OVERFETCH)
    pg_hits = pg.index.search_campaign(campaign, q, K * OVERFETCH)
    assert_same_hits(mem_hits, pg_hits)
    flt = CandidateFilter.from_campaign(campaign)
    assert all(flt.allows(BY_ID[h.creator_id]) for h in pg_hits)


def test_filter_variants_actually_restrict_the_pool(matchers):
    mem, pg = matchers
    unfiltered = {h.creator_id for h in pg.index.search_campaign(
        filter_variants()[-1], mem.embed_brief(CAMPAIGNS[0]), K)}
    for variant in filter_variants()[:-1]:
        flt = CandidateFilter.from_campaign(variant)
        pool = [c for c in CREATORS if flt.allows(c)]
        assert 0 < len(pool) < len(CREATORS)
        got = {h.creator_id for h in pg.index.search_campaign(variant, mem.embed_brief(variant), K)}
        assert got != unfiltered


@pytest.mark.parametrize("campaign", CAMPAIGNS, ids=lambda c: c.id)
def test_final_ranking_equals_in_memory(matchers, campaign):
    mem, pg = matchers
    a, b = mem.match(campaign, K), pg.match(campaign, K)
    assert [m.creator_id for m in b] == [m.creator_id for m in a]
    assert [m.score for m in b] == pytest.approx([m.score for m in a], abs=2e-4)


def test_api_state_selects_postgres_from_database_url(matchers, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", DSN)
    s = build_state("tfidf")
    assert s["backend"] == "postgres" and len(s["matcher"].index) == len(CREATORS)
    mem, _ = matchers
    for cid in ("cp_000", "cp_001", "cp_002"):
        campaign = s["campaigns"][cid]
        assert s["matcher"].match(campaign, 1)[0].creator_id == mem.match(campaign, 1)[0].creator_id
    s["matcher"].index.close()
