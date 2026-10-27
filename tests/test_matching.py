import numpy as np

from matching.embedding import DIM, TfidfEmbedder
from matching.index import CandidateFilter, InMemoryIndex
from matching.matcher import Matcher
from matching.models import Campaign, Creator

CORPUS = ["skincare serum makeup routine", "gym workout strength training", "gaming esports stream"]


def mk(i, bio, **kw):
    base = dict(id=f"c{i}", handle=f"h{i}", bio=bio, followers=50_000, engagement_rate=0.05,
                rate_usd=400, past_performance=0.5, audience_countries={"US": 0.7, "DE": 0.3})
    return Creator(**{**base, **kw})


def campaign(**kw):
    base = dict(id="p", brand_id="b", title="Launch", brief="skincare serum routine",
                target_countries=["US"], budget_usd=1000)
    return Campaign(**{**base, **kw})


def test_tfidf_is_deterministic_normalised_and_fixed_dim():
    e1, e2 = TfidfEmbedder(CORPUS), TfidfEmbedder(CORPUS)
    a, b = e1.encode(CORPUS), e2.encode(CORPUS)
    assert a.shape == (3, DIM)
    assert np.allclose(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)


def build(creators):
    m = Matcher(TfidfEmbedder(CORPUS + [c.text for c in creators]), InMemoryIndex())
    m.add_creators(creators)
    return m


def test_semantically_closest_creator_ranks_first():
    m = build([mk(1, "gaming esports stream"), mk(2, "skincare serum makeup routine"),
               mk(3, "gym workout strength")])
    assert m.match(campaign(), k=3)[0].creator_id == "c2"


def test_filters_exclude_candidates():
    creators = [
        mk(1, "skincare serum routine"),
        mk(2, "skincare serum routine", followers=500),                       # too small
        mk(3, "skincare serum routine", rate_usd=5000),                       # over budget
        mk(4, "skincare serum routine", audience_countries={"TR": 0.9}),      # wrong country
    ]
    ids = {x.creator_id for x in build(creators).match(campaign(min_followers=10_000), k=10)}
    assert ids == {"c1"}


def test_filter_object_directly():
    f = CandidateFilter.from_campaign(campaign(max_followers=60_000))
    assert f.allows(mk(1, "x"))
    assert not f.allows(mk(2, "x", followers=70_000))


def test_empty_pool_returns_no_matches():
    assert build([mk(1, "skincare", rate_usd=9999)]).match(campaign(), k=5) == []


def test_upsert_replaces_existing_creator():
    idx = InMemoryIndex()
    e = TfidfEmbedder(CORPUS)
    idx.upsert(mk(1, "x"), e.encode(["gaming esports"])[0])
    idx.upsert(mk(1, "x"), e.encode(["skincare serum"])[0])
    assert len(idx) == 1
    hit = idx.search(e.encode(["skincare serum"])[0], 1, CandidateFilter())[0]
    assert hit.similarity > 0.99


def test_results_sorted_and_ranked():
    creators = [mk(i, "skincare serum routine", engagement_rate=0.01 * i) for i in range(1, 6)]
    res = build(creators).match(campaign(), k=5)
    scores = [r.score for r in res]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in res] == [1, 2, 3, 4, 5]
