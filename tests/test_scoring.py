import pytest

from matching.models import Campaign, Creator
from matching.scoring import DEFAULT_WEIGHTS, SEMANTIC_ONLY, audience_overlap, budget_fit, explain, score


def creator(**kw):
    base = dict(id="c1", handle="h", bio="b", followers=50_000, engagement_rate=0.08,
                rate_usd=500, past_performance=0.5, audience_countries={"US": 0.6, "GB": 0.4})
    return Creator(**{**base, **kw})


CAMPAIGN = Campaign(id="p1", brand_id="b1", title="t", brief="x", target_countries=["US"], budget_usd=1000)


def test_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.as_dict().values()) == pytest.approx(1.0)


def test_audience_overlap_sums_target_countries():
    assert audience_overlap(creator(), CAMPAIGN) == pytest.approx(0.6)
    both = CAMPAIGN.model_copy(update={"target_countries": ["US", "GB"]})
    assert audience_overlap(creator(), both) == pytest.approx(1.0)


def test_budget_fit_over_budget_is_zero_and_half_budget_is_full():
    assert budget_fit(creator(rate_usd=1200), CAMPAIGN) == 0.0
    assert budget_fit(creator(rate_usd=500), CAMPAIGN) == pytest.approx(1.0)
    assert 0.6 < budget_fit(creator(rate_usd=100), CAMPAIGN) < 1.0


def test_breakdown_contributions_sum_to_total():
    b = score(0.7, creator(), CAMPAIGN)
    assert b.total == pytest.approx(sum(c.contribution for c in b.components.values()), abs=1e-3)
    assert set(b.components) == {"semantic", "audience_overlap", "engagement",
                                 "budget_fit", "past_performance"}


def test_semantic_only_ignores_rule_features():
    a = score(0.5, creator(engagement_rate=0.01), CAMPAIGN, SEMANTIC_ONLY)
    b = score(0.5, creator(engagement_rate=0.2), CAMPAIGN, SEMANTIC_ONLY)
    assert a.total == b.total == 0.5


def test_higher_engagement_scores_higher_all_else_equal():
    lo = score(0.5, creator(engagement_rate=0.02), CAMPAIGN).total
    hi = score(0.5, creator(engagement_rate=0.08), CAMPAIGN).total
    assert hi > lo


def test_explanation_mentions_strongest_signal():
    assert "Strongest signals" in explain(score(0.9, creator(), CAMPAIGN))
