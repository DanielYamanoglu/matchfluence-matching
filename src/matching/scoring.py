"""Hybrid score = semantic similarity + rule features, with a per-feature breakdown."""
from __future__ import annotations

from dataclasses import dataclass

from .models import Campaign, Component, Creator, ScoreBreakdown

ER_BENCHMARK = 0.08  # engagement rate that earns full marks


@dataclass(frozen=True)
class Weights:
    semantic: float = 0.50
    audience_overlap: float = 0.20
    engagement: float = 0.10
    budget_fit: float = 0.10
    past_performance: float = 0.10

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


DEFAULT_WEIGHTS = Weights()
SEMANTIC_ONLY = Weights(1.0, 0.0, 0.0, 0.0, 0.0)


def audience_overlap(cr: Creator, campaign: Campaign) -> float:
    """Share of the creator's audience located in the campaign's target countries."""
    if not campaign.target_countries:
        return 1.0
    return min(1.0, sum(cr.audience_countries.get(c, 0.0) for c in campaign.target_countries))


def engagement(cr: Creator) -> float:
    return min(1.0, cr.engagement_rate / ER_BENCHMARK)


def budget_fit(cr: Creator, campaign: Campaign) -> float:
    """1.0 when the rate uses at least half the budget; cheaper is a bit lower
    (very low rates correlate with low reach); over budget scores 0."""
    if cr.rate_usd > campaign.budget_usd:
        return 0.0
    return 0.6 + 0.4 * min(1.0, cr.rate_usd / (0.5 * campaign.budget_usd))


def rule_features(cr: Creator, campaign: Campaign) -> dict[str, float]:
    return {
        "audience_overlap": audience_overlap(cr, campaign),
        "engagement": engagement(cr),
        "budget_fit": budget_fit(cr, campaign),
        "past_performance": cr.past_performance,
    }


def score(semantic: float, cr: Creator, campaign: Campaign,
          weights: Weights = DEFAULT_WEIGHTS) -> ScoreBreakdown:
    """`semantic` must already be scaled to [0, 1] (see Matcher)."""
    values = {"semantic": max(0.0, min(1.0, semantic)), **rule_features(cr, campaign)}
    w = weights.as_dict()
    comps = {n: Component(value=round(v, 4), weight=w[n], contribution=round(v * w[n], 4))
             for n, v in values.items()}
    return ScoreBreakdown(total=round(sum(v * w[n] for n, v in values.items()), 4), components=comps)


def explain(b: ScoreBreakdown) -> str:
    """One sentence naming the two strongest contributors and the weakest feature."""
    ranked = sorted(b.components.items(), key=lambda kv: kv[1].contribution, reverse=True)
    label = {"semantic": "brief/profile similarity", "audience_overlap": "audience overlap",
             "engagement": "engagement", "budget_fit": "budget fit",
             "past_performance": "past performance"}
    top = ", ".join(f"{label[n]} ({c.value:.2f})" for n, c in ranked[:2])
    weakest = min((kv for kv in b.components.items() if kv[1].weight > 0),
                  key=lambda kv: kv[1].value)
    return f"Strongest signals: {top}. Weakest: {label[weakest[0]]} ({weakest[1].value:.2f})."
