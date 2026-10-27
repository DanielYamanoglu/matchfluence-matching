"""Two-stage matcher: vector retrieval with hard filters, then hybrid re-rank."""
from __future__ import annotations

from .embedding import Embedder
from .index import CandidateFilter, VectorIndex
from .models import Campaign, Creator, CreatorIn, Match
from .scoring import DEFAULT_WEIGHTS, Weights, explain, score

OVERFETCH = 5  # retrieve k * OVERFETCH candidates, re-rank, keep k


class Matcher:
    def __init__(self, embedder: Embedder, index: VectorIndex,
                 weights: Weights = DEFAULT_WEIGHTS):
        self.embedder, self.index, self.weights = embedder, index, weights

    def add_creator(self, creator: Creator) -> None:
        self.index.upsert(creator, self.embedder.encode([creator.text])[0])

    def add_creators(self, creators: list[Creator]) -> None:
        for cr, vec in zip(creators, self.embedder.encode([c.text for c in creators])):
            self.index.upsert(cr, vec)

    def embed_brief(self, campaign: Campaign):
        return self.embedder.encode([f"{campaign.title}. {campaign.brief}"])[0]

    def match(self, campaign: Campaign, k: int = 10, weights: Weights | None = None) -> list[Match]:
        w = weights or self.weights
        hits = self.index.search(self.embed_brief(campaign), k * OVERFETCH,
                                 CandidateFilter.from_campaign(campaign))
        if not hits:
            return []
        creators = {c.id: c for c in self.index.get_many([h.creator_id for h in hits])}
        sims = [h.similarity for h in hits]
        lo, hi = min(sims), max(sims)
        span = hi - lo
        ranked = []
        for h in hits:
            # min-max scale within the retrieved pool so semantic and rule features share a range
            sem = (h.similarity - lo) / span if span > 1e-9 else 1.0
            ranked.append((score(sem, creators[h.creator_id], campaign, w), h))
        ranked.sort(key=lambda t: t[0].total, reverse=True)
        return [
            Match(rank=i + 1, creator_id=h.creator_id, handle=creators[h.creator_id].handle,
                  score=b.total, similarity=round(h.similarity, 4), breakdown=b,
                  explanation=explain(b))
            for i, (b, h) in enumerate(ranked[:k])
        ]


def creator_from_input(data: CreatorIn, creator_id: str) -> Creator:
    return Creator(id=creator_id, **data.model_dump())
