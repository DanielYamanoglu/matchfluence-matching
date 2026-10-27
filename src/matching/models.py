from __future__ import annotations

from pydantic import BaseModel, Field


class CreatorIn(BaseModel):
    handle: str
    platform: str = "instagram"
    bio: str
    tags: list[str] = Field(default_factory=list)
    audience_countries: dict[str, float] = Field(default_factory=dict)
    followers: int = Field(ge=0)
    engagement_rate: float = Field(ge=0, le=1)
    rate_usd: float = Field(ge=0)
    past_performance: float = Field(default=0.5, ge=0, le=1)
    # Latent label used only by the synthetic ground truth; never embedded or scored.
    niche: str | None = None
    secondary_niche: str | None = None


class Creator(CreatorIn):
    id: str

    @property
    def text(self) -> str:
        """Text that gets embedded for this creator."""
        return f"{self.bio} Topics: {', '.join(self.tags)}."


class Campaign(BaseModel):
    id: str
    brand_id: str
    title: str
    brief: str
    target_countries: list[str] = Field(default_factory=list)
    min_followers: int = 0
    max_followers: int = 2_000_000_000
    budget_usd: float = Field(gt=0, description="Budget per creator")
    niche: str | None = None  # latent label, ground truth only


class Brand(BaseModel):
    id: str
    name: str
    industry: str


class Component(BaseModel):
    value: float = Field(description="Feature value in [0, 1]")
    weight: float
    contribution: float = Field(description="value * weight")


class ScoreBreakdown(BaseModel):
    total: float
    components: dict[str, Component]


class Match(BaseModel):
    rank: int
    creator_id: str
    handle: str
    score: float
    similarity: float = Field(description="Raw cosine similarity of brief and profile")
    breakdown: ScoreBreakdown
    explanation: str
