"""FastAPI service. Run: uvicorn matching.api:app --app-dir src"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .dataset import load_campaigns, load_creators
from .embedding import get_embedder
from .index import InMemoryIndex
from .matcher import Matcher, creator_from_input
from .models import Campaign, Creator, CreatorIn, Match


class MatchRequest(BaseModel):
    k: int = Field(default=10, ge=1, le=100, description="Number of matches to return")


class MatchResponse(BaseModel):
    campaign_id: str
    embedder: str
    matches: list[Match]


def build_state(embedder_pref: str | None = None) -> dict:
    creators, campaigns = load_creators(), load_campaigns()
    corpus = [c.text for c in creators] + [f"{c.title}. {c.brief}" for c in campaigns]
    matcher = Matcher(get_embedder(corpus, embedder_pref), InMemoryIndex())
    matcher.add_creators(creators)
    return {"matcher": matcher, "campaigns": {c.id: c for c in campaigns}}


def create_app(embedder_pref: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.s = build_state(embedder_pref)
        yield

    app = FastAPI(
        title="Creator matching demo",
        version="0.1.0",
        description="Embedding retrieval + hybrid re-rank for brand/creator matching "
                    "(synthetic data, standalone demo).",
        lifespan=lifespan,
    )

    def state(request: Request) -> dict:
        return request.app.state.s

    @app.get("/health", tags=["meta"])
    def health(s: dict = Depends(state)):
        m: Matcher = s["matcher"]
        return {"status": "ok", "creators": len(m.index), "embedder": m.embedder.name}

    @app.get("/campaigns", response_model=list[Campaign], tags=["campaigns"])
    def list_campaigns(s: dict = Depends(state)):
        return list(s["campaigns"].values())

    @app.post("/campaigns/{campaign_id}/matches", response_model=MatchResponse, tags=["matching"])
    def match(campaign_id: str, body: MatchRequest | None = None, s: dict = Depends(state)):
        campaign = s["campaigns"].get(campaign_id)
        if campaign is None:
            raise HTTPException(404, f"campaign {campaign_id!r} not found")
        m: Matcher = s["matcher"]
        matches = m.match(campaign, k=(body or MatchRequest()).k)
        return MatchResponse(campaign_id=campaign_id, embedder=m.embedder.name,
                             matches=matches)

    @app.post("/creators", response_model=Creator, status_code=201, tags=["creators"])
    def add_creator(body: CreatorIn, s: dict = Depends(state)):
        creator = creator_from_input(body, f"cr_{uuid.uuid4().hex[:8]}")
        s["matcher"].add_creator(creator)
        return creator

    return app


app = create_app()
