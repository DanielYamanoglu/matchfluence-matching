from matching.dataset import load_creators


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["creators"] == 800 and body["embedder"] == "tfidf"


def test_matches_for_seeded_campaign(client):
    r = client.post("/campaigns/cp_000/matches", json={"k": 5})
    assert r.status_code == 200
    body = r.json()
    assert len(body["matches"]) == 5
    top = body["matches"][0]
    assert top["rank"] == 1
    assert set(top["breakdown"]["components"]) >= {"semantic", "audience_overlap", "budget_fit"}
    assert top["explanation"]


def test_matches_respect_hard_filters(client):
    campaign = next(c for c in client.get("/campaigns").json() if c["id"] == "cp_000")
    ids = [m["creator_id"] for m in client.post("/campaigns/cp_000/matches", json={"k": 20}).json()["matches"]]
    creators = {c.id: c.model_dump() for c in load_creators()}
    for i in ids:
        c = creators[i]
        assert campaign["min_followers"] <= c["followers"] <= campaign["max_followers"]
        assert c["rate_usd"] <= campaign["budget_usd"]


def test_unknown_campaign_404(client):
    assert client.post("/campaigns/nope/matches", json={}).status_code == 404


def test_k_validation(client):
    assert client.post("/campaigns/cp_000/matches", json={"k": 0}).status_code == 422


def test_add_creator_then_it_can_be_matched(client):
    campaign = next(c for c in client.get("/campaigns").json() if c["id"] == "cp_001")
    payload = {
        "handle": "demo_new_creator", "bio": campaign["brief"], "tags": [],
        "audience_countries": {c: 0.9 for c in campaign["target_countries"][:1]},
        "followers": max(campaign["min_followers"], 10_000), "engagement_rate": 0.09,
        "rate_usd": campaign["budget_usd"] / 2, "past_performance": 0.9,
    }
    r = client.post("/creators", json=payload)
    assert r.status_code == 201
    new_id = r.json()["id"]
    ids = [m["creator_id"] for m in client.post("/campaigns/cp_001/matches", json={"k": 5}).json()["matches"]]
    assert new_id in ids


def test_openapi_lists_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/creators", "/campaigns/{campaign_id}/matches"} <= set(paths)
