"""Generate fully synthetic, fictional brands, creators, campaigns and relevance labels.

Deterministic: fixed seed, so `python data/generate.py` reproduces the committed JSON.
Handles and brand names are randomly generated and not real accounts or companies.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from faker import Faker

SEED = 7
N_CREATORS, N_BRANDS, N_CAMPAIGNS = 800, 40, 80
OUT = Path(__file__).parent
COUNTRIES = ["US", "GB", "DE", "TR", "FR", "CA", "BR", "ES"]

# niche -> (topical words, activities, product types). Related niches share some vocabulary,
# so a pure text model can confuse them.
NICHES: dict[str, dict[str, list[str]]] = {
    "fitness": {"w": ["workout", "gym", "strength", "training", "hiit", "mobility", "athlete", "protein"],
                "a": ["lift weights", "run marathons", "coach home workouts"], "p": ["resistance bands", "protein powder", "running shoes"]},
    "nutrition": {"w": ["recipes", "healthy", "meal prep", "protein", "plant-based", "wellness", "diet", "cooking"],
                  "a": ["share meal prep plans", "review healthy snacks", "cook plant-based dinners"], "p": ["meal kits", "superfood snacks", "vitamin supplements"]},
    "beauty": {"w": ["skincare", "makeup", "routine", "glow", "serum", "cosmetics", "tutorial", "haircare"],
               "a": ["post skincare routines", "review makeup palettes", "teach hair tutorials"], "p": ["facial serum", "makeup palette", "hair oil"]},
    "fashion": {"w": ["outfit", "style", "streetwear", "wardrobe", "vintage", "lookbook", "sustainable", "denim"],
                "a": ["post daily outfits", "thrift vintage pieces", "style streetwear looks"], "p": ["denim jackets", "sneakers", "capsule wardrobe pieces"]},
    "gaming": {"w": ["gameplay", "esports", "streaming", "console", "pc build", "rpg", "multiplayer", "speedrun"],
               "a": ["stream ranked matches", "review new releases", "build gaming pcs"], "p": ["gaming headsets", "mechanical keyboards", "energy drinks"]},
    "tech": {"w": ["gadgets", "review", "smartphone", "software", "ai tools", "laptop", "setup", "coding"],
             "a": ["review gadgets", "explain ai tools", "show desk setups"], "p": ["wireless earbuds", "productivity apps", "laptop stands"]},
    "travel": {"w": ["adventure", "hotels", "backpacking", "itinerary", "city guide", "road trip", "hiking", "destinations"],
               "a": ["document road trips", "guide city weekends", "go backpacking"], "p": ["travel backpacks", "booking apps", "luggage"]},
    "parenting": {"w": ["toddler", "family", "newborn", "kids", "school", "mom life", "activities", "baby"],
                  "a": ["share toddler activities", "review baby gear", "document family life"], "p": ["baby carriers", "kids snacks", "educational toys"]},
    "finance": {"w": ["investing", "budgeting", "savings", "side hustle", "stocks", "credit", "frugal", "money"],
                "a": ["explain budgeting", "teach investing basics", "review money apps"], "p": ["budgeting apps", "investment platforms", "credit cards"]},
    "home": {"w": ["interior", "diy", "decor", "renovation", "plants", "kitchen", "organizing", "furniture"],
             "a": ["renovate apartments", "style interiors", "teach diy projects"], "p": ["houseplants", "kitchen gadgets", "storage furniture"]},
}
RELATED = {frozenset(p) for p in [("fitness", "nutrition"), ("beauty", "fashion"), ("gaming", "tech"),
                                  ("travel", "fashion"), ("parenting", "home"), ("finance", "tech"),
                                  ("nutrition", "home")]}
GENERIC = ["content creator", "daily life", "good vibes", "collabs open", "storytelling", "behind the scenes",
           "authentic", "community", "weekend", "inspiration", "creative", "original", "trending"]
PLATFORMS = ["instagram", "tiktok", "youtube"]
SYL = ["lu", "mo", "ra", "ven", "tal", "zi", "ko", "mar", "bel", "dur", "nex", "or", "qui", "sa", "tri", "van"]
SUFFIX = ["Labs", "Co", "Goods", "Studio", "Works", "Supply", "Collective"]


def related(a: str, b: str) -> bool:
    return frozenset((a, b)) in RELATED


def brand_name(rng: random.Random) -> str:
    return "".join(rng.choice(SYL) for _ in range(rng.choice([2, 3]))).title() + " " + rng.choice(SUFFIX)


def make_creator(i: int, rng: random.Random, fake: Faker) -> dict:
    niche = rng.choice(list(NICHES))
    secondary = rng.choice([n for n in NICHES if n != niche]) if rng.random() < 0.4 else None
    # ~20% of creators have a vague, off-topic bio: semantic models can't see their niche.
    vague = rng.random() < 0.20
    words = NICHES[niche]["w"]
    tags = rng.sample(GENERIC, 3) if vague else rng.sample(words, 3)
    bio_words = rng.sample(GENERIC, 4) if vague else rng.sample(words, 4)
    bio = f"{rng.choice(['Passionate', 'Curious', 'Full-time', 'Weekend', 'Self-taught'])} {rng.choice(['creator', 'storyteller', 'enthusiast'])} "\
          f"sharing {', '.join(bio_words)}."
    if secondary and not vague:
        bio += f" Also into {rng.choice(NICHES[secondary]['w'])} and {rng.choice(NICHES[secondary]['w'])}."
        tags.append(rng.choice(NICHES[secondary]["w"]))
    followers = int(min(2_000_000, max(5_000, math.exp(rng.gauss(10.3, 1.1)))))
    primary = rng.choice(COUNTRIES)
    share = round(rng.uniform(0.4, 0.8), 2)
    others = rng.sample([c for c in COUNTRIES if c != primary], 2)
    second = round((1 - share) * rng.uniform(0.4, 0.7), 2)
    audience = {primary: share, others[0]: second, others[1]: round(1 - share - second, 2)}
    # engagement tends to fall with size; plus noise
    er = max(0.005, min(0.20, rng.gauss(0.075 - 0.007 * math.log10(followers / 5000) * 3, 0.02)))
    return {
        "id": f"cr_{i:04d}", "handle": f"demo_creator_{i:04d}",
        "platform": rng.choice(PLATFORMS), "bio": bio, "tags": tags, "audience_countries": audience,
        "followers": followers, "engagement_rate": round(er, 4),
        "rate_usd": round(followers * 0.01 * rng.uniform(0.5, 1.5), 2),
        "past_performance": round(min(1.0, max(0.0, rng.betavariate(5, 3))), 3),
        "niche": niche, "secondary_niche": secondary,
    }


def make_campaign(i: int, brand_ids: list[str], rng: random.Random) -> dict:
    niche = rng.choice(list(NICHES))
    spec = NICHES[niche]
    words = ", ".join(rng.sample(spec["w"], 3))
    tone = rng.choice(["authentic", "educational", "fun", "premium", "down-to-earth"])
    brief = (f"We are launching {rng.choice(spec['p'])} and want creators who {rng.choice(spec['a'])}. "
             f"Looking for {tone} content about {words}.")
    lo = rng.choice([5_000, 10_000, 25_000, 50_000])
    return {
        "id": f"cp_{i:03d}", "brand_id": rng.choice(brand_ids),
        "title": f"{niche.title()} launch {i:03d}", "brief": brief,
        "target_countries": rng.sample(COUNTRIES, rng.choice([1, 2])),
        "min_followers": lo, "max_followers": lo * rng.choice([8, 20, 50]),
        "budget_usd": float(rng.choice([800, 1500, 3000, 6000, 12000])), "niche": niche,
    }


def relevance(cr: dict, cp: dict, rng: random.Random) -> int:
    """Graded 0-3 label from latent attributes the text never states directly, plus noise."""
    if cr["niche"] == cp["niche"]:
        topic = 1.0
    elif cr["secondary_niche"] == cp["niche"]:
        topic = 0.6
    elif related(cr["niche"], cp["niche"]):
        topic = 0.4
    else:
        topic = 0.0
    aud = min(1.0, sum(cr["audience_countries"].get(c, 0) for c in cp["target_countries"]))
    eng = min(1.0, cr["engagement_rate"] / 0.08)
    fit = min(1.0, cr["rate_usd"] / (0.5 * cp["budget_usd"])) if cr["rate_usd"] <= cp["budget_usd"] else 0.0
    s = 0.55 * topic + 0.15 * aud + 0.15 * eng + 0.05 * fit + 0.10 * cr["past_performance"] + rng.gauss(0, 0.05)
    return 3 if s >= 0.80 else 2 if s >= 0.65 else 1 if s >= 0.45 else 0


def main() -> None:
    rng, fake = random.Random(SEED), Faker()
    Faker.seed(SEED)
    brands = [{"id": f"br_{i:03d}", "name": brand_name(rng), "industry": rng.choice(list(NICHES))}
              for i in range(N_BRANDS)]
    creators = [make_creator(i, rng, fake) for i in range(N_CREATORS)]
    campaigns = [make_campaign(i, [b["id"] for b in brands], rng) for i in range(N_CAMPAIGNS)]
    labels = {}
    for cp in campaigns:  # only non-zero grades are stored; a missing creator means grade 0
        grades = {cr["id"]: relevance(cr, cp, rng) for cr in creators}
        labels[cp["id"]] = {k: v for k, v in grades.items() if v > 0}
    for name, obj in [("brands", brands), ("creators", creators), ("campaigns", campaigns),
                      ("relevance", labels)]:
        (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1) + "\n")
    print(f"wrote {len(brands)} brands, {len(creators)} creators, {len(campaigns)} campaigns")


if __name__ == "__main__":
    main()
