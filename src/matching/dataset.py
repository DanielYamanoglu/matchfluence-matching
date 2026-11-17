from __future__ import annotations

import json
from pathlib import Path

from .models import Brand, Campaign, Creator

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _load(name: str, model):
    return [model(**row) for row in json.loads((DATA_DIR / f"{name}.json").read_text())]


def load_creators() -> list[Creator]:
    return _load("creators", Creator)


def load_campaigns() -> list[Campaign]:
    return _load("campaigns", Campaign)


def load_brands() -> list[Brand]:
    return _load("brands", Brand)


def load_relevance() -> dict[str, dict[str, int]]:
    return json.loads((DATA_DIR / "relevance.json").read_text())


def embedding_corpus(creators: list[Creator], campaigns: list[Campaign]) -> list[str]:
    """Texts the TF-IDF vocabulary is fitted on (creator profiles + campaign briefs)."""
    return [c.text for c in creators] + [f"{c.title}. {c.brief}" for c in campaigns]
