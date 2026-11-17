import os

os.environ["MATCHING_EMBEDDER"] = "tfidf"  # keep tests offline and deterministic
os.environ.pop("DATABASE_URL", None)  # the API tests use the in-memory index

import pytest
from fastapi.testclient import TestClient

from matching.api import create_app


@pytest.fixture(scope="session")
def client():
    with TestClient(create_app("tfidf")) as c:
        yield c
