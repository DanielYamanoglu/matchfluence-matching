import os

os.environ["MATCHING_EMBEDDER"] = "tfidf"  # keep tests offline and deterministic

import pytest
from fastapi.testclient import TestClient

from matching.api import create_app


@pytest.fixture(scope="session")
def client():
    with TestClient(create_app("tfidf")) as c:
        yield c
