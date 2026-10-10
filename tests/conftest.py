import pytest

from legalrag import demo
from legalrag.index.bm25 import BM25Index
from legalrag.rag.pipeline import Pipeline


@pytest.fixture(scope="session")
def cases():
    return demo.cases()


@pytest.fixture(scope="session")
def store():
    return demo.store()


@pytest.fixture(scope="session")
def bm25(store):
    return BM25Index.build(store.texts)


@pytest.fixture()
def pipeline():
    return Pipeline.demo()
