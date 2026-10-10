import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from legalrag.api import service
from legalrag.api.app import app
from legalrag.rag.pipeline import Pipeline


@pytest.fixture(scope="module")
def client():
    service.pipeline.cache_clear()
    demo = Pipeline.demo()
    service.pipeline = lambda: demo  # the synthetic corpus, whatever is on disk
    return TestClient(app)


def test_health_states_scope_and_limits(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["demo_corpus"] is True and body["cases"] == 8
    assert "not legal advice" in body["limitation"] and "no citator" in body["authority_notice"]


def test_search_ranks_and_filters(client):
    body = client.post(
        "/search",
        json={"question": "irreparable harm preliminary injunction", "binding_on": "2", "k": 5},
    ).json()
    assert body["cases"][0]["case_id"] == "cap:9006" and body["cases"][0]["authority"] == "binding"
    assert all(c["circuit"] in ("2", None) for c in body["cases"])
    assert "not legal advice" in body["limitation"]


def test_date_filter(client):
    q = "which prong of qualified immunity should be addressed first"
    late = client.post("/search", json={"question": q}).json()["cases"]
    early = client.post("/search", json={"question": q, "before": "2005-01-01"}).json()["cases"]
    assert late[0]["case_id"] == "cap:9004" and all(
        c["decision_date"] < "2005-01-01" for c in early
    )


def test_answer_has_cited_verified_statements(client):
    body = client.post(
        "/answer", json={"question": "what must a plaintiff show for standing"}
    ).json()
    assert not body["abstained"] and body["evidence"] and body["verification"]["judge"]
    cited = [s for s in body["statements"] if s["citations"]]
    assert cited and all(s["verification"]["status"] == "verified" for s in cited)


def test_answer_abstains_out_of_scope(client):
    body = client.post(
        "/answer", json={"question": "What is the right to erasure under the GDPR?"}
    ).json()
    assert body["abstained"] and not body["statements"]


def test_report_download(client):
    r = client.post(
        "/report", json={"question": "summary judgment genuine dispute of material fact"}
    )
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/markdown")
    assert "attachment" in r.headers["content-disposition"] and "# Legal research report" in r.text


def test_citation_validation_separates_three_outcomes(client):
    text = "See Alder, 9990 U.S. 242, 250; Smith v. Jones, 410 U.S. 113; and 12 Xyz.3d 4."
    body = client.post("/citations/validate", json={"text": text}).json()
    status = {c["normalised"]: c["status"] for c in body["citations"]}
    assert status == {"9990 U.S. 242": "in_corpus", "410 U.S. 113": "not_in_corpus"}
    assert "not thereby shown to support" in body["note"]


def test_case_inspection_and_404(client):
    body = client.get("/cases/cap:9003").json()
    assert body["treatment"] is None  # synthetic treatment is pipeline-level, not in the real table
    assert any(p["opinion_type"] == "majority" for p in body["paragraphs"])
    assert client.get("/cases/cap:nope").status_code == 404


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "x"},
        {"question": "standing", "before": "2005"},
        {"question": "standing", "binding_on": "12"},
        {"question": "standing", "k": 0},
    ],
)
def test_validation_errors(client, payload):
    path = "/search" if "k" in payload else "/answer"
    assert client.post(path, json=payload).status_code == 422
