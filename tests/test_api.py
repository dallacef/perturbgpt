"""Endpoint contract tests for the FastAPI service.

All heavy dependencies are stubbed: no checkpoint, FAISS index, LLM or
network access is required.  Tests are skipped entirely when FastAPI is not
installed (it is an optional extra).
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from perturbgpt.agent.orchestrator import AgentResponse, ToolCall  # noqa: E402
from perturbgpt.api import main as api_main  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(api_main.app)


@pytest.fixture()
def patched_artifacts(tmp_path, monkeypatch):
    """Pretend the checkpoint and gene embeddings exist."""
    ckpt = tmp_path / "film_head.pt"
    ckpt.write_bytes(b"not-a-real-checkpoint")
    emb = tmp_path / "gene_embeddings.npz"
    emb.write_bytes(b"not-a-real-npz")
    monkeypatch.setattr(api_main.tools, "DEFAULT_CHECKPOINT", ckpt)
    monkeypatch.setattr(api_main.tools, "DEFAULT_GENE_EMBEDDINGS", emb)
    return ckpt, emb


# ------------------------------------------------------------------ /health


def test_health_returns_status_and_artifacts(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert set(body["artifacts"]) >= {
        "checkpoint", "gene_embeddings", "perturbation_index",
        "response_index", "literature_index", "pathway_sets",
    }
    assert all(isinstance(v, bool) for v in body["artifacts"].values())


# ----------------------------------------------------------------- /predict


def test_predict_returns_ranked_genes(client, patched_artifacts, monkeypatch):
    monkeypatch.setattr(
        api_main.tools,
        "get_predicted_genes",
        lambda perturbation_id, top_k=20: [
            {"gene": "HBB", "delta": 1.5},
            {"gene": "HBA1", "delta": 1.1},
        ][:top_k],
    )
    response = client.post("/predict", json={"perturbation_id": "KLF1", "top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["perturbation_id"] == "KLF1"
    assert [g["gene"] for g in body["genes"]] == ["HBB", "HBA1"]
    assert body["checkpoint"].endswith("film_head.pt")


def test_predict_503_when_checkpoint_missing(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api_main.tools, "DEFAULT_CHECKPOINT", tmp_path / "absent.pt")
    response = client.post("/predict", json={"perturbation_id": "KLF1"})
    assert response.status_code == 503
    assert "checkpoint" in response.json()["detail"].lower()


def test_predict_503_when_model_returns_nothing(client, patched_artifacts, monkeypatch):
    monkeypatch.setattr(
        api_main.tools, "get_predicted_genes", lambda perturbation_id, top_k=20: []
    )
    response = client.post("/predict", json={"perturbation_id": "KLF1"})
    assert response.status_code == 503


@pytest.mark.parametrize("payload", [
    {"perturbation_id": "KLF1", "top_k": 0},
    {"perturbation_id": "KLF1", "top_k": 101},
    {"perturbation_id": "", "top_k": 10},
    {"perturbation_id": "KLF1", "unexpected": 1},
])
def test_predict_validation_errors(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


# --------------------------------------------------- /similar-perturbations


class _FakeIndex:
    ids = ["KLF1", "GATA1"]


@pytest.fixture()
def fake_index(monkeypatch):
    monkeypatch.setattr(api_main.tools, "load_index", lambda index_type="perturbation": _FakeIndex())
    monkeypatch.setattr(
        api_main.tools,
        "find_similar_perturbations",
        lambda perturbation_id, top_k=10, index_type="perturbation": [
            {"pert_id": "GATA1", "score": 0.91, "gene_symbols": ["GATA1"], "cell_count": 42}
        ],
    )


def test_similar_perturbations_happy_path(client, fake_index):
    response = client.post(
        "/similar-perturbations",
        json={"perturbation_id": "KLF1", "top_k": 5, "index_type": "perturbation"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "KLF1"
    assert body["results"][0]["pert_id"] == "GATA1"
    assert body["results"][0]["cell_count"] == 42


def test_similar_perturbations_404_for_unknown_query(client, fake_index):
    response = client.post(
        "/similar-perturbations", json={"perturbation_id": "NOT_IN_INDEX"}
    )
    assert response.status_code == 404


def test_similar_perturbations_503_without_index(client, monkeypatch):
    monkeypatch.setattr(api_main.tools, "load_index", lambda index_type="perturbation": None)
    response = client.post("/similar-perturbations", json={"perturbation_id": "KLF1"})
    assert response.status_code == 503


def test_similar_perturbations_rejects_bad_index_type(client):
    response = client.post(
        "/similar-perturbations",
        json={"perturbation_id": "KLF1", "index_type": "magic"},
    )
    assert response.status_code == 422


# ----------------------------------------------------------------- /analyze


def test_analyze_returns_grounded_answer(client, monkeypatch):
    evidence = [
        ToolCall(
            tool_name="get_predicted_genes",
            args={"perturbation_id": "KLF1"},
            result=[{"gene": "HBB", "delta": 1.2}],
        )
    ]
    monkeypatch.setattr(api_main, "make_llm_fn", lambda **kwargs: (lambda prompt: "ignored"))
    monkeypatch.setattr(
        api_main,
        "run_agent",
        lambda question, llm_fn, perturbation_id=None, tools=None: AgentResponse(
            answer="KLF1 induces HBB [source: get_predicted_genes → HBB].",
            evidence=evidence,
            perturbation_id=perturbation_id,
        ),
    )
    response = client.post(
        "/analyze", json={"question": "What does KLF1 induce?", "perturbation_id": "KLF1"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["perturbation_id"] == "KLF1"
    assert body["citation_valid"] is True
    assert body["grounding_rate"] == 1.0
    assert body["evidence"][0]["tool"] == "get_predicted_genes"
    assert body["citations"] == ["get_predicted_genes → HBB"]


def test_analyze_400_when_provider_unconfigured(client, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("OPENAI_API_KEY is not set")

    monkeypatch.setattr(api_main, "make_llm_fn", boom)
    response = client.post("/analyze", json={"question": "What does KLF1 do?"})
    assert response.status_code == 400
    assert "OPENAI_API_KEY" in response.json()["detail"]


@pytest.mark.parametrize("payload", [
    {"question": "ab"},
    {"question": "What does KLF1 do?", "provider": "not-a-provider"},
    {"question": "What does KLF1 do?", "bogus": True},
])
def test_analyze_validation_errors(client, payload):
    assert client.post("/analyze", json=payload).status_code == 422
