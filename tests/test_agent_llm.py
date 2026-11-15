"""Tests for the pluggable LLM backends and the deterministic planner.

No network and no API keys: the deterministic planner drives the real
orchestrator loop against lightweight fake tools.
"""

from __future__ import annotations

import pytest

from perturbgpt.agent.llm import DeterministicPlanner, make_llm_fn
from perturbgpt.agent.orchestrator import run
from perturbgpt.agent.prompts import NO_EVIDENCE_PHRASE
from perturbgpt.eval import agent_eval


def _fake_tools():
    def get_predicted_genes(perturbation_id, top_k=20):
        return [{"gene": "HBB", "delta": 1.0}, {"gene": "HBA1", "delta": 0.8}][:top_k]

    def get_pathways(gene_symbols):
        return [
            {"pathway": "HALLMARK_ERYTHROID", "p_adj": 0.01, "genes": list(gene_symbols)}
        ]

    def find_similar_perturbations(perturbation_id, top_k=10, index_type="perturbation"):
        return [
            {"pert_id": "GATA1", "score": 0.9, "gene_symbols": ["GATA1"], "cell_count": 10}
        ][:top_k]

    return {
        "get_predicted_genes": get_predicted_genes,
        "get_pathways": get_pathways,
        "find_similar_perturbations": find_similar_perturbations,
    }


# --------------------------------------------------------------- factory


def test_deterministic_is_the_default(monkeypatch):
    monkeypatch.delenv("PERTURBGPT_LLM_PROVIDER", raising=False)
    assert isinstance(make_llm_fn(), DeterministicPlanner)


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        make_llm_fn(provider="not-a-provider")


def test_openai_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        make_llm_fn(provider="openai")


def test_anthropic_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        make_llm_fn(provider="anthropic")


@pytest.mark.parametrize("provider,env", [
    ("openai", "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
])
def test_http_backends_build_callables(monkeypatch, provider, env):
    monkeypatch.setenv(env, "dummy-key")
    llm_fn = make_llm_fn(provider=provider)
    assert callable(llm_fn)


def test_ollama_backend_builds_without_key():
    assert callable(make_llm_fn(provider="ollama"))


# --------------------------------------------------------------- planner


def test_planner_emits_tool_call_on_first_turn():
    planner = DeterministicPlanner()
    out = planner("## CURRENT PERTURBATION\nKLF1\n\n## USER QUESTION\nWhat happens?")
    assert out.startswith('<tool_call name="get_predicted_genes">')
    assert '"perturbation_id": "KLF1"' in out


def test_planner_end_to_end_is_fully_grounded():
    response = run(
        "What are the downstream effects of KLF1 perturbation?",
        make_llm_fn(),
        perturbation_id="KLF1",
        tools=_fake_tools(),
    )
    assert [c.tool_name for c in response.evidence] == [
        "get_predicted_genes", "get_pathways", "find_similar_perturbations",
    ]
    assert agent_eval.grounding_rate(response.answer, response.evidence) == 1.0
    assert agent_eval.check_citation_validity(response.answer, response.evidence) == []
    assert agent_eval.check_no_evidence_fallback(response.answer, response.evidence)
    assert "HBB" in response.answer
    assert "HALLMARK_ERYTHROID" in response.answer
    assert "GATA1" in response.answer


def test_planner_falls_back_to_no_evidence_phrase():
    tools = {
        "get_predicted_genes": lambda perturbation_id, top_k=20: [],
        "get_pathways": lambda gene_symbols: [],
        "retrieve_literature": lambda query, top_k=5: [],
        "find_similar_perturbations": lambda perturbation_id, top_k=10: [],
    }
    response = run(
        "What are the downstream effects of ZZZ9 perturbation?",
        make_llm_fn(),
        perturbation_id="ZZZ9",
        tools=tools,
    )
    assert response.answer.strip() == NO_EVIDENCE_PHRASE
    assert agent_eval.check_no_evidence_fallback(response.answer, response.evidence)


def test_planner_without_perturbation_uses_literature():
    calls: list[str] = []

    def retrieve_literature(query, top_k=5):
        calls.append(query)
        return [{"pmid": "123", "title": "t", "abstract": "a", "score": 0.5,
                 "journal": "j", "year": "2020"}]

    response = run(
        "What is known about erythroid maturation?",
        make_llm_fn(),
        tools={"retrieve_literature": retrieve_literature},
    )
    assert calls and "erythroid" in calls[0]
    assert "PMID:123" in response.answer
    assert agent_eval.grounding_rate(response.answer, response.evidence) == 1.0
