"""Tests for the agent orchestration loop and citation-grounding evaluation.

The orchestration loop is exercised with a mock ``llm_fn`` (a scripted queue
of responses) and lightweight fake tools, so no model, index, or network is
required.
"""

from __future__ import annotations

import pytest

from perturbgpt.agent.orchestrator import AgentResponse, ToolCall, run
from perturbgpt.eval import agent_eval


def _scripted_llm(*responses: str):
    """Return an ``llm_fn`` that pops the next scripted response each call."""
    queue = list(responses)

    def llm_fn(prompt: str) -> str:
        return queue.pop(0) if queue else ""

    return llm_fn


# ------------------------------------------------------- orchestration loop


def test_run_returns_final_answer_when_no_tool_calls():
    llm_fn = _scripted_llm("KLF1 is an erythroid master regulator.")
    resp = run("What does KLF1 do?", llm_fn)
    assert isinstance(resp, AgentResponse)
    assert resp.answer == "KLF1 is an erythroid master regulator."
    assert resp.evidence == []
    assert resp.perturbation_id is None


def test_run_executes_tools_and_collects_evidence():
    tools = {
        "get_pathways": lambda gene_symbols: [{"pathway": "HALLMARK_ERYTHROID"}]
    }
    llm_fn = _scripted_llm(
        '<tool_call name="get_pathways">{"gene_symbols": ["KLF1"]}</tool_call>',
        "KLF1 activates erythroid pathways [source: get_pathways → HALLMARK_ERYTHROID]",
    )
    resp = run("Which pathways does KLF1 activate?", llm_fn, tools=tools)

    assert resp.answer.startswith("KLF1 activates erythroid pathways")
    assert len(resp.evidence) == 1
    assert resp.evidence[0].tool_name == "get_pathways"
    assert resp.evidence[0].result == [{"pathway": "HALLMARK_ERYTHROID"}]


def test_run_supports_multiple_tool_rounds():
    tools = {
        "get_predicted_genes": (
            lambda perturbation_id, top_k=20: [{"gene": "HBB", "delta": 1.0}]
        ),
        "get_pathways": lambda gene_symbols: [{"pathway": "HALLMARK_ERYTHROID"}],
    }
    llm_fn = _scripted_llm(
        '<tool_call name="get_predicted_genes">{"perturbation_id": "KLF1", "top_k": 5}</tool_call>',
        '<tool_call name="get_pathways">{"gene_symbols": ["HBB"]}</tool_call>',
        "KLF1 induces HBB [source: get_predicted_genes → HBB] and erythroid "
        "pathways [source: get_pathways → HALLMARK_ERYTHROID]",
    )
    resp = run("What does KLF1 induce?", llm_fn, tools=tools)

    assert [e.tool_name for e in resp.evidence] == [
        "get_predicted_genes",
        "get_pathways",
    ]
    assert resp.evidence[0].args == {"perturbation_id": "KLF1", "top_k": 5}


def test_run_records_unknown_tool_as_error():
    llm_fn = _scripted_llm(
        '<tool_call name="does_not_exist">{"x": 1}</tool_call>',
        "No supporting evidence found in the current analysis.",
    )
    resp = run("q?", llm_fn)
    assert len(resp.evidence) == 1
    assert resp.evidence[0].result.startswith("Error: unknown tool")


def test_run_stops_at_iteration_limit():
    tools = {"noop": lambda **kwargs: []}
    llm_fn = _scripted_llm(
        *['<tool_call name="noop">{"a": 1}</tool_call>'] * 100
    )
    resp = run("q?", llm_fn, tools=tools, max_iterations=3)
    assert len(resp.evidence) == 3
    assert "iteration limit" in resp.answer


def test_run_records_perturbation_id():
    llm_fn = _scripted_llm("done")
    resp = run("q?", llm_fn, perturbation_id="KLF1")
    assert resp.perturbation_id == "KLF1"


# ------------------------------------------------------ citation grounding


def test_citation_validity_all_grounded():
    evidence = [
        ToolCall(
            tool_name="get_pathways",
            args={"gene_symbols": ["KLF1"]},
            result=[{"pathway": "HALLMARK_ERYTHROID"}],
        )
    ]
    answer = "KLF1 activates erythroid pathways [source: get_pathways → HALLMARK_ERYTHROID]"
    assert agent_eval.check_citation_validity(answer, evidence) == []


def test_citation_validity_detects_unbacked_tool():
    evidence = [
        ToolCall(
            tool_name="get_pathways",
            args={},
            result=[{"pathway": "HALLMARK_ERYTHROID"}],
        )
    ]
    answer = "KLF1 is cited by literature [source: retrieve_literature → PMID:12345678]"
    invalid = agent_eval.check_citation_validity(answer, evidence)
    assert invalid == [("retrieve_literature", "PMID:12345678")]


def test_citation_validity_empty_answer():
    assert agent_eval.check_citation_validity("", []) == []


def test_grounding_rate_counts_grounded_sentences():
    evidence = [
        ToolCall(tool_name="get_pathways", args={}, result=[{"pathway": "X"}]),
        ToolCall(
            tool_name="get_predicted_genes", args={}, result=[{"gene": "HBB"}]
        ),
    ]
    answer = (
        "KLF1 is an erythroid regulator [source: get_pathways → HALLMARK_ERYTHROID]. "
        "It induces HBB [source: get_predicted_genes → HBB]. "
        "This sentence has no citation."
    )
    assert agent_eval.grounding_rate(answer, evidence) == pytest.approx(2 / 3)


def test_grounding_rate_excludes_unbacked_citations():
    evidence = [
        ToolCall(tool_name="get_pathways", args={}, result=[{"pathway": "X"}])
    ]
    answer = (
        "KLF1 activates X [source: get_pathways → X]. "
        "Prior work agrees [source: retrieve_literature → PMID:1]."
    )
    assert agent_eval.grounding_rate(answer, evidence) == pytest.approx(0.5)


def test_grounding_rate_empty_answer_is_one():
    assert agent_eval.grounding_rate("", []) == 1.0


def test_no_evidence_fallback_used_when_no_results():
    evidence = [ToolCall(tool_name="get_pathways", args={}, result=[])]
    answer = "No supporting evidence found in the current analysis."
    assert agent_eval.check_no_evidence_fallback(answer, evidence) is True


def test_no_evidence_fallback_not_used_when_results_present():
    evidence = [
        ToolCall(tool_name="get_pathways", args={}, result=[{"pathway": "X"}])
    ]
    answer = "KLF1 activates X [source: get_pathways → X]"
    assert agent_eval.check_no_evidence_fallback(answer, evidence) is True
