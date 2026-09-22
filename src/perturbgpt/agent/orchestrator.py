"""Tool-calling control loop for the perturbation analysis agent.

Takes a user question plus a perturbation ID, decides which tools to
call, executes them, and returns a final answer with structured
evidence.  The LLM is abstracted as a callable ``llm_fn(prompt) -> str``
so the orchestrator can be tested with a mock.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from perturbgpt.agent.prompts import SYSTEM_PROMPT
from perturbgpt.agent.tools import (
    get_predicted_genes,
    find_similar_perturbations,
    get_pathways,
    retrieve_literature,
)


@dataclass
class ToolCall:
    """Record of a single tool invocation within an agent run."""

    tool_name: str
    args: dict
    result: Any  # list[dict], dict, or str


@dataclass
class AgentResponse:
    """Final output of an agent run."""

    answer: str
    evidence: list[ToolCall] = field(default_factory=list)
    perturbation_id: Optional[str] = None


# Default tool registry
TOOLS: dict[str, Callable] = {
    "get_predicted_genes": get_predicted_genes,
    "find_similar_perturbations": find_similar_perturbations,
    "get_pathways": get_pathways,
    "retrieve_literature": retrieve_literature,
}


# ── tool-call parsing ─────────────────────────────────────────────────

TOOL_CALL_PATTERN = re.compile(
    r'<tool_call\s+name="(\w+)">\s*(\{.*?\})\s*',
    re.DOTALL,
)


def parse_tool_calls(response: str) -> list[tuple[str, dict]]:
    """Parse tool-call requests from an LLM response.

    Returns
    -------
    list[tuple[str, dict]]
        List of (tool_name, args_dict) tuples.
    """
    calls = []
    for m in TOOL_CALL_PATTERN.finditer(response):
        name = m.group(1)
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            continue
        calls.append((name, args))
    return calls


def execute_tool(name: str, args: dict, tools: dict[str, Callable]) -> ToolCall:
    """Execute a single tool call and return a ToolCall record."""
    fn = tools.get(name)
    if fn is None:
        return ToolCall(tool_name=name, args=args, result=f"Error: unknown tool {name!r}")
    try:
        result = fn(**args)
    except Exception as exc:
        result = f"Error: {exc}"
    return ToolCall(tool_name=name, args=args, result=result)

MAX_ITERATIONS = 8


def build_prompt(question: str, perturbation_id: Optional[str] = None) -> str:
    """Assemble the full prompt: system prompt + optional perturbation context + question."""
    parts = [SYSTEM_PROMPT.strip()]
    if perturbation_id:
        parts.append(f"## CURRENT PERTURBATION\n{perturbation_id}")
    parts.append(f"## USER QUESTION\n{question}")
    return "\n\n".join(parts)


def format_tool_result(record: ToolCall) -> str:
    """Serialize a :class:`ToolCall` result for inclusion in the next prompt."""
    try:
        result_str = json.dumps(record.result, default=str)
    except (TypeError, ValueError):
        result_str = str(record.result)
    args_str = json.dumps(record.args, default=str)
    return f"[TOOL RESULT] {record.tool_name}({args_str})\n{result_str}"


def run(
    question: str,
    llm_fn: Callable[[str], str],
    perturbation_id: Optional[str] = None,
    tools: Optional[dict[str, Callable]] = None,
    max_iterations: int = MAX_ITERATIONS,
) -> AgentResponse:
    """Run the tool-calling control loop for a single question.

    Repeats: call ``llm_fn(prompt)``, parse ``<tool_call>`` requests, execute
    them, and append their results to the prompt — until the LLM returns a
    response with no tool calls (the final answer) or the iteration limit is
    reached.

    Parameters
    ----------
    question : str
        The user's question.
    llm_fn : callable
        ``llm_fn(prompt: str) -> str`` returning the LLM's next response.
    perturbation_id : str, optional
        Perturbation context to include in the prompt and record in the result.
    tools : dict[str, callable], optional
        Tool registry; defaults to :data:`TOOLS`.
    max_iterations : int
        Maximum number of LLM/tool rounds before giving up.

    Returns
    -------
    AgentResponse
        Final answer, accumulated tool-call evidence, and perturbation id.
    """
    if tools is None:
        tools = TOOLS

    prompt = build_prompt(question, perturbation_id)
    evidence: list[ToolCall] = []

    for _ in range(max_iterations):
        response = llm_fn(prompt)
        calls = parse_tool_calls(response)
        if not calls:
            return AgentResponse(
                answer=response,
                evidence=evidence,
                perturbation_id=perturbation_id,
            )

        for name, args in calls:
            record = execute_tool(name, args, tools)
            evidence.append(record)
            prompt += "\n\n" + format_tool_result(record)

    return AgentResponse(
        answer="No final answer produced within the iteration limit.",
        evidence=evidence,
        perturbation_id=perturbation_id,
    )