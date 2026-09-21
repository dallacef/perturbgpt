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

from perturbgpt.agent.prompts import SYSTEM_PROMPT, NO_EVIDENCE_PHRASE
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

# todo finish orchestrator loop, including LLM calls, tool execution, and final answer assembly