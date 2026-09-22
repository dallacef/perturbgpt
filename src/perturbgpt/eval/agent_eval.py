"""Automated citation-validity checking and evidence-grounding-rate computation.

Works over :class:`~perturbgpt.agent.orchestrator.AgentResponse` objects (or
their constituent ``answer`` and ``evidence`` fields) to verify that every
``[source: tool_name → detail]`` citation emitted by the agent corresponds to
an actual tool-call result from the same conversation, and to measure what
fraction of the agent's factual sentences carry a grounded citation.
"""

from __future__ import annotations

import re
from typing import Sequence

from perturbgpt.agent.orchestrator import ToolCall
from perturbgpt.agent.prompts import (
    CITATION_REGEX,
    extract_citations,
    has_no_evidence_phrase,
)

#: Sentence boundary: terminal punctuation followed by whitespace. Kept
#: deliberately simple; abbreviations such as ``e.g.`` are not handled.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences on terminal punctuation + whitespace."""
    stripped = text.strip()
    if not stripped:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(stripped) if s.strip()]


def check_citation_validity(
    answer: str,
    evidence: Sequence[ToolCall],
) -> list[tuple[str, str]]:
    """Return citations in *answer* that have no matching tool call.

    A citation ``[source: tool_name → detail]`` is *valid* iff ``tool_name``
    matches a tool that was actually invoked in this conversation (i.e.
    appears as a :attr:`ToolCall.tool_name` in *evidence*).

    Returns the list of invalid ``(tool_name, detail)`` tuples; an empty list
    means every citation is grounded in the evidence.
    """
    citations = extract_citations(answer)
    called = {t.tool_name for t in evidence}
    return [c for c in citations if c[0] not in called]


def grounding_rate(
    answer: str,
    evidence: Sequence[ToolCall],
) -> float:
    """Fraction of the answer's sentences carrying a grounded citation.

    A sentence is *grounded* when it contains at least one citation and every
    citation in it names a tool that was actually invoked (present in
    *evidence*). Returns 1.0 for an empty answer (vacuously grounded).
    """
    sentences = split_sentences(answer)
    if not sentences:
        return 1.0

    called = {t.tool_name for t in evidence}
    grounded = 0
    for sentence in sentences:
        citations = extract_citations(sentence)
        if citations and all(name in called for name, _ in citations):
            grounded += 1
    return grounded / len(sentences)


def check_no_evidence_fallback(
    answer: str,
    evidence: Sequence[ToolCall],
) -> bool:
    """Whether the answer uses the no-evidence fallback correctly.

    When *no* tool call produced a non-empty result, the answer must contain
    the prescribed no-evidence phrase; when at least one tool returned
    evidence, the answer must *not* use the fallback phrase.
    """
    has_results = any(bool(t.result) for t in evidence)
    if has_results:
        return not has_no_evidence_phrase(answer)
    return has_no_evidence_phrase(answer)
