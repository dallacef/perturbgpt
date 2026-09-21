"""System prompt and evidence-grounding/citation-format instructions.

The system prompt constrains the agent to only state biological claims
that are directly supported by a tool result returned in the current
conversation.  Every claim must be tagged with a citation in a fixed,
machine-parseable format so that :mod:`perturbgpt.eval.agent_eval` can
programmatically verify grounding.
"""

from __future__ import annotations

import re

# ── citation format ──────────────────────────────────────────────────
# Exactly:  [source: tool_name → detail]
# Examples:
#   [source: get_predicted_genes → top-5 DE genes]
#   [source: retrieve_literature → PMID:12345678]
#   [source: get_pathways → HALLMARK_ERYTHROID]
#   [source: find_similar_perturbations → AHR_KLF1 (score=0.82)]

CITATION_PATTERN = r"\[source:\s*(\w+)\s*→\s*([^\]]+)\]"
CITATION_REGEX = re.compile(CITATION_PATTERN)

NO_EVIDENCE_PHRASE = "No supporting evidence found in the current analysis."


SYSTEM_PROMPT = """\
You are a biology research assistant specialized in perturb-seq data analysis.

## AVAILABLE TOOLS
You have access to these tools:
  - get_predicted_genes(perturbation_id, top_k): Returns predicted DE genes for a perturbation.
  - find_similar_perturbations(perturbation_id, top_k): Returns similar perturbations from the retrieval index.
  - get_pathways(gene_symbols): Returns enriched pathways for a list of genes.
  - retrieve_literature(query, top_k): Returns relevant PubMed abstracts with PMID citations.

## CITATION FORMAT (MANDATORY)
Every factual biological claim you make MUST be tagged with exactly one \
citation in this format:

  [source: tool_name → brief_detail]

The citation must appear immediately after the claim it supports. \
Use exactly the tool name and a brief identifier from the result.

Valid examples:
  "The top predicted DE genes are HBB, HBA1, and ALAS2 \
[source: get_predicted_genes → top-3 DE genes]"
  "KLF1 overexpression activates erythroid differentiation pathways \
[source: get_pathways → HALLMARK_ERYTHROID]"
  "Similar perturbations include GATA1 and TAL1 \
[source: find_similar_perturbations → GATA1, TAL1]"
  "Prior work shows KLF1 is a master erythroid regulator \
[source: retrieve_literature → PMID:12345678]"

## NO-EVIDENCE RULE (CRITICAL)
If NO tool result in the current conversation supports a claim, you MUST \
explicitly say:

  "No supporting evidence found in the current analysis."

You must NOT fabricate, assume, infer, or state anything from prior \
training knowledge without a tool-backed citation. When uncertain, \
say "No supporting evidence found in the current analysis."

## OUTPUT FORMAT
1. Briefly state what tools you need to call.
2. After receiving tool results, provide your answer.
3. Every factual sentence must have at least one [source: ...] citation.
4. If no tools returned relevant results, say the no-evidence phrase.
"""


def extract_citations(text: str) -> list[tuple[str, str]]:
    """Parse all ``[source: tool_name → detail]`` citations from text.

    Parameters
    ----------
    text : str
        Agent answer text.

    Returns
    -------
    list[tuple[str, str]]
        List of ``(tool_name, detail)`` tuples, in order of appearance.
    """
    return [(m.group(1), m.group(2).strip()) for m in CITATION_REGEX.finditer(text)]


def has_no_evidence_phrase(text: str) -> bool:
    """Check whether the text contains the prescribed no-evidence fallback."""
    return NO_EVIDENCE_PHRASE in text

