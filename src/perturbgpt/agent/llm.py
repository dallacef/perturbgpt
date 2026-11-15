"""Pluggable LLM backends for the tool-calling orchestrator.

The orchestrator (:func:`perturbgpt.agent.orchestrator.run`) only needs a
callable ``llm_fn(prompt: str) -> str``.  This module builds such callables
for real backends as well as a deterministic, fully offline planner.

Providers
---------
``deterministic`` (default)
    No network, no API key.  A scripted planner that issues the canonical
tool calls (predicted genes -> pathways -> similar perturbations, or a
literature search when no perturbation is given) and then composes an
answer whose every sentence is backed by a tool result.  Used for offline
integration tests and CI, and as the fallback when no backend is configured.
``openai``
    Any OpenAI-compatible ``/chat/completions`` endpoint.  Configure with
    ``OPENAI_API_KEY`` and optionally ``OPENAI_BASE_URL`` (e.g. a local
    vLLM/LM Studio server).
``anthropic``
    Anthropic Messages API.  Configure with ``ANTHROPIC_API_KEY``.
``ollama``
    Local Ollama daemon (``OLLAMA_HOST``, default ``http://localhost:11434``).

Environment
-----------
``PERTURBGPT_LLM_PROVIDER``  provider name when none is passed explicitly
``PERTURBGPT_LLM_MODEL``     model name when none is passed explicitly

Only ``requests`` (a core dependency) is used — no vendor SDKs.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Callable, Optional

import requests

from perturbgpt.agent.prompts import NO_EVIDENCE_PHRASE

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "ollama": "llama3.1",
}

PROVIDERS = ("deterministic", "openai", "anthropic", "ollama")

#: Prompt section the orchestrator emits for perturbation context.
_PERT_SECTION = re.compile(
    r"## CURRENT PERTURBATION\s*\n([^\n]+)", re.IGNORECASE
)
_QUESTION_SECTION = re.compile(
    r"## USER QUESTION\s*\n(.+?)(?:\n\n\[TOOL RESULT\]|\Z)", re.DOTALL
)
_TOOL_RESULT = re.compile(
    r"\[TOOL RESULT\] (\w+)\((.*?)\)\n(.*?)(?=\n\n\[TOOL RESULT\]|\Z)",
    re.DOTALL,
)


# --------------------------------------------------------------- deterministic


def _tool_calls_in(prompt: str) -> list[str]:
    """Names of tools that have already returned a result in *prompt*."""
    return [m.group(1) for m in _TOOL_RESULT.finditer(prompt)]


def _results_in(prompt: str) -> dict[str, object]:
    """Parse ``[TOOL RESULT]`` blocks into ``{tool: parsed_result}``."""
    results: dict[str, object] = {}
    for match in _TOOL_RESULT.finditer(prompt):
        name, _args, payload = match.groups()
        payload = payload.strip()
        try:
            results[name] = json.loads(payload)
        except json.JSONDecodeError:
            results[name] = payload
    return results


def _perturbation_from(prompt: str) -> Optional[str]:
    match = _PERT_SECTION.search(prompt)
    return match.group(1).strip() if match else None


def _question_from(prompt: str) -> str:
    match = _QUESTION_SECTION.search(prompt)
    return match.group(1).strip() if match else ""


def _emit_tool(name: str, **args) -> str:
    return f'<tool_call name="{name}">{json.dumps(args)}</tool_call>'


class DeterministicPlanner:
    """Offline, key-free planner that exercises the real tool pipeline.

    Emits tool calls in a fixed order and then answers using **only** the
    parsed tool results, tagging each sentence with the citation format the
    system prompt requires.  When no tool produced usable data it returns
    the prescribed no-evidence phrase verbatim (and nothing else).
    """

    def __init__(self, max_genes: int = 5, max_similar: int = 3) -> None:
        self.max_genes = max_genes
        self.max_similar = max_similar

    def __call__(self, prompt: str) -> str:
        called = _tool_calls_in(prompt)
        results = _results_in(prompt)
        pert = _perturbation_from(prompt)

        # No perturbation context: answer from the literature instead.
        if pert is None and not called:
            return _emit_tool(
                "retrieve_literature",
                query=self._literature_query(None, prompt),
                top_k=3,
            )

        if pert and "get_predicted_genes" not in called:
            return _emit_tool("get_predicted_genes", perturbation_id=pert, top_k=self.max_genes)

        if "get_predicted_genes" in called and "get_pathways" not in called:
            genes = self._gene_symbols(results.get("get_predicted_genes"))
            if genes:
                return _emit_tool("get_pathways", gene_symbols=genes)
            # No genes predicted -> skip enrichment and try literature instead.
            if "retrieve_literature" not in called:
                return _emit_tool(
                    "retrieve_literature", query=self._literature_query(pert, prompt), top_k=3
                )

        if (
            pert
            and "find_similar_perturbations" not in called
            and self._has_usable(results.get("get_pathways"))
        ):
            return _emit_tool(
                "find_similar_perturbations", perturbation_id=pert, top_k=self.max_similar
            )

        return self._compose_answer(results, pert)

    # ----- helpers

    @staticmethod
    def _has_usable(result: object) -> bool:
        return bool(result)

    @staticmethod
    def _gene_symbols(result: object) -> list[str]:
        if not isinstance(result, list):
            return []
        out = []
        for row in result:
            if isinstance(row, dict) and row.get("gene"):
                out.append(str(row["gene"]))
        return out

    def _literature_query(self, pert: Optional[str], prompt: str) -> str:
        if pert:
            return f"{pert} perturbation transcriptome"
        return _question_from(prompt) or "perturbation transcriptome"

    def _compose_answer(self, results: dict[str, object], pert: Optional[str]) -> str:
        sentences: list[str] = []

        genes = self._gene_symbols(results.get("get_predicted_genes"))
        if genes:
            top = ", ".join(genes[: self.max_genes])
            sentences.append(
                f"The top predicted DE genes are {top} "
                f"[source: get_predicted_genes → top-{min(len(genes), self.max_genes)} DE genes]"
            )

        pathways = results.get("get_pathways")
        if isinstance(pathways, list) and pathways:
            names = [str(p.get("pathway")) for p in pathways if isinstance(p, dict) and p.get("pathway")]
            if names:
                sentences.append(
                    f"Enriched pathways include {', '.join(names[:3])} "
                    f"[source: get_pathways → {names[0]}]"
                )

        similar = results.get("find_similar_perturbations")
        if isinstance(similar, list) and similar:
            ids = [str(r.get("pert_id")) for r in similar if isinstance(r, dict) and r.get("pert_id")]
            if ids:
                sentences.append(
                    f"Similar perturbations include {', '.join(ids[: self.max_similar])} "
                    f"[source: find_similar_perturbations → {', '.join(ids[: self.max_similar])}]"
                )

        literature = results.get("retrieve_literature")
        if isinstance(literature, list) and literature:
            pmids = [str(a.get("pmid")) for a in literature if isinstance(a, dict) and a.get("pmid")]
            if pmids:
                sentences.append(
                    f"Supporting literature includes PMID:{', PMID:'.join(pmids[:3])} "
                    f"[source: retrieve_literature → PMID:{pmids[0]}]"
                )

        if not sentences:
            return NO_EVIDENCE_PHRASE

        subject = f" for {pert}" if pert else ""
        return f"Analysis{subject}: " + " ".join(sentences)


# --------------------------------------------------------------- HTTP backends


def _openai_llm_fn(
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    timeout: float,
) -> Callable[[str], str]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def llm_fn(prompt: str) -> str:
        resp = requests.post(
            url,
            headers=headers,
            json={
                "model": model,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"] or ""

    return llm_fn


def _anthropic_llm_fn(
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    timeout: float,
) -> Callable[[str], str]:
    url = base_url.rstrip("/") + "/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    def llm_fn(prompt: str) -> str:
        resp = requests.post(
            url,
            headers=headers,
            json={
                "model": model,
                "max_tokens": 1024,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        blocks = resp.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    return llm_fn


def _ollama_llm_fn(
    model: str,
    host: str,
    temperature: float,
    timeout: float,
) -> Callable[[str], str]:
    url = host.rstrip("/") + "/api/chat"

    def llm_fn(prompt: str) -> str:
        resp = requests.post(
            url,
            json={
                "model": model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": temperature},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "") or ""

    return llm_fn


# --------------------------------------------------------------- factory


def make_llm_fn(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
    timeout: float = 60.0,
) -> Callable[[str], str]:
    """Build an ``llm_fn`` for the orchestrator.

    Parameters
    ----------
    provider : str, optional
        ``"deterministic"``, ``"openai"``, ``"anthropic"`` or ``"ollama"``.
        Defaults to ``PERTURBGPT_LLM_PROVIDER`` and then
        ``"deterministic"`` (works offline with no API key).
    model : str, optional
        Model name.  Defaults to ``PERTURBGPT_LLM_MODEL`` or a provider
        default.
    temperature : float
        Sampling temperature (0.0 for reproducible grounding behaviour).
    base_url : str, optional
        Override the provider endpoint (OpenAI-compatible servers, self-hosted
        Anthropic proxies, or an Ollama host).
    timeout : float
        Per-request timeout in seconds.

    Returns
    -------
    callable
        ``llm_fn(prompt: str) -> str``.

    Raises
    ------
    ValueError
        If *provider* is unknown.
    RuntimeError
        If the chosen provider's credentials/environment are missing.
    """
    provider = (provider or os.environ.get("PERTURBGPT_LLM_PROVIDER") or "deterministic").lower()
    model = model or os.environ.get("PERTURBGPT_LLM_MODEL")

    if provider == "deterministic":
        return DeterministicPlanner()

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI backend selected but OPENAI_API_KEY is not set. "
                "Export it or use --llm-provider deterministic."
            )
        return _openai_llm_fn(
            model=model or DEFAULT_MODELS["openai"],
            api_key=api_key,
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL,
            temperature=temperature,
            timeout=timeout,
        )

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Anthropic backend selected but ANTHROPIC_API_KEY is not set."
            )
        return _anthropic_llm_fn(
            model=model or DEFAULT_MODELS["anthropic"],
            api_key=api_key,
            base_url=base_url or os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_ANTHROPIC_BASE_URL,
            temperature=temperature,
            timeout=timeout,
        )

    if provider == "ollama":
        return _ollama_llm_fn(
            model=model or DEFAULT_MODELS["ollama"],
            host=base_url or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST,
            temperature=temperature,
            timeout=timeout,
        )

    raise ValueError(
        f"Unknown LLM provider {provider!r}; expected one of {', '.join(PROVIDERS)}."
    )
