"""FastAPI application exposing the model, retrieval and agent as a service.

Endpoints
---------
``GET  /health``                liveness + which artefacts are available
``POST /predict``               top-k predicted DE genes for a perturbation
``POST /analyze``               run the tool-calling agent on a question
``POST /similar-perturbations``  FAISS neighbour lookup (perturbation or response)

This module is a **thin** layer: all modelling, retrieval and agent logic
lives in :mod:`perturbgpt.agent.tools`, :mod:`perturbgpt.retrieval` and
:mod:`perturbgpt.models`.  Artefacts are loaded lazily and cached by the tool
layer, so the first request is slower than the rest but no model is loaded at
import time.

Run locally::

    uvicorn perturbgpt.api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from perturbgpt import __version__
from perturbgpt.agent import tools
from perturbgpt.agent.llm import make_llm_fn
from perturbgpt.agent.orchestrator import TOOLS as AGENT_TOOLS
from perturbgpt.agent.orchestrator import run as run_agent
from perturbgpt.agent.prompts import extract_citations
from perturbgpt.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    PredictedGene,
    PredictRequest,
    PredictResponse,
    SimilarPerturbation,
    SimilarPerturbationsRequest,
    SimilarPerturbationsResponse,
    ToolEvidence,
)
from perturbgpt.eval import agent_eval

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PerturbGPT API",
    version=__version__,
    summary="Perturb-seq response prediction, retrieval and tool-calling agent",
    description=(
        "Thin HTTP layer over the perturbgpt package. Artefacts (prediction "
        "checkpoint, FAISS indices, pathway sets, literature index) are "
        "produced by the scripts in `scripts/` and loaded lazily."
    ),
)

PATHWAY_SETS_PATH = tools.PROJECT_ROOT / "data" / "pathways" / "msigdb_hallmark.json"


def artifact_status() -> dict[str, bool]:
    """Report which optional artefacts exist on disk."""
    return {
        "checkpoint": tools.DEFAULT_CHECKPOINT.exists(),
        "gene_embeddings": tools.DEFAULT_GENE_EMBEDDINGS.exists(),
        "perturbation_index": (
            tools.DEFAULT_INDICES_DIR / "perturbation_index.faiss"
        ).exists(),
        "response_index": (
            tools.DEFAULT_INDICES_DIR / "response_index.faiss"
        ).exists(),
        "literature_index": (
            tools.DEFAULT_LITERATURE_INDEX / "literature.faiss"
        ).exists(),
        "pathway_sets": PATHWAY_SETS_PATH.exists(),
    }


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe that also reports artefact availability."""
    return HealthResponse(status="ok", version=__version__, artifacts=artifact_status())


@app.post("/predict", response_model=PredictResponse, tags=["model"])
def predict(request: PredictRequest) -> PredictResponse:
    """Predict the top-k differentially expressed genes for a perturbation.

    Returns 503 when the inference artefacts are missing, so a client can
    distinguish "model not deployed" from "bad request".
    """
    if not tools.DEFAULT_CHECKPOINT.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Prediction checkpoint not found at {tools.DEFAULT_CHECKPOINT}. "
                "Run scripts/train_prediction_head.py to create it."
            ),
        )
    if not tools.DEFAULT_GENE_EMBEDDINGS.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Gene embeddings not found at {tools.DEFAULT_GENE_EMBEDDINGS}. "
                "Run scripts/build_embeddings.py first."
            ),
        )

    genes = tools.get_predicted_genes(request.perturbation_id, top_k=request.top_k)
    if not genes:
        raise HTTPException(
            status_code=503,
            detail="Prediction head could not be loaded; check the server logs.",
        )

    return PredictResponse(
        perturbation_id=request.perturbation_id,
        genes=[PredictedGene(**row) for row in genes],
        checkpoint=str(tools.DEFAULT_CHECKPOINT),
    )


@app.post(
    "/similar-perturbations",
    response_model=SimilarPerturbationsResponse,
    tags=["retrieval"],
)
def similar_perturbations(
    request: SimilarPerturbationsRequest,
) -> SimilarPerturbationsResponse:
    """Return the nearest perturbations in the gene-embedding or response index."""
    try:
        index = tools.load_index(request.index_type)
    except (ImportError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if index is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"The {request.index_type} index is not available. "
                "Run scripts/build_retrieval_index.py first."
            ),
        )
    if request.perturbation_id not in index.ids:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Perturbation {request.perturbation_id!r} is not in the "
                f"{request.index_type} index."
            ),
        )

    rows = tools.find_similar_perturbations(
        request.perturbation_id,
        top_k=request.top_k,
        index_type=request.index_type,
    )
    return SimilarPerturbationsResponse(
        query=request.perturbation_id,
        index_type=request.index_type,
        results=[SimilarPerturbation(**row) for row in rows],
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["agent"])
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run the tool-calling agent and return the answer plus its evidence."""
    try:
        llm_fn = make_llm_fn(provider=request.provider)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response = run_agent(
        request.question,
        llm_fn,
        perturbation_id=request.perturbation_id,
        tools=AGENT_TOOLS,
    )

    invalid_citations = agent_eval.check_citation_validity(
        response.answer, response.evidence
    )
    citations = [f"{tool} → {detail}" for tool, detail in extract_citations(response.answer)]

    return AnalyzeResponse(
        answer=response.answer,
        perturbation_id=response.perturbation_id,
        citations=citations,
        evidence=[
            ToolEvidence(tool=call.tool_name, args=call.args, result=call.result)
            for call in response.evidence
        ],
        citation_valid=not invalid_citations,
        grounding_rate=agent_eval.grounding_rate(response.answer, response.evidence),
        no_evidence_fallback=agent_eval.check_no_evidence_fallback(
            response.answer, response.evidence
        ),
    )


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    import uvicorn

    uvicorn.run("perturbgpt.api.main:app", host="0.0.0.0", port=8000)
