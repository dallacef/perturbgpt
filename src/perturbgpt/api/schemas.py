"""Pydantic request/response models for every API endpoint.

Kept deliberately thin: no model logic lives here, only the wire contract.
All request models reject unknown fields (``extra="forbid"``) so typos in a
client payload fail loudly with HTTP 422 instead of being silently ignored.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# ------------------------------------------------------------------ health


class HealthResponse(BaseModel):
    """Liveness probe plus which optional artefacts are present on disk."""

    status: Literal["ok"] = "ok"
    version: str
    artifacts: dict[str, bool] = Field(
        ..., description="Artefact name -> whether it is available."
    )


# ----------------------------------------------------------------- predict


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    perturbation_id: str = Field(
        ..., min_length=1, max_length=128, description="e.g. 'KLF1' or 'AHR_KLF1'."
    )
    top_k: int = Field(20, ge=1, le=100, description="Number of top DE genes to return.")


class PredictedGene(BaseModel):
    gene: str
    delta: float


class PredictResponse(BaseModel):
    perturbation_id: str
    genes: list[PredictedGene]
    checkpoint: str


# ----------------------------------------------------- similar perturbations


class SimilarPerturbationsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    perturbation_id: str = Field(..., min_length=1, max_length=128)
    top_k: int = Field(10, ge=1, le=50)
    index_type: Literal["perturbation", "response"] = "perturbation"


class SimilarPerturbation(BaseModel):
    pert_id: str
    score: float
    gene_symbols: list[str] = []
    cell_count: int = 0


class SimilarPerturbationsResponse(BaseModel):
    query: str
    index_type: str
    results: list[SimilarPerturbation]


# ----------------------------------------------------------------- analyze


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=3, max_length=2000)
    perturbation_id: Optional[str] = Field(None, max_length=128)
    provider: Optional[Literal["deterministic", "openai", "anthropic", "ollama"]] = Field(
        None, description="LLM backend override; defaults to the server configuration."
    )


class ToolEvidence(BaseModel):
    """One tool call made by the agent, with its raw result."""

    tool: str
    args: dict[str, Any] = {}
    result: Any = None


class AnalyzeResponse(BaseModel):
    answer: str
    perturbation_id: Optional[str] = None
    citations: list[str] = []
    evidence: list[ToolEvidence] = []
    citation_valid: bool
    grounding_rate: float
    no_evidence_fallback: bool
