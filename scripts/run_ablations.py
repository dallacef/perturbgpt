#!/usr/bin/env python3
"""Runs the full ablation suite and writes the consolidated comparison table.

The ablation suite compares four progressively more capable variants against
the same perturbation-level split:

1. **baseline**   — PCA-reduced control expression + learned perturbation
                    embedding through an MLP (``scripts/train_baseline.py``).
2. **film_head**  — FiLM-conditioned MLP over frozen scGPT cell/gene
                    embeddings (``scripts/train_prediction_head.py``).
3. **+retrieval** — FiLM head + FAISS perturbation/response retrieval
                    indices (``scripts/build_retrieval_index.py``).
4. **full agent** — FiLM head + retrieval + pathway enrichment + literature
                    search, orchestrated by the tool-calling agent
                    (``perturbgpt.agent``).

Prediction metrics (MSE, MAE, Pearson, Spearman, top-k) are read from
``results/metrics.csv`` for the baseline and FiLM head. Retrieval metrics
(Recall@K, Precision@K, MRR) are read from
``data/indices/retrieval_results.json``. Agent-level metrics (citation
grounding rate, citation validity, no-evidence fallback correctness) are
computed by running the orchestrator against scripted mock-LLM probes, so
no live LLM or model checkpoint is required.

Usage
-----
    # Read existing results and print the consolidated comparison table:
    python scripts/run_ablations.py

    # Re-run every training/evaluation stage first, then compare:
    python scripts/run_ablations.py --run-all

    # Re-run individual stages:
    python scripts/run_ablations.py --run-baseline
    python scripts/run_ablations.py --run-film-head
    python scripts/run_ablations.py --run-retrieval

    # Custom paths and output:
    python scripts/run_ablations.py \
        --results results/metrics.csv \
        --embeddings-dir data/embeddings \
        --output results/ablation_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from perturbgpt.agent.orchestrator import run  # noqa: E402
from perturbgpt.eval import agent_eval  # noqa: E402

#: Model names written to ``results/metrics.csv`` by each training script.
BASELINE_MODEL = "baseline"
FILM_HEAD_MODEL = "film_head"

#: Prediction metric columns in ``results/metrics.csv``.
PREDICTION_METRIC_KEYS = [
    "pearson", "spearman", "mse", "mae",
    "top50_precision", "top50_recall",
]


def run_stage(name: str, cmd: list[str]) -> int:
    """Run a subprocess stage, echoing its command and exit status.

    Parameters
    ----------
    name : str
        Human-readable stage label for logging.
    cmd : list[str]
        Command + arguments to execute.

    Returns
    -------
    int
        Subprocess return code (0 on success).
    """
    print(f"\n{'=' * 72}")
    print(f"  STAGE: {name}")
    print(f"  CMD: {' '.join(cmd)}")
    print("=" * 72)
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"  ✓ {name} completed successfully\n")
    else:
        print(f"  ✗ {name} failed with exit code {result.returncode}\n")
    return result.returncode


def load_prediction_metrics(
    results_path: Path,
) -> dict[str, dict[str, dict[str, float]]]:
    """Read ``results/metrics.csv`` and keep the *last* row per (model, split).

    Returns
    -------
    dict[str, dict[str, dict[str, float]]]
        Mapping model name → split name → metric dict.
    """
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    if not results_path.exists():
        return metrics
    with results_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            model = row["model"]
            split = row["split"]
            metrics.setdefault(model, {})[split] = {
                k: float(row[k]) for k in PREDICTION_METRIC_KEYS
            }
    return metrics


def load_retrieval_metrics(
    json_path: Path,
) -> dict[str, dict[int, dict[str, float]]]:
    """Read ``retrieval_results.json`` produced by the retrieval-index builder.

    Returns
    -------
    dict[str, dict[int, dict[str, float]]]
        Mapping system name → k → {"recall", "precision", "mrr"}.
    """
    if not json_path.exists():
        return {}
    with json_path.open() as fh:
        data = json.load(fh)
    systems: dict[str, dict[int, dict[str, float]]] = {}
    for sys_name, k_results in data.get("systems", {}).items():
        systems[sys_name] = {
            int(k): {m: float(v) for m, v in metrics.items()}
            for k, metrics in k_results.items()
        }
    return systems


# ----------------------------------------------------------- agent eval probes


def _scripted_llm(*responses: str):
    """Return an ``llm_fn`` that pops the next scripted response each call."""
    queue = list(responses)

    def llm_fn(prompt: str) -> str:
        return queue.pop(0) if queue else ""

    return llm_fn


def _make_probe_tools() -> dict:
    """Build a lightweight mock tool registry for the agent-eval probes.

    ``get_pathways`` and ``get_predicted_genes`` return empty lists for the
    sentinel query ``"ZZZ9"`` so the no-evidence fallback path is exercised.
    """

    def get_pathways(gene_symbols: list[str]) -> list[dict]:
        if not gene_symbols or gene_symbols == ["ZZZ9"]:
            return []
        return [
            {"pathway": "HALLMARK_ERYTHROID", "p_adj": 0.01, "genes": gene_symbols}
        ]

    def get_predicted_genes(perturbation_id: str, top_k: int = 20) -> list[dict]:
        if perturbation_id == "ZZZ9":
            return []
        return [{"gene": "HBB", "delta": 1.0}]

    return {
        "get_predicted_genes": get_predicted_genes,
        "get_pathways": get_pathways,
    }


def run_agent_eval_probes() -> dict[str, float]:
    """Run scripted orchestrator probes and aggregate agent-eval metrics.

    Uses mock tools and a mock LLM so no live model, index, or network is
    required. Probes exercise three behaviours: grounded answers, unbacked
    citations, and the no-evidence fallback.

    Returns
    -------
    dict[str, float]
        Keys: ``grounding_rate``, ``citation_validity_rate``,
        ``no_evidence_fallback_rate``.
    """
    tools = _make_probe_tools()

    probes: list[tuple[str, list[str]]] = [
        (
            "grounded",
            [
                '<tool_call name="get_predicted_genes">{"perturbation_id": "KLF1", "top_k": 5}</tool_call>',
                "KLF1 induces HBB [source: get_predicted_genes → HBB].",
            ],
        ),
        (
            "grounded_multi",
            [
                '<tool_call name="get_pathways">{"gene_symbols": ["HBB"]}</tool_call>',
                "KLF1 activates erythroid pathways [source: get_pathways → HALLMARK_ERYTHROID].",
            ],
        ),
        (
            "ungrounded",
            [
                "KLF1 is a known regulator [source: retrieve_literature → PMID:99999999].",
            ],
        ),
        (
            "no_evidence",
            [
                '<tool_call name="get_pathways">{"gene_symbols": ["ZZZ9"]}</tool_call>',
                "No supporting evidence found in the current analysis.",
            ],
        ),
    ]

    grounding_rates: list[float] = []
    valid_citations: list[bool] = []
    fallback_correct: list[bool] = []

    for _, responses in probes:
        resp = run("probe", _scripted_llm(*responses), tools=tools)
        grounding_rates.append(
            agent_eval.grounding_rate(resp.answer, resp.evidence)
        )
        valid_citations.append(
            not agent_eval.check_citation_validity(resp.answer, resp.evidence)
        )
        fallback_correct.append(
            agent_eval.check_no_evidence_fallback(resp.answer, resp.evidence)
        )

    return {
        "grounding_rate": sum(grounding_rates) / len(grounding_rates),
        "citation_validity_rate": sum(valid_citations) / len(valid_citations),
        "no_evidence_fallback_rate": sum(fallback_correct) / len(fallback_correct),
    }


# ------------------------------------------------------------- table printing


def print_ablation_table(
    prediction_metrics: dict[str, dict[str, dict[str, float]]],
    retrieval_metrics: dict[str, dict[int, dict[str, float]]],
    agent_metrics: dict[str, float],
) -> None:
    """Print a consolidated comparison table across all four ablation levels."""
    bar = "=" * 72

    print("\n" + bar)
    print("  ABLATION RESULTS")
    print(bar)

    # --- Level 1 + 2: prediction metrics -------------------------------------
    models = [BASELINE_MODEL, FILM_HEAD_MODEL]
    splits = sorted({s for m in models for s in prediction_metrics.get(m, {})})
    if splits:
        print("\n  ── Prediction metrics (baseline vs. film_head) ──")
        for split in splits:
            print(f"\n    [{split}]")
            header = f"    {'metric':<22s}"
            for m in models:
                header += f" {m:>14s}"
            print(header)
            print(f"    {'─' * 22}" + f" {'─' * 14}" * len(models))
            for key in PREDICTION_METRIC_KEYS:
                row = f"    {key:<22s}"
                for m in models:
                    val = prediction_metrics.get(m, {}).get(split, {}).get(key)
                    row += f" {val:>14.6f}" if val is not None else f" {'—':>14s}"
                print(row)

    # --- Level 3: retrieval metrics ------------------------------------------
    if retrieval_metrics:
        k_values = sorted({k for v in retrieval_metrics.values() for k in v})
        metric_names = ["recall", "precision", "mrr"]
        systems = list(retrieval_metrics.keys())
        print("\n  ── Retrieval metrics (+retrieval) ──")
        for k in k_values:
            print(f"\n    [K={k}]")
            header = f"    {'system':<28s}"
            for m in metric_names:
                header += f" {m:>12s}"
            print(header)
            print(f"    {'─' * 28}" + f" {'─' * 12}" * len(metric_names))
            for sys_name in systems:
                row = f"    {sys_name:<28s}"
                for m in metric_names:
                    val = retrieval_metrics[sys_name].get(k, {}).get(m)
                    row += f" {val:>12.6f}" if val is not None else f" {'—':>12s}"
                print(row)

    # --- Level 4: agent metrics ----------------------------------------------
    if agent_metrics:
        print("\n  ── Agent-level metrics (full agent) ──")
        for key, val in agent_metrics.items():
            print(f"    {key:<34s} {val:>12.6f}")

    print("\n" + bar + "\n")


# ----------------------------------------------------------------------- main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default=str(PROJECT_ROOT / "results" / "metrics.csv"),
        help="Path to the prediction-metrics CSV.",
    )
    parser.add_argument(
        "--model-config",
        default=str(PROJECT_ROOT / "configs" / "model.yaml"),
    )
    parser.add_argument(
        "--training-config",
        default=str(PROJECT_ROOT / "configs" / "training.yaml"),
    )
    parser.add_argument(
        "--data-config",
        default=str(PROJECT_ROOT / "configs" / "data.yaml"),
    )
    parser.add_argument(
        "--embeddings-dir",
        default=str(PROJECT_ROOT / "data" / "embeddings"),
        help="Directory with cached cell/gene embeddings (for film_head + retrieval).",
    )
    parser.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "data" / "processed" / "perturbseq.h5ad"),
        help="Path to the processed AnnData file (for retrieval).",
    )
    parser.add_argument(
        "--indices-dir",
        default=str(PROJECT_ROOT / "data" / "indices"),
        help="Directory to read/write FAISS retrieval indices.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results" / "ablation_results.json"),
        help="Path to save the consolidated ablation results as JSON.",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Re-run all training/evaluation stages before comparing.",
    )
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="Re-run the PCA+MLP baseline training.",
    )
    parser.add_argument(
        "--run-film-head",
        action="store_true",
        help="Re-run the FiLM-MLP prediction-head training.",
    )
    parser.add_argument(
        "--run-retrieval",
        action="store_true",
        help="Re-build the FAISS retrieval indices.",
    )
    parser.add_argument(
        "--skip-agent-eval",
        action="store_true",
        help="Skip the scripted agent-eval probes.",
    )
    args = parser.parse_args(argv)

    scripts_dir = PROJECT_ROOT / "scripts"
    results_path = Path(args.results)
    embeddings_dir = Path(args.embeddings_dir)
    indices_dir = Path(args.indices_dir)

    # 1. Run requested stages.
    run_baseline = args.run_all or args.run_baseline
    run_film_head = args.run_all or args.run_film_head
    run_retrieval = args.run_all or args.run_retrieval

    if run_baseline:
        rc = run_stage("baseline (PCA+MLP)", [
            sys.executable, str(scripts_dir / "train_baseline.py"),
            "--model-config", args.model_config,
            "--training-config", args.training_config,
            "--data-config", args.data_config,
            "--results", str(results_path),
        ])
        if rc != 0:
            return rc

    if run_film_head:
        rc = run_stage("film_head (scGPT FiLM-MLP)", [
            sys.executable, str(scripts_dir / "train_prediction_head.py"),
            "--model-config", args.model_config,
            "--training-config", args.training_config,
            "--data-config", args.data_config,
            "--embeddings-dir", str(embeddings_dir),
            "--results", str(results_path),
        ])
        if rc != 0:
            return rc

    if run_retrieval:
        rc = run_stage("+retrieval (FAISS indices)", [
            sys.executable, str(scripts_dir / "build_retrieval_index.py"),
            "--embeddings-dir", str(embeddings_dir),
            "--data", args.data,
            "--output-dir", str(indices_dir),
        ])
        if rc != 0:
            return rc

    # 2. Gather results from disk.
    print("Gathering ablation results ...")
    prediction_metrics = load_prediction_metrics(results_path)
    retrieval_metrics = load_retrieval_metrics(
        indices_dir / "retrieval_results.json"
    )
    agent_metrics: dict[str, float] = {}
    if not args.skip_agent_eval:
        agent_metrics = run_agent_eval_probes()

    # 3. Print consolidated table.
    print_ablation_table(prediction_metrics, retrieval_metrics, agent_metrics)

    # 4. Persist consolidated results.
    output = {
        "created": datetime.now(timezone.utc).isoformat(),
        "prediction": prediction_metrics,
        "retrieval": retrieval_metrics,
        "agent": agent_metrics,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(output, fh, indent=2)
    print(f"Consolidated results saved to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
