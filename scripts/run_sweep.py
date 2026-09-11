#!/usr/bin/env python3
"""Launch a W&B hyperparameter sweep for the baseline model.

Usage:
    python scripts/run_sweep.py --count 20
    python scripts/run_sweep.py --count 20 --sweep configs/sweep_baseline.yaml

Uses the W&B Python SDK (wandb.sweep) to create the sweep, which returns
the sweep ID directly — no fragile CLI output parsing. Then launches
``count`` sweep agents via subprocess. Each agent calls
``train_baseline.py``, which auto-detects the sweep environment (via the
WANDB_SWEEP_ID env var set by the agent) and reads hyperparameters from
wandb.config.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        default=str(PROJECT_ROOT / "configs" / "sweep_baseline.yaml"),
        help="Path to sweep YAML config",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of sweep runs (agents) to launch",
    )
    parser.add_argument(
        "--project",
        default="perturbgpt-baseline",
        help="W&B project name",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity (username or team). Defaults to your logged-in user.",
    )
    args = parser.parse_args(argv)

    # 1. Load sweep config from YAML
    with open(args.sweep) as fh:
        sweep_config = yaml.safe_load(fh)

    # 2. Create the sweep via Python SDK (returns sweep_id directly)
    try:
        import wandb
    except ImportError:
        print("ERROR: wandb is not installed. Run: pip install wandb")
        return 1

    print(f"Creating sweep in project '{args.project}'...")
    sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
    print(f"Sweep created successfully!")
    print(f"  Sweep ID:  {sweep_id}")
    print(f"  Project:   {args.project}")
    entity_str = args.entity or wandb.api.default_entity
    print(f"  Entity:    {entity_str}")
    print(f"  Dashboard: https://wandb.ai/{entity_str}/{args.project}/sweeps/{sweep_id}")

    # 3. Launch agent(s) via CLI
    # The agent runs the 'program' field from the sweep config
    # (scripts/train_baseline.py). The training script auto-detects the
    # sweep via WANDB_SWEEP_ID and reads hyperparams from wandb.config.
    agent_cmd = [
        "wandb", "agent",
        "--project", args.project,
        "--count", str(args.count),
    ]
    if args.entity:
        agent_cmd += ["--entity", args.entity]
    agent_cmd.append(sweep_id)

    print(f"\nLaunching agent ({args.count} runs): {' '.join(agent_cmd)}")
    result = subprocess.run(agent_cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
