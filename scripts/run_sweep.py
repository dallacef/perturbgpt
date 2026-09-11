#!/usr/bin/env python3
"""Launch a W&B hyperparameter sweep for the baseline model.

Usage:
    python scripts/run_sweep.py --count 20
    python scripts/run_sweep.py --count 20 --sweep configs/sweep_baseline.yaml

This creates a W&B sweep from the YAML config and runs ``count`` agents,
each of which calls ``train_baseline.py --use-wandb`` with hyperparameters
sampled by the sweep controller.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

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
        "--epochs",
        type=int,
        default=100,
        help="Epochs per sweep run",
    )
    args = parser.parse_args(argv)

    # 1. Create the sweep
    create_cmd = [
        "wandb", "sweep", args.sweep,
        "--project", args.project,
    ]
    print(f"Creating sweep: {' '.join(create_cmd)}")
    result = subprocess.run(create_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: wandb sweep creation failed:\n{result.stderr}")
        return 1
    print(result.stdout)
    # Extract sweep ID from output (last line typically: "wandb agent <sweep_id>")
    sweep_id = result.stdout.strip().split()[-1]
    print(f"Sweep ID: {sweep_id}")

    # 2. Launch agents
    agent_cmd = [
        "wandb", "agent", sweep_id,
        "--project", args.project,
        "--count", str(args.count),
    ]
    print(f"\nLaunching {args.count} agents: {' '.join(agent_cmd)}")
    # Each agent calls: python scripts/train_baseline.py --use-wandb --epochs N
    # The wandb agent infrastructure injects hyperparameters via wandb.config
    result = subprocess.run(agent_cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
