#!/usr/bin/env python3
"""Train a D1 locomotion policy with NP3O.

Usage::

    conda activate mjlab_env

    # List available tasks
    python scripts/train.py --list-tasks

    # Flat terrain training
    python scripts/train.py Mjlab-Velocity-Flat-D1

    # Rough terrain training
    python scripts/train.py Mjlab-Velocity-Rough-D1

    # Custom env count and iterations
    python scripts/train.py Mjlab-Velocity-Flat-D1 --num_envs=2048 --max_iterations=5000

    # CPU training
    python scripts/train.py Mjlab-Velocity-Flat-D1 --device=cpu

    # Resume from checkpoint
    python scripts/train.py Mjlab-Velocity-Flat-D1 --resume --load_run logs/np3o/d1_flat/my_run
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

# Add project root to sys.path so np3o is importable.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Trigger task registration (side effect: populates mjlab task registry).
import np3o.tasks.locomotion  # noqa: F401 (auto-discovers all robot configs)

from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from np3o.algorithms.np3o.wrapper import MjlabNP3OWrapper
from np3o.algorithms.np3o.runner import MjlabNP3ORunner


def main() -> None:
    all_tasks = list_tasks()

    parser = argparse.ArgumentParser(
        description="Train a D1 locomotion policy with NP3O",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("task_id", type=str, nargs="?", default=None,
                        help=f"Task ID (choices: {', '.join(all_tasks)})")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List available tasks and exit")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda:0 if available, else cpu)")
    parser.add_argument("--num_envs", type=int, default=None,
                        help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=None,
                        help="Maximum number of training iterations")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name of the experiment (overrides config)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name of the run")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Log directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from a checkpoint")
    parser.add_argument("--load_run", type=str, default=None,
                        help="Run directory to load checkpoint from")
    parser.add_argument("--no-costs", action="store_true",
                        help="Disable Lagrangian constraint costs")
    parser.add_argument("--no-imi", action="store_true",
                        help="Disable BarlowTwins imitation loss")
    args = parser.parse_args()

    if args.list_tasks:
        print("Available tasks:")
        for t in all_tasks:
            print(f"  {t}")
        return

    if args.task_id is None:
        parser.error(f"task_id is required. Choices: {all_tasks}")
    if args.task_id not in all_tasks:
        parser.error(f"Unknown task '{args.task_id}'. Choices: {all_tasks}")

    task_id = args.task_id

    # ---- Device ----
    if args.device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Task: {task_id}")
    print(f"Device: {device}")

    # ---- Load configs from registry ----
    cfg = load_env_cfg(task_id)
    train_cfg = load_rl_cfg(task_id)

    if args.max_iterations is not None:
        train_cfg["runner"]["max_iterations"] = args.max_iterations
    if args.no_imi:
        train_cfg["policy"]["imi_flag"] = False

    num_envs = args.num_envs if args.num_envs is not None else 4096
    print(f"Num envs: {num_envs}")
    if args.max_iterations:
        print(f"Max iterations: {args.max_iterations}")

    cfg.scene.num_envs = num_envs

    # ---- Build env ----
    print(f"\nBuilding environment ({task_id}, {num_envs} envs)...")
    from mjlab.envs import ManagerBasedRlEnv
    raw_env = ManagerBasedRlEnv(cfg, device=device)

    if args.no_costs:
        cost_terms = None
        print("Cost terms: disabled")
    else:
        cost_terms = cfg.cost_terms
        print(f"Cost terms: {list(cost_terms.keys())}")

    env = MjlabNP3OWrapper(raw_env, cost_terms=cost_terms, device=device, use_action_filter=True)
    print(f"NP3O env: {env.num_envs} envs, {env.num_actions} actions, "
          f"n_prop={env.n_proprio}, n_critic={env.n_critic}, "
          f"n_priv={env.n_priv_latent}, n_scan={env.n_scan}, "
          f"num_costs={env.num_costs}")

    # ---- Log directory ----
    if args.log_dir is None:
        exp_name = args.experiment_name or train_cfg["runner"]["experiment_name"]
        run_label = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.resume and args.load_run:
            load_path = Path(args.load_run)
            log_dir = str(load_path.parent if load_path.is_file() else load_path)
        else:
            log_dir = os.path.join("logs", "np3o", exp_name, run_label)
        train_cfg["runner"]["experiment_name"] = exp_name
        train_cfg["runner"]["run_name"] = run_label
    else:
        log_dir = args.log_dir

    os.makedirs(log_dir, exist_ok=True)
    print(f"Log dir: {log_dir}")

    # ---- Runner ----
    runner = MjlabNP3ORunner(env, train_cfg, log_dir=log_dir, device=device)

    # ---- Resume ----
    if args.resume:
        if args.load_run:
            load_path = Path(args.load_run)
            if load_path.is_file():
                resume_path = str(load_path)
            else:
                checkpoints = sorted(load_path.glob("model_*.pt"))
                if not checkpoints:
                    print(f"[ERROR] No model_*.pt found in: {args.load_run}")
                    sys.exit(1)
                resume_path = str(checkpoints[-1])
            print(f"Resuming from: {resume_path}")
            runner.load(resume_path, load_optimizer=True)

    # ---- Train ----
    max_iter = train_cfg["runner"]["max_iterations"]
    print(f"\nStarting training: {max_iter} iterations "
          f"({max_iter * train_cfg['runner']['num_steps_per_env'] * env.num_envs} total env steps)")
    runner.learn(num_learning_iterations=max_iter, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
