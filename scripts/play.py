#!/usr/bin/env python3
"""Play a trained D1 locomotion policy.

Usage::

    conda activate mjlab_env

    # List available tasks
    python scripts/play.py --list-tasks

    # Play with a trained checkpoint
    python scripts/play.py Mjlab-Velocity-Flat-D1 --checkpoint-file logs/np3o/d1_flat/.../model_1000.pt

    # Zero agent (stand still)
    python scripts/play.py Mjlab-Velocity-Flat-D1 --agent=zero

    # Random agent
    python scripts/play.py Mjlab-Velocity-Flat-D1 --agent=random

    # Viser viewer (browser)
    python scripts/play.py Mjlab-Velocity-Flat-D1 --checkpoint-file ... --viewer=viser

    # Native viewer (MuJoCo window)
    python scripts/play.py Mjlab-Velocity-Flat-D1 --checkpoint-file ... --viewer=native

    # CPU
    python scripts/play.py Mjlab-Velocity-Flat-D1 --checkpoint-file ... --device=cpu

    # Rough terrain
    python scripts/play.py Mjlab-Velocity-Rough-D1 --checkpoint-file ...
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
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


def _resolve_checkpoint(checkpoint: str | None, task_id: str) -> str | None:
    """Resolve a checkpoint path.

    - explicit path → use it (directory = latest model_*.pt inside)
    - None → auto-find latest run under logs/np3o/<experiment_name>/
    """
    if checkpoint is not None:
        path = Path(checkpoint)
        if path.is_dir():
            matches = sorted(path.glob("model_*.pt"))
            if not matches:
                raise FileNotFoundError(f"No model_*.pt found in: {checkpoint}")
            return str(matches[-1])
        return checkpoint

    # Auto-discover latest run — use experiment_name from registered rl_cfg.
    rl_cfg = load_rl_cfg(task_id)
    exp_name = rl_cfg["runner"]["experiment_name"]
    task_log_dir = Path("logs") / "np3o" / exp_name
    if not task_log_dir.is_dir():
        return None
    runs = sorted(task_log_dir.iterdir(), reverse=True)
    for run in runs:
        if run.is_dir():
            models = sorted(run.glob("model_*.pt"))
            if models:
                return str(models[-1])
    return None


def _detect_num_costs(ckpt_path: str) -> tuple[int, int]:
    """Read num_costs and iteration from an NP3O checkpoint.

    Returns:
        (num_costs, iter).
    """
    state = torch.load(ckpt_path, map_location="cpu")
    ckpt_iter = state.get("iter", 0)
    cost_keys = [k for k in state["model_state_dict"] if k.startswith("cost.") and k.endswith(".weight")]
    num_costs = state["model_state_dict"][cost_keys[-1]].shape[0] if cost_keys else 1
    return num_costs, ckpt_iter


def main() -> None:
    all_tasks = list_tasks()

    parser = argparse.ArgumentParser(
        description="Play a trained D1 locomotion policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("task_id", type=str, nargs="?", default=None,
                        help=f"Task ID (choices: {', '.join(all_tasks)})")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List available tasks and exit")
    parser.add_argument("--checkpoint-file", type=str, default=None,
                        help="Path to a trained model .pt file (or directory with model_*.pt)")
    parser.add_argument("--agent", type=str, default="trained",
                        choices=["zero", "random", "trained"],
                        help="Agent type (default: trained)")
    parser.add_argument("--viewer", type=str, default="auto",
                        choices=["auto", "native", "viser"],
                        help="Viewer type (default: auto)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda:0 if available, else cpu)")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip ONNX export")
    parser.add_argument("--export-output", type=str, default=None,
                        help="Output .onnx path (default: <checkpoint_dir>/exported/policy.onnx)")
    parser.add_argument("--num_envs", type=int, default=50,
                        help="Number of play environments.")
    parser.add_argument("--use-train-randomization", action="store_true",
                        help="Keep train-time randomization/push events during play.")
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

    # Resolve checkpoint
    ckpt_path = _resolve_checkpoint(args.checkpoint_file, task_id)

    if ckpt_path:
        num_costs, ckpt_iter = _detect_num_costs(ckpt_path)
        print(f"Checkpoint: {ckpt_path}  (iter={ckpt_iter}, num_costs={num_costs})")
    else:
        num_costs = 5
        print(f"No checkpoint found for task '{task_id}', using --agent mode")

    # ---- Build env ----
    # Build from train cfg (not play) to keep terrain, commands, reset profile
    # consistent with training. Only viewer/noise/runtime overrides below.
    cfg = load_env_cfg(task_id)
    cfg.scene.num_envs = args.num_envs

    if "policy" in cfg.observations:
        cfg.observations["policy"].enable_corruption = False

    if not args.use_train_randomization:
        for event_name in (
            "base_external_force_torque",
            "push_robot",
            "randomize_actuator_gains",
        ):
            cfg.events.pop(event_name, None)

    if hasattr(cfg, "viewer"):
        cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_ROOT
        cfg.viewer.entity_name = "robot"

    from mjlab.envs import ManagerBasedRlEnv
    raw_env = ManagerBasedRlEnv(cfg, device=device)
    cost_terms = cfg.cost_terms if num_costs > 1 else None
    env = MjlabNP3OWrapper(raw_env, cost_terms=cost_terms, device=device, use_action_filter=True)

    # ---- Build runner & load ----
    train_cfg = load_rl_cfg(task_id)
    runner = MjlabNP3ORunner(env, train_cfg, log_dir=None, device=device)

    if ckpt_path and args.agent == "trained":
        runner.load(ckpt_path, load_optimizer=False)
        print(f"Loaded checkpoint: iter {runner.current_learning_iteration}")

    # ---- ONNX export ----
    if ckpt_path and args.agent == "trained" and not args.no_export:
        if args.export_output:
            export_dir = str(Path(args.export_output).parent)
            os.makedirs(export_dir, exist_ok=True)
            result = runner.alg.actor_critic.save_torch_jit_policy(export_dir, device)
            default_onnx = os.path.join(export_dir, "policy.onnx")
            target_onnx = args.export_output
            if os.path.abspath(default_onnx) != os.path.abspath(target_onnx):
                shutil.move(default_onnx, target_onnx)
                print(f"  ONNX:        {target_onnx}")
            else:
                print(f"  ONNX:        {default_onnx}")
        else:
            export_dir = os.path.join(os.path.dirname(ckpt_path), "exported")
            print(f"Exporting to: {export_dir}")
            result = runner.alg.actor_critic.save_torch_jit_policy(export_dir, device)
            print(f"  TorchScript: {result['jit']}")
            print(f"  ONNX:        {result['onnx']}")

    # ---- Play loop ----
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    viewer_type = args.viewer
    if viewer_type == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        viewer_type = "native" if has_display else "viser"

    if args.agent == "trained" and ckpt_path:
        policy = runner.get_inference_policy(device)
        print(f"Running trained policy with {viewer_type} viewer (Ctrl+C to stop)...")
    elif args.agent == "trained" and not ckpt_path:
        print("[WARN] Agent is 'trained' but no checkpoint provided, falling back to zero agent.")
        n_actions = env.num_actions
        device_t = torch.device(device)

        def policy(obs):
            return torch.zeros(obs.shape[0], n_actions, device=device_t)
    elif args.agent == "zero":
        n_actions = env.num_actions
        device_t = torch.device(device)

        def policy(obs):
            return torch.zeros(obs.shape[0], n_actions, device=device_t)
        print(f"Running zero agent with {viewer_type} viewer (Ctrl+C to stop)...")
    elif args.agent == "random":
        n_actions = env.num_actions
        device_t = torch.device(device)

        def policy(obs):
            return torch.randn(obs.shape[0], n_actions, device=device_t)
        print(f"Running random agent with {viewer_type} viewer (Ctrl+C to stop)...")

    if viewer_type == "native":
        class _PlayViewer(NativeMujocoViewer):
            def _add_env_selection_marker(self, viewer):
                pass

        _PlayViewer(env, policy).run()
    elif viewer_type == "viser":
        ViserPlayViewer(env, policy).run()

    raw_env.close()


if __name__ == "__main__":
    main()
