# DDT_Mjlab — NP3O Locomotion for Wheel-Legged Robots

Locomotion training for the **D1** (quadruped with wheels), **D1H** (wheel-legged biped), and **TITA** (wheel-legged biped) robots, using **NP3O** (BarlowTwins-augmented constrained PPO) built on [mjlab](https://github.com/mujocolab/mjlab).

This is an **mjlab-native** migration from the original IsaacLab-based `DDT_Lab-np3o` codebase, ported to the MuJoCo-backed mjlab framework with behavior equivalence.

---

## Prerequisites


| Dependency | Version |
| ---------- | ------- |
| Python     | 3.11    |
| CUDA       | > 12.4  |
| mjlab      | 1.4.0   |

---

## Installation

### 1. Create conda environment

```bash
conda create -n <your_env_name> python=3.11
conda activate your_env_name
```

### 2. Install PyTorch with CUDA support

mjlab 1.4.0 recommends CUDA version > 12.4. Lower versions may work but training will be significantly slower.

```bash
# Example: install PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install mjlab

```bash
pip install mjlab==1.4.0
```

### 4. Clone this repo

```bash
git clone <repo-url> ddt_mjlab-np3o
cd ddt_mjlab-np3o
```

### 5. Verify installation

```bash
python scripts/train.py --list-tasks
```

Expected output:

```
Available tasks:
  Mjlab-Velocity-Flat-D1
  Mjlab-Velocity-Rough-D1
  Mjlab-Velocity-Flat-D1H
  Mjlab-Velocity-Rough-D1H
  Mjlab-Velocity-Flat-TITA
  Mjlab-Velocity-Rough-TITA
```

---

## Training

```bash
conda activate <your_env_name>

# D1 — flat ground
python scripts/train.py Mjlab-Velocity-Flat-D1 # Task's name
```

### Common flags


| Flag               | Default | Description                           |
| ------------------ | ------- | ------------------------------------- |
| `--num_envs`       | 4096    | Number of parallel environments       |
| `--max_iterations` | 5000    | Override total training iterations    |
| `--device`         | `auto`  | Training device (`cuda:0` or `cpu`)   |
| `--resume`         | False   | Resume training from a checkpoint     |
| `--load_run`       | None    | Run directory to load checkpoint from |

### Logs

Checkpoints and TensorBoard events are written to:

```
logs/np3o/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>/
├── model_<iter>.pt      # policy checkpoint
├── events.out.tfevents…  # TensorBoard
└── ...
```

## Resume training

```bash
python scripts/train.py Mjlab-Velocity-Flat-D1 \
    --num_envs=4096 \
    --resume \
    --load_run logs/np3o/d1_flat/YYYY-MM-DD_HH-MM-SS/model_xx.pt
```

---

## Play / Evaluate

```bash
# Play with a specific checkpoint
python scripts/play.py Mjlab-Velocity-Flat-D1 \
    --checkpoint-file logs/np3o/d1_flat/YYYY-MM-DD_HH-MM-SS/model_xxxx.pt
```

---

## Available robots & tasks


| Robot    | Description           | Flat task                  | Rough task                  |
| -------- | --------------------- | -------------------------- | --------------------------- |
| **D1**   | Quadruped with wheels | `Mjlab-Velocity-Flat-D1`   | `Mjlab-Velocity-Rough-D1`   |
| **D1H**  | Wheel-legged biped    | `Mjlab-Velocity-Flat-D1H`  | `Mjlab-Velocity-Rough-D1H`  |
| **TITA** | Wheel-legged biped    | `Mjlab-Velocity-Flat-TITA` | `Mjlab-Velocity-Rough-TITA` |

---

## Project Structure

Key source files:

```
np3o/
├── algorithms/np3o/              # NP3O algorithm, BarlowTwins actor-critic, runner, wrapper
│   ├── actor_critic.py
│   ├── np3o.py
│   ├── runner.py
│   └── wrapper.py
└── tasks/locomotion/
    ├── cmdp/                     # Commands, costs, rewards, observations, terminations, curriculums
    │   ├── commands.py
    │   ├── costs.py
    │   ├── rewards.py
    │   ├── observations.py
    │   ├── terminations.py
    │   └── curriculums.py
    ├── config/                   # Per-robot, per-terrain env configs
    │   ├── d1/
    │   ├── d1h/
    │   └── tita/
    └── rl/                       # Shared NP3O runner hyper-parameters
        └── rl_cfg.py
```

---

## Task Registration

Tasks are registered automatically via `mjlab.tasks.registry.register_mjlab_task`.

When you run a script, the side-effect import

```python
import np3o.tasks.locomotion  # noqa: F401
```

triggers `np3o/tasks/locomotion/__init__.py` to auto-discover every package under `config/` (e.g. `d1/`, `d1h/`, `tita/`) and import them. Each package then registers its flat / rough tasks.

### Adding your own tasks

Suppose you want a custom D1 task with different rewards. Create a new package under `config/` (e.g. `my_tasks/`) and reuse the existing D1 assets:

```python
# np3o/tasks/locomotion/config/my_tasks/__init__.py
from mjlab.tasks.registry import register_mjlab_task
from np3o.algorithms.np3o.runner import MjlabNP3ORunner

# Reuse existing D1 env configs, or import your own variants
from ..d1.d1_flat_cfg import d1_flat_env_cfg, d1_flat_play_env_cfg
from ...rl.rl_cfg import d1_np3o_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Mytask-D1",
    env_cfg=d1_flat_env_cfg(play=False),
    play_env_cfg=d1_flat_play_env_cfg(),
    rl_cfg=d1_np3o_runner_cfg(experiment_name="mytask_d1", max_iterations=10000),
    runner_cls=MjlabNP3ORunner,
)
```

No manual per-package import is needed in `train.py` or `play.py` — `np3o/tasks/locomotion/__init__.py` auto-discovers `my_tasks/` at runtime.

Verify that your task is registered:

```bash
python scripts/train.py --list-tasks
```

