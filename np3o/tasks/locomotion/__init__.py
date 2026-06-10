"""Locomotion velocity-tracking tasks for mjlab (D1 + D1H + TITA).

Importing this package automatically discovers and registers all robot configs
under ``config/``.  No manual per-robot import is needed in scripts.
"""

from __future__ import annotations

import importlib
from pathlib import Path

_config_dir = Path(__file__).resolve().parent / "config"
for _p in sorted(_config_dir.iterdir()):
    if _p.is_dir() and (_p / "__init__.py").exists():
        _module = f"np3o.tasks.locomotion.config.{_p.name}"
        importlib.import_module(_module)
