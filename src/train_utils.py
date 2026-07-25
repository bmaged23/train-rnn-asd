"""Shared start-of-run cleanup for scripts/train_*.py and scripts/evaluate_*.py.

Every run starts from a clean slate: reset_dir(dir_path) deletes `dir_path`
(and everything in it) if present, then recreates it empty. Used at the start
of both train_*.py (config.CHECKPOINTS_DIR, config.LOGS_DIR — shared across
every training script, so this clears every prior run's checkpoints/logs, not
just the current script's own) and evaluate_*.py (each script's own
evaluation_*/ directory) runs, so a rerun never mixes with — or silently
appends to, in the case of the log file — a previous run's output.
"""
from __future__ import annotations

import shutil
from pathlib import Path


def reset_dir(dir_path: Path) -> None:
    shutil.rmtree(dir_path, ignore_errors=True)
    dir_path.mkdir(parents=True, exist_ok=True)
