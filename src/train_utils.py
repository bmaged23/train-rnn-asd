"""Shared start-of-run cleanup for scripts/modeling/train/*.py and
scripts/modeling/evaluate/*.py.

The legacy reset_dir helper is used by evaluation entry points to replace their
own named report directory. Training entry points create the shared checkpoints/
and logs/ folders without deleting other model families; their named files may
still be overwritten by a rerun. Never pass a shared data/checkpoint directory
to reset_dir. Release evaluations should use isolated output paths and preserve
previous failed runs rather than calling this destructive helper.
"""
from __future__ import annotations

import shutil
from pathlib import Path


def reset_dir(dir_path: Path) -> None:
    shutil.rmtree(dir_path, ignore_errors=True)
    dir_path.mkdir(parents=True, exist_ok=True)
