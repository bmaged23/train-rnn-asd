"""Explicitly (re)builds config.IMAGE_RESIZED_CACHE_DIR — see
src/image_dataset.py's build_image_cache() docstring (and this module's own
docstring) for what this does and why: moves the raw-pixel CNN+LSTM
pipeline's decode+pad-to-square+resize work out of the per-epoch hot path,
doing it once per run instead.

scripts/modeling/train/image_windowed_all_combined.py and
scripts/modeling/evaluate/image_windowed_all_combined_per_source.py already
call build_image_cache() themselves at the start of every run (explicit
user request, 2026-08-01: process the data from the beginning every
training run, never silently reuse a cache from an earlier, possibly
differently-configured run) — this script exists only so the cache can be
(re)built standalone, e.g. to inspect its output or time the build in
isolation, without also running a full training/eval pass.

Usage:
    python scripts/features/build_image_resized_cache.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
from image_dataset import build_image_cache

if __name__ == "__main__":
    build_image_cache()
