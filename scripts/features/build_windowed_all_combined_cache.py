"""Explicitly (re)builds config.WINDOWED_CACHE_DIR's cached
WindowedLandmarkSequenceDataset for the TRAIN split of the 6-dataset
combined mix scripts/modeling/train/windowed_all_combined.py and its
evaluate counterpart use — same extra_sources list both scripts pass to
get_windowed_dataloader(). Train only — val/test are never cached (explicit
user request, see config.WINDOWED_CACHE_DIR's comment for why), so there's
nothing for this script to pre-build for them; they stay cheap to rebuild
from the CSVs on every use regardless.

Not required — WindowedLandmarkSequenceDataset already builds-then-caches
train transparently the first time any script constructs it, and loads
straight from cache on every later call with the same settings (see
config.WINDOWED_CACHE_DIR's comment for exactly what "same settings" means).
This script exists purely to pay that first-build cost (CSV parsing +
windowing + augmentation — several minutes for the full combined train
split) once, deliberately, ahead of time — e.g. right after changing a
setting that invalidates the old cache (WINDOW_SECONDS, AUGMENT_*,
USE_MOUTH_LANDMARKS_ONLY, etc.) — rather than having it happen as a
side effect of the next `python scripts/modeling/train/windowed_all_combined.py`
run.

Usage:
    python scripts/features/build_windowed_all_combined_cache.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
from windowed_dataset import (
    ava_source,
    avspeech_source,
    columbia_source,
    get_windowed_dataloader,
    speakingfaces_source,
    wasd_source,
)

_EXTRA_SOURCES = [ava_source, wasd_source, columbia_source, avspeech_source, speakingfaces_source]

if __name__ == "__main__":
    t0 = time.time()
    # get_windowed_dataloader (not WindowedLandmarkSequenceDataset directly)
    # so train correctly gets WINDOW_TRAIN_STRIDE (overlapping) instead of
    # WINDOW_STRIDE — using the wrong stride here would build a cache under
    # a different key than what scripts/modeling/train/windowed_all_combined.py
    # actually requests, silently wasting the whole point of pre-building it.
    loader = get_windowed_dataloader("train", extra_sources=[fn("train") for fn in _EXTRA_SOURCES])
    print(f"[train] {len(loader.dataset)} windows ready in {time.time() - t0:.1f}s")
