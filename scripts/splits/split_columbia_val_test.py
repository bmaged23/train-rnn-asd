"""Splits data/processed_columbia/val/combined_landmarks.csv into
val_split.csv and test_split.csv, exactly like scripts/splits/split_val_test.py
does for UniTalk-ASD's own val data — same stratified-by-track logic
(bucketed on detection-quality x speaking-majority, VAL_SPLIT_RATIO,
VAL_TEST_SPLIT_SEED), reused unmodified via split_val_test.main()'s path
overrides.

Like UniTalk/AVA/WASD, Columbia's raw "val" split holds out
config.COLUMBIA_NUM_VAL_SPEAKERS of its 6 speakers (see
scripts/dataset/download_columbia_subset.py's module docstring for how
train/val are divided) — this script further splits that small val pool
into val_split/test_split, same stratified logic as the other three
sources, just operating on far fewer tracks (2 speakers in, ~1/1 out) given
Columbia's size.

Usage:
    python scripts/splits/split_columbia_val_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import COLUMBIA_PROCESSED_DATASET_DIR, COMBINED_LANDMARKS_FILENAME, TEST_SPLIT_FILENAME, VAL_SPLIT_FILENAME
from split_val_test import main as split_main

if __name__ == "__main__":
    columbia_val_dir = COLUMBIA_PROCESSED_DATASET_DIR / "val"
    split_main(
        source_csv=columbia_val_dir / COMBINED_LANDMARKS_FILENAME,
        val_out=columbia_val_dir / VAL_SPLIT_FILENAME,
        test_out=columbia_val_dir / TEST_SPLIT_FILENAME,
    )
