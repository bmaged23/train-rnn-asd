"""Splits data/processed_wasd/val/combined_landmarks.csv into val_split.csv
and test_split.csv, exactly like scripts/splits/split_val_test.py does for
UniTalk-ASD's own val data — same stratified-by-track logic (bucketed on
detection-quality x speaking-majority, VAL_SPLIT_RATIO, VAL_TEST_SPLIT_SEED),
reused unmodified via split_val_test.main()'s path overrides. No separate WASD
"test video" pool exists (WASD's own train_orig.csv/val_orig.csv are the only
splits), so test is carved out of val here the same way it already is for
UniTalk and AVA (see scripts/splits/split_ava_val_test.py).

Usage:
    python scripts/splits/split_wasd_val_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import COMBINED_LANDMARKS_FILENAME, TEST_SPLIT_FILENAME, VAL_SPLIT_FILENAME, WASD_PROCESSED_DATASET_DIR
from split_val_test import main as split_main

if __name__ == "__main__":
    wasd_val_dir = WASD_PROCESSED_DATASET_DIR / "val"
    split_main(
        source_csv=wasd_val_dir / COMBINED_LANDMARKS_FILENAME,
        val_out=wasd_val_dir / VAL_SPLIT_FILENAME,
        test_out=wasd_val_dir / TEST_SPLIT_FILENAME,
    )
