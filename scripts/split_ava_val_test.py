"""Splits data/processed_ava/val/combined_landmarks.csv into val_split.csv
and test_split.csv, exactly like scripts/split_val_test.py does for
UniTalk-ASD's own val data — same stratified-by-track logic (bucketed on
detection-quality x speaking-majority, VAL_SPLIT_RATIO, VAL_TEST_SPLIT_SEED),
reused unmodified via split_val_test.main()'s path overrides. No separate AVA
"test video" pool exists (AVA itself only ships train/val), so test is carved
out of val here the same way it already is for UniTalk.

Usage:
    python scripts/split_ava_val_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import AVA_PROCESSED_DATASET_DIR, COMBINED_LANDMARKS_FILENAME, TEST_SPLIT_FILENAME, VAL_SPLIT_FILENAME
from split_val_test import main as split_main

if __name__ == "__main__":
    ava_val_dir = AVA_PROCESSED_DATASET_DIR / "val"
    split_main(
        source_csv=ava_val_dir / COMBINED_LANDMARKS_FILENAME,
        val_out=ava_val_dir / VAL_SPLIT_FILENAME,
        test_out=ava_val_dir / TEST_SPLIT_FILENAME,
    )
