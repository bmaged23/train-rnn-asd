"""Splits data/processed_avspeech/val/combined_landmarks.csv into
val_split.csv and test_split.csv, exactly like scripts/splits/split_val_test.py
does for UniTalk-ASD's own val data — same stratified-by-track logic
(bucketed on detection-quality x speaking-majority, VAL_SPLIT_RATIO,
VAL_TEST_SPLIT_SEED), reused unmodified via split_val_test.main()'s path
overrides.

Every AVSpeech track is 100% SPEAKING_AUDIBLE by construction (see
scripts/dataset/download_avspeech_subset.py's module docstring), so the
"speaking-majority" stratification bucket here is trivially all-True —
harmless, the split still balances on detection-quality, just not on a
speaking-ratio axis that has no variance in this source.

Usage:
    python scripts/splits/split_avspeech_val_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import AVSPEECH_PROCESSED_DATASET_DIR, COMBINED_LANDMARKS_FILENAME, TEST_SPLIT_FILENAME, VAL_SPLIT_FILENAME
from split_val_test import main as split_main

if __name__ == "__main__":
    avspeech_val_dir = AVSPEECH_PROCESSED_DATASET_DIR / "val"
    split_main(
        source_csv=avspeech_val_dir / COMBINED_LANDMARKS_FILENAME,
        val_out=avspeech_val_dir / VAL_SPLIT_FILENAME,
        test_out=avspeech_val_dir / TEST_SPLIT_FILENAME,
    )
