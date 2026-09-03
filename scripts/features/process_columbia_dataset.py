"""Runs the same landmark extraction as scripts/features/process_dataset.py,
but over the Columbia ASD subset (scripts/dataset/download_columbia_subset.py's
output) instead of UniTalk-ASD's data/raw/.

Reuses DatasetProcessor unmodified (see its clips_videos_dir/raw_csv_dir
override params in process_dataset.py) — same parallel extraction, resume
manifest, and CSV-join logic, just pointed at
config.COLUMBIA_RAW_CLIPS_VIDEOS_DIR / config.COLUMBIA_RAW_CSV_DIR, writing to
config.COLUMBIA_PROCESSED_DATASET_DIR instead of the UniTalk-ASD processed/
directory.

Iterates over both config.SPLITS ("train", "val"), same as
process_ava_dataset.py/process_wasd_dataset.py — download_columbia_subset.py
splits Columbia's 6 speakers whole between train/val (see its module
docstring), so both splits have real data here, just far less of it than
AVA/WASD/UniTalk (only 6 discrete speaker tracks total).

Output layout (config.COLUMBIA_PROCESSED_DATASET_DIR — data/processed_columbia/):
    data/processed_columbia/<split>/combined_landmarks.csv
    data/processed_columbia/<split>/completed_ids.txt

Usage:
    python scripts/features/process_columbia_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import COLUMBIA_PROCESSED_DATASET_DIR, COLUMBIA_RAW_CLIPS_VIDEOS_DIR, COLUMBIA_RAW_CSV_DIR, SPLITS
from process_dataset import DatasetProcessor

if __name__ == "__main__":
    extractor_kwargs = dict(normalize=True, save_annotated=False, save_json=False)
    processor = DatasetProcessor(
        extractor_kwargs,
        landmarks_dir=COLUMBIA_PROCESSED_DATASET_DIR,
        clips_videos_dir=COLUMBIA_RAW_CLIPS_VIDEOS_DIR,
        raw_csv_dir=COLUMBIA_RAW_CSV_DIR,
    )
    for split in SPLITS:
        processor.process_split(split)
