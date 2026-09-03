"""Runs the same landmark extraction as scripts/features/process_dataset.py, but over
the AVA-ActiveSpeaker subset (scripts/dataset/download_ava_subset.py's output) instead
of UniTalk-ASD's data/raw/.

Reuses DatasetProcessor unmodified (see its clips_videos_dir/raw_csv_dir
override params in process_dataset.py) — same parallel extraction, resume
manifest, and CSV-join logic, just pointed at config.AVA_RAW_CLIPS_VIDEOS_DIR /
config.AVA_RAW_CSV_DIR, writing to config.AVA_PROCESSED_DATASET_DIR instead of
the UniTalk-ASD processed/ directory.

Output layout (config.AVA_PROCESSED_DATASET_DIR — data/processed_ava/):
    data/processed_ava/<split>/combined_landmarks.csv
    data/processed_ava/<split>/completed_ids.txt

Usage:
    python scripts/features/process_ava_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import AVA_PROCESSED_DATASET_DIR, AVA_RAW_CLIPS_VIDEOS_DIR, AVA_RAW_CSV_DIR, SPLITS
from process_dataset import DatasetProcessor

if __name__ == "__main__":
    extractor_kwargs = dict(normalize=True, save_annotated=False, save_json=False)
    processor = DatasetProcessor(
        extractor_kwargs,
        landmarks_dir=AVA_PROCESSED_DATASET_DIR,
        clips_videos_dir=AVA_RAW_CLIPS_VIDEOS_DIR,
        raw_csv_dir=AVA_RAW_CSV_DIR,
    )
    for split in SPLITS:
        processor.process_split(split)
