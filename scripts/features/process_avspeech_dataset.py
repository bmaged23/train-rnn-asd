"""Runs the same landmark extraction as scripts/features/process_dataset.py,
but over the AVSpeech subset (scripts/dataset/download_avspeech_subset.py's
output) instead of UniTalk-ASD's data/raw/.

Reuses DatasetProcessor unmodified (see its clips_videos_dir/raw_csv_dir
override params in process_dataset.py) — same parallel extraction, resume
manifest, and CSV-join logic, just pointed at
config.AVSPEECH_RAW_CLIPS_VIDEOS_DIR / config.AVSPEECH_RAW_CSV_DIR, writing
to config.AVSPEECH_PROCESSED_DATASET_DIR instead of the UniTalk-ASD
processed/ directory.

Output layout (config.AVSPEECH_PROCESSED_DATASET_DIR — data/processed_avspeech/):
    data/processed_avspeech/<split>/combined_landmarks.csv
    data/processed_avspeech/<split>/completed_ids.txt

Usage:
    python scripts/features/process_avspeech_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import AVSPEECH_PROCESSED_DATASET_DIR, AVSPEECH_RAW_CLIPS_VIDEOS_DIR, AVSPEECH_RAW_CSV_DIR, SPLITS
from process_dataset import DatasetProcessor

if __name__ == "__main__":
    extractor_kwargs = dict(normalize=True, save_annotated=False, save_json=False)
    processor = DatasetProcessor(
        extractor_kwargs,
        landmarks_dir=AVSPEECH_PROCESSED_DATASET_DIR,
        clips_videos_dir=AVSPEECH_RAW_CLIPS_VIDEOS_DIR,
        raw_csv_dir=AVSPEECH_RAW_CSV_DIR,
    )
    for split in SPLITS:
        processor.process_split(split)
