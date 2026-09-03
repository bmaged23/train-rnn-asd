"""Rebuild AVSpeech CSV rows for clips that are on disk but missing from
{train,val}_orig.csv.

WHY THIS EXISTS
---------------
download_avspeech_subset.py's process_batch() accumulates CSV rows in memory and
writes {train,val}_orig.csv only once, at the very END of a batch. The attempted-
manifest, by contrast, is flushed and fsync'd after every source video. So if a
run is interrupted mid-batch:

  * the downloaded frames are safely on disk,
  * their ids ARE recorded as attempted (so a rerun skips them),
  * but their CSV rows die with the process.

The result is orphaned clips: real footage no CSV references, which the
feature-extraction stage therefore ignores entirely. On 2026-09-01 that was
13,165 clips / ~2.0M frames / ~22 hours of footage.

WHAT IT RECONSTRUCTS
--------------------
process_dataset.py::_load_labels() joins frames to labels on exactly five
columns - video_id, entity_id, frame_timestamp (formatted "%.2f"), label,
label_id. The four entity_box_* columns are NEVER read for AVSpeech, because its
frames are already cropped on disk by the RetinaFace tracker; they are written
empty here so the column count still matches CSV_COLUMNS.

Everything needed is recoverable from the directory layout:

    clips_videos/{split}/{video_id}/{entity_id}/{frame_timestamp}.jpg

plus AVSpeech's defining property - every track is SPEAKING_AUDIBLE by
construction - so label/label_id are constant.

Usage:
    python scripts/dataset/recover_avspeech_csv_rows.py            # report only
    python scripts/dataset/recover_avspeech_csv_rows.py --write    # merge them in
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (  # noqa: E402
    AVSPEECH_RAW_CLIPS_VIDEOS_DIR,
    AVSPEECH_RAW_CSV_DIR,
    CSV_COLUMNS,
    DATASET_FPS,
    SPLITS,
)

LABEL = "SPEAKING_AUDIBLE"
LABEL_ID = 1


def existing_rows(split: str) -> list[list]:
    path = AVSPEECH_RAW_CSV_DIR / f"{split}_orig.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        return [r for r in reader if r]


def already_keyed(rows: list[list]) -> set:
    """(video_id, entity_id, ts_key) already present - the same triple
    _load_labels() joins on, so a duplicate is never written."""
    keyed = set()
    for r in rows:
        try:
            keyed.add((r[0], r[7], f"{float(r[1]):.2f}"))
        except (IndexError, ValueError):
            continue
    return keyed


def scan_split(split: str, keyed: set):
    """Walk clips_videos/{split} and build rows for frames not already keyed."""
    base = AVSPEECH_RAW_CLIPS_VIDEOS_DIR / split
    rows: list[list] = []
    clips_seen = clips_recovered = 0
    if not base.is_dir():
        return rows, clips_seen, clips_recovered

    for video_dir in sorted(base.iterdir()):
        if not video_dir.is_dir():
            continue
        for entity_dir in sorted(video_dir.iterdir()):
            if not entity_dir.is_dir():
                continue
            clips_seen += 1
            added_here = 0
            for frame in entity_dir.glob("*.jpg"):
                try:
                    ts = float(frame.stem)
                except ValueError:
                    continue  # not a timestamp-named frame
                ts_key = f"{ts:.2f}"
                if (video_dir.name, entity_dir.name, ts_key) in keyed:
                    continue
                rows.append([
                    video_dir.name, ts_key,
                    "", "", "", "",           # entity_box_* - unused for AVSpeech
                    LABEL, entity_dir.name, LABEL_ID, video_dir.name,
                ])
                added_here += 1
            if added_here:
                clips_recovered += 1
    return rows, clips_seen, clips_recovered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="merge recovered rows into {train,val}_orig.csv (default: report only)")
    args = ap.parse_args()

    grand_new = 0
    for split in SPLITS:
        old = existing_rows(split)
        keyed = already_keyed(old)
        new_rows, clips_seen, clips_recovered = scan_split(split, keyed)
        grand_new += len(new_rows)

        print(f"[{split}] {clips_seen:,} clips on disk, {len(old):,} rows already in CSV")
        print(f"[{split}] {clips_recovered:,} clips had unreferenced frames "
              f"-> {len(new_rows):,} rows recoverable (~{len(new_rows)/DATASET_FPS/3600:.2f} h)")

        if args.write and new_rows:
            merged = old + new_rows
            merged.sort(key=lambda r: (r[0], float(r[1])))
            path = AVSPEECH_RAW_CSV_DIR / f"{split}_orig.csv"
            backup = path.with_suffix(".csv.pre-recovery")
            if path.exists() and not backup.exists():
                backup.write_bytes(path.read_bytes())
                print(f"[{split}] backed up -> {backup.name}")
            AVSPEECH_RAW_CSV_DIR.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(CSV_COLUMNS)
                w.writerows(merged)
            print(f"[{split}] WROTE {len(merged):,} rows ({len(new_rows):,} recovered)")

    print()
    print(f"TOTAL recoverable: {grand_new:,} rows (~{grand_new/DATASET_FPS/3600:.2f} h)")
    if not args.write:
        print("dry run - rerun with --write to merge them in")


if __name__ == "__main__":
    main()
