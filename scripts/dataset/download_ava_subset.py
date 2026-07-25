"""Downloads a subset (config.AVA_SUBSET_HOURS — None means every available
video, no cap) of AVA-ActiveSpeaker and reshapes it into the exact data/raw/
layout UniTalk-ASD already uses:
    data/raw_ava/clips_videos/<split>/<video_id>/<entity_id>/<frame_timestamp:.2f>.jpg
    data/raw_ava/csv/<split>_orig.csv
so scripts/features/process_dataset.py's DatasetProcessor can run over it unmodified,
just pointed at config.AVA_RAW_CLIPS_VIDEOS_DIR / config.AVA_RAW_CSV_DIR.

AVA ships only an annotation CSV (video_id, frame_timestamp, entity_box_x1/
y1/x2/y2, label, entity_id) and full-length movie files hosted on S3 — no
pre-cropped face images like UniTalk-ASD provides. This script:
  1. Downloads + parses both official annotation CSVs (train/val — 120 + 33
     videos, the full public release; commonly-cited larger figures include
     a held-out test set Google never released annotations for).
  2. Greedily selects videos (fixed-seed shuffled order, so a partial subset
     is a random spread rather than always the alphabetically-first movies)
     until the target annotated-hours budget (config.AVA_SUBSET_HOURS, split
     train/val via config.AVA_SUBSET_VAL_FRACTION) is reached per split, or
     selects every video when config.AVA_SUBSET_HOURS is None.
  3. Per selected video, concurrently (config.NUM_AVA_DOWNLOAD_WORKERS
     threads — each video's ffmpeg subprocess is independent, so this
     parallelizes well): seeks into the S3-hosted movie with ffmpeg (avoiding
     a full-movie download — ffmpeg seeks via HTTP range requests) and
     decodes only that video's own annotated span, resampled to a constant
     25fps (config.DATASET_FPS) so every annotated frame_timestamp maps to a
     predictable decoded-frame index; crops every (entity_id,
     frame_timestamp) row's human-annotated face box out of the matching
     decoded frame and writes it to the expected path. No face/head detector
     needed — the box is already ground truth.
  4. Writes data/raw_ava/csv/<split>_orig.csv in the same 10-column shape
     config.CSV_COLUMNS defines, deriving label_id via config.AVA_LABEL_ID_MAP
     and instance_id = entity_id (nothing downstream reads instance_id's
     actual value — confirmed against process_dataset.py's _load_labels,
     which only touches video_id/entity_id/frame_timestamp/label/label_id).
     Rows are written incrementally as each video finishes, so a crash
     partway through a long run still leaves a valid, usable partial CSV +
     partial set of cropped images rather than losing everything.

No audio is downloaded — this project never trains on it.

Usage:
    python scripts/dataset/download_ava_subset.py
    python scripts/dataset/download_ava_subset.py --splits train --limit 3   # smoke test
"""
from __future__ import annotations

import argparse
import bz2
import csv
import random
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (
    AVA_ANNOTATIONS_URL_TRAIN,
    AVA_ANNOTATIONS_URL_VAL,
    AVA_FILE_NAMES_URL,
    AVA_LABEL_ID_MAP,
    AVA_RAW_CLIPS_VIDEOS_DIR,
    AVA_RAW_CSV_DIR,
    AVA_SUBSET_HOURS,
    AVA_SUBSET_VAL_FRACTION,
    AVA_VIDEO_BASE_URL,
    CSV_COLUMNS,
    DATASET_FPS,
    NUM_AVA_DOWNLOAD_WORKERS,
)

# AVA's raw annotation CSVs ship the first 8 of our 10 CSV_COLUMNS, headerless.
AVA_RAW_CSV_COLUMNS = CSV_COLUMNS[:8]
_ANNOTATION_URLS = {"train": AVA_ANNOTATIONS_URL_TRAIN, "val": AVA_ANNOTATIONS_URL_VAL}
_SELECTION_SEED = 42


def _download(url: str, dest_path: Path) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if not dest_path.exists():
        urllib.request.urlretrieve(url, dest_path)
    return dest_path


def download_and_parse_annotations(split: str, tmp_dir: Path) -> pd.DataFrame:
    tar_path = _download(_ANNOTATION_URLS[split], tmp_dir / f"ava_activespeaker_{split}_v1.0.tar.bz2")
    extract_dir = tmp_dir / f"ava_{split}_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:bz2") as tar:
        tar.extractall(extract_dir)
    csv_paths = list(extract_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"no CSV found inside {tar_path}")
    df = pd.concat(
        (pd.read_csv(p, header=None, names=AVA_RAW_CSV_COLUMNS) for p in csv_paths), ignore_index=True
    )
    print(f"[{split}] parsed {len(csv_paths)} CSV file(s) inside the tarball, {len(df)} total rows, {df['video_id'].nunique()} unique videos")

    unknown_labels = sorted(set(df["label"].unique()) - set(AVA_LABEL_ID_MAP))
    if unknown_labels:
        raise ValueError(
            f"[{split}] unrecognized AVA label value(s) {unknown_labels} — "
            f"not in config.AVA_LABEL_ID_MAP {sorted(AVA_LABEL_ID_MAP)}. Update the map before proceeding."
        )
    return df


def download_file_name_map(tmp_dir: Path) -> dict[str, str]:
    """video_id (filename stem) -> exact S3 filename (with extension)."""
    list_path = _download(AVA_FILE_NAMES_URL, tmp_dir / "ava_speech_file_names_v1.txt")
    lines = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    return {Path(line).stem: line for line in lines}


def select_videos(
    df: pd.DataFrame, target_hours: float | None, seed: int = _SELECTION_SEED
) -> tuple[list[str], float]:
    """Greedily picks video_ids (fixed-seed shuffled order) until their
    combined annotated span (max frame_timestamp - min per video, a proxy
    for "how much video this contributes") reaches target_hours. Returns the
    selected ids and the realized total hours.

    target_hours=None selects every video in `df` (no cap) — the shuffle
    still runs for a consistent, reproducible processing order even though
    nothing is actually being excluded.
    """
    spans = df.groupby("video_id")["frame_timestamp"].agg(lambda s: s.max() - s.min())
    video_ids = list(spans.index)
    random.Random(seed).shuffle(video_ids)

    if target_hours is None:
        return video_ids, float(spans.sum()) / 3600

    selected: list[str] = []
    total_seconds = 0.0
    target_seconds = target_hours * 3600
    for video_id in video_ids:
        if total_seconds >= target_seconds:
            break
        selected.append(video_id)
        total_seconds += float(spans[video_id])
    return selected, total_seconds / 3600


def extract_frames(video_url: str, min_ts: float, max_ts: float, out_dir: Path) -> bool:
    """ffmpeg seeks (via HTTP range requests, no full-file download) to
    min_ts and decodes through max_ts, resampled to a constant DATASET_FPS
    so frame index i corresponds to time min_ts + i/DATASET_FPS. Returns
    False (and leaves out_dir possibly partially populated) on ffmpeg failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = max_ts - min_ts + 1.0  # +1s pad so the last annotated frame is covered
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{min_ts:.3f}", "-i", video_url, "-t", f"{duration:.3f}",
        "-r", str(DATASET_FPS), "-q:v", "2",
        str(out_dir / "f_%08d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    return result.returncode == 0


def crop_and_save(frame_path: Path, box_norm: tuple[float, float, float, float], out_path: Path) -> bool:
    if not frame_path.exists():
        return False
    with Image.open(frame_path) as img:
        w, h = img.size
        x1, y1, x2, y2 = (max(0.0, min(1.0, c)) for c in box_norm)
        box_px = (x1 * w, y1 * h, x2 * w, y2 * h)
        if box_px[2] <= box_px[0] or box_px[3] <= box_px[1]:
            return False
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.crop(box_px).save(out_path)
    return True


def process_video(
    video_id: str, group: pd.DataFrame, split: str, video_url: str, video_out_dir: Path
) -> tuple[list[list], int, int]:
    """Returns (csv_rows, n_cropped, n_missing_frame)."""
    min_ts, max_ts = float(group["frame_timestamp"].min()), float(group["frame_timestamp"].max())

    with tempfile.TemporaryDirectory(prefix=f"ava_frames_{video_id}_") as frames_dir:
        frames_dir = Path(frames_dir)
        if not extract_frames(video_url, min_ts, max_ts, frames_dir):
            return [], 0, len(group)

        rows, n_cropped, n_missing = [], 0, 0
        for row in group.itertuples(index=False):
            frame_idx = round((row.frame_timestamp - min_ts) * DATASET_FPS) + 1  # ffmpeg output is 1-indexed
            frame_path = frames_dir / f"f_{frame_idx:08d}.jpg"
            out_path = video_out_dir / video_id / row.entity_id / f"{row.frame_timestamp:.2f}.jpg"
            box = (row.entity_box_x1, row.entity_box_y1, row.entity_box_x2, row.entity_box_y2)
            if crop_and_save(frame_path, box, out_path):
                n_cropped += 1
                label_id = AVA_LABEL_ID_MAP[row.label]
                rows.append([
                    video_id, row.frame_timestamp, row.entity_box_x1, row.entity_box_y1,
                    row.entity_box_x2, row.entity_box_y2, row.label, row.entity_id,
                    label_id, row.entity_id,
                ])
            else:
                n_missing += 1
        return rows, n_cropped, n_missing


def process_split(split: str, tmp_dir: Path, limit: int | None) -> None:
    df = download_and_parse_annotations(split, tmp_dir)
    file_name_map = download_file_name_map(tmp_dir)

    target_hours = (
        None if AVA_SUBSET_HOURS is None
        else AVA_SUBSET_HOURS * (AVA_SUBSET_VAL_FRACTION if split == "val" else 1 - AVA_SUBSET_VAL_FRACTION)
    )
    selected_ids, realized_hours = select_videos(df, target_hours)
    if limit is not None:
        selected_ids = selected_ids[:limit]

    video_out_dir = AVA_RAW_CLIPS_VIDEOS_DIR / split
    n_cropped_total = n_missing_total = n_skipped_videos = n_rows_written = 0

    # Videos are processed concurrently (NUM_AVA_DOWNLOAD_WORKERS threads) —
    # each is an independent ffmpeg subprocess (CPU-bound decode, releases
    # the GIL while the subprocess runs) plus lightweight PIL cropping, so
    # this parallelizes well without the GPU-VRAM contention concern
    # NUM_EXTRACTION_WORKERS is capped for. Rows are written to the output
    # CSV incrementally as each video finishes (file opened fresh at the
    # start of this run, not appended across runs) so a crash partway
    # through a long full-dataset pull still leaves a valid, usable partial
    # CSV + partial set of cropped images, rather than losing everything —
    # a rerun after a crash reprocesses all selected videos from scratch
    # (parallelized, so cheap enough not to need finer-grained resumability).
    AVA_RAW_CSV_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AVA_RAW_CSV_DIR / f"{split}_orig.csv"
    with open(out_path, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        writer.writerow(CSV_COLUMNS)

        eligible_ids = [v for v in selected_ids if v in file_name_map]
        n_skipped_videos += len(selected_ids) - len(eligible_ids)

        # Grouped once here rather than re-filtering the full (multi-million-row)
        # dataframe per video inside the loop below.
        groups = {video_id: group for video_id, group in df.groupby("video_id") if video_id in eligible_ids}

        with ThreadPoolExecutor(max_workers=NUM_AVA_DOWNLOAD_WORKERS) as pool:
            futures = {
                pool.submit(
                    process_video, video_id, groups[video_id], split,
                    f"{AVA_VIDEO_BASE_URL}/{file_name_map[video_id]}", video_out_dir,
                ): video_id
                for video_id in eligible_ids
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"[{split}]"):
                video_id = futures[future]
                group_len = len(groups[video_id])
                rows, n_cropped, n_missing = future.result()
                if not rows and n_missing == group_len:
                    n_skipped_videos += 1
                writer.writerows(rows)
                out_f.flush()
                n_rows_written += len(rows)
                n_cropped_total += n_cropped
                n_missing_total += n_missing

    print(
        f"[{split}] {len(selected_ids)} videos selected (~{realized_hours:.2f}h annotated span, "
        f"target {'unlimited' if target_hours is None else f'{target_hours:.2f}h'}), "
        f"{n_skipped_videos} videos skipped entirely (missing filename or ffmpeg failure), "
        f"{n_cropped_total} frames cropped, {n_missing_total} frames missing (decode/crop failure) "
        f"-> wrote {n_rows_written} rows to {out_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download an AVA-ActiveSpeaker subset into data/raw_ava/")
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"])
    parser.add_argument("--limit", type=int, default=None, help="only process the first N selected videos per split (smoke test)")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="ava_dl_") as tmp:
        tmp_dir = Path(tmp)
        for split in args.splits:
            process_split(split, tmp_dir, args.limit)


if __name__ == "__main__":
    main()
