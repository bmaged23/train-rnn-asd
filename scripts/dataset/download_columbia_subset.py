"""Downloads the Columbia Active Speaker Detection dataset (Chakravarty &
Tuytelaars, ACM ICMI 2016) and reshapes it into the data/raw/ layout
UniTalk-ASD already uses:
    data/raw_columbia/clips_videos/<split>/<video_id>/<entity_id>/<frame_timestamp:.2f>.jpg
    data/raw_columbia/csv/{train,val}_orig.csv
so scripts/features/process_dataset.py's DatasetProcessor can run over it
unmodified, just pointed at config.COLUMBIA_RAW_CLIPS_VIDEOS_DIR /
config.COLUMBIA_RAW_CSV_DIR (see scripts/features/process_columbia_dataset.py).

Small, single-video dataset — one 87-minute panel discussion ("Wikileaks and
Academia - a Panel Discussion at Columbia SIPA", youtu.be/6GzxbrO0DHM) with 6
named speakers manually labeled speak/non-speak per frame, ~149K total
labeled frames (far less than AVA/WASD's millions across hundreds of
videos). Each speaker is one single, long, continuous track — a much
coarser unit than AVA/WASD/UniTalk's many short per-scene tracks — so the 6
speakers are split whole between "train" and "val" (config.COLUMBIA_NUM_VAL_SPEAKERS
go to "val", fixed-seed shuffled, the rest to "train"), same track-level,
no-leakage principle used everywhere else in this project, just applied at
a much smaller N. "val" then further divides into val_split/test_split via
scripts/splits/split_columbia_val_test.py, exactly like AVA/WASD.

No official public dataset page could be found for Columbia despite
extensive search (2026-07-25) — the video + label-archive pair used here is
the exact one github.com/Junhua-Liao/LR-ASD's Columbia_test.py already
relies on for its own benchmark evaluation, both independently verified
working before writing this script: the YouTube video is public, and the
Drive-hosted col_labels.tar.gz (985KB) downloads and extracts cleanly via
gdown, containing genuine ground-truth files crediting the original paper.

Ground truth ships very differently from AVA/WASD's already-normalized [0,1]
box coordinates: per-speaker txt files (framenum, TLx, TLy, square-box-size,
speak-flag 0/1) in PIXEL coordinates at the video's native frame rate. This
script converts both — frame_timestamp = framenum / native_fps and box
coordinates normalized by width/height — using the actual downloaded
video's probed fps/resolution, not assumed values (no reason this source
video happens to match config.DATASET_FPS=25).

Pipeline:
  1. Download the single YouTube video (same avc1-pinned YOUTUBE_FORMAT
     WASD uses, for the same cv2-decodability reason — see config.py).
  2. Download + extract the Drive-hosted label archive.
  3. Probe the downloaded video's actual fps/width/height via cv2.
  4. Per speaker (config.COLUMBIA_SPEAKER_NAMES), parse fusion/<name>.txt
     (already the complete per-speaker ground truth, merged across every
     tracks_*/ frame-range subfolder — confirmed by comparing row counts),
     sort by frame number (the raw file isn't sorted — frame ranges are
     concatenated in directory-glob order, not chronological — sorting
     first keeps cv2 seeks monotonic/forward-only, avoiding the repeated
     backward-seek decode cost non-monotonic access triggers), then crop
     each row's normalized box out of that frame.
  5. Writes data/raw_columbia/csv/{train,val}_orig.csv in CSV_COLUMNS shape,
     deriving label_id directly from the speak-flag column (already binary)
     and instance_id = entity_id (same convention download_ava_subset.py
     uses — nothing downstream reads instance_id's actual value).

No audio is downloaded — this project never trains on it.

Usage:
    python scripts/dataset/download_columbia_subset.py
    python scripts/dataset/download_columbia_subset.py --limit 1   # smoke test: 1 speaker
"""
from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import gdown
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (
    COLUMBIA_LABEL_SCALE_FACTOR,
    COLUMBIA_LABELS_GDRIVE_ID,
    COLUMBIA_NUM_VAL_SPEAKERS,
    COLUMBIA_RAW_CLIPS_VIDEOS_DIR,
    COLUMBIA_RAW_CSV_DIR,
    COLUMBIA_SPEAKER_NAMES,
    COLUMBIA_VIDEO_YOUTUBE_ID,
    CSV_COLUMNS,
    NUM_COLUMBIA_CROP_WORKERS,
    YOUTUBE_FORMAT,
    YOUTUBE_URL_TEMPLATE,
)

Row = tuple[int, int, int, int, int]  # framenum, TLx, TLy, size, speak_flag
_SELECTION_SEED = 42  # same fixed-seed convention as download_ava_subset.py/download_wasd_subset.py


def assign_speaker_splits() -> dict[str, str]:
    """Fixed-seed shuffle of the 6 speakers -> {speaker: "train"|"val"},
    config.COLUMBIA_NUM_VAL_SPEAKERS of them to "val", the rest to "train" —
    see module docstring for why whole speakers move together.
    """
    speakers = list(COLUMBIA_SPEAKER_NAMES)
    random.Random(_SELECTION_SEED).shuffle(speakers)
    val_speakers = set(speakers[:COLUMBIA_NUM_VAL_SPEAKERS])
    return {speaker: ("val" if speaker in val_speakers else "train") for speaker in COLUMBIA_SPEAKER_NAMES}


def download_video(tmp_dir: Path) -> Path:
    video_path = tmp_dir / "columbia.mp4"
    if not video_path.exists():
        cmd = [
            "yt-dlp", "--quiet", "--no-warnings",
            "-f", YOUTUBE_FORMAT,
            "-o", str(video_path),
            YOUTUBE_URL_TEMPLATE.format(video_id=COLUMBIA_VIDEO_YOUTUBE_ID),
        ]
        subprocess.run(cmd, check=True, timeout=3600)
    return video_path


def download_labels(tmp_dir: Path) -> Path:
    """Returns the extracted col_labels/ directory (containing fusion/<speaker>.txt)."""
    archive_path = tmp_dir / "col_labels.tar.gz"
    extract_dir = tmp_dir / "col_labels_extracted"
    if not (extract_dir / "col_labels" / "fusion").exists():
        gdown.download(id=COLUMBIA_LABELS_GDRIVE_ID, output=str(archive_path), quiet=False)
        with tarfile.open(archive_path) as tar:
            tar.extractall(extract_dir, filter="data")
    return extract_dir / "col_labels"


def parse_speaker_rows(labels_dir: Path, speaker: str) -> list[Row]:
    path = labels_dir / "fusion" / f"{speaker}.txt"
    rows = [
        (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
        for line in path.read_text().splitlines()
        if (parts := line.split())
    ]
    rows.sort(key=lambda r: r[0])  # not sorted on disk — see module docstring
    return rows


def crop_speaker_frames(
    video_path: Path, fps: float, width: int, height: int, speaker: str, rows: list[Row], out_dir: Path
) -> tuple[list[list], int, int]:
    """Returns (csv_rows, n_cropped, n_failed)."""
    cap = cv2.VideoCapture(str(video_path))
    csv_rows, n_cropped, n_failed = [], 0, 0
    try:
        for framenum, tlx, tly, size, speak in rows:
            # Raw label coordinates are half-scale relative to this video's
            # actual resolution — see config.COLUMBIA_LABEL_SCALE_FACTOR.
            tlx, tly, size = (v * COLUMBIA_LABEL_SCALE_FACTOR for v in (tlx, tly, size))
            frame_timestamp = framenum / fps
            out_path = out_dir / COLUMBIA_VIDEO_YOUTUBE_ID / speaker / f"{frame_timestamp:.2f}.jpg"
            if out_path.exists():
                n_cropped += 1
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
                ok, frame = cap.read()
                if not ok or frame is None:
                    n_failed += 1
                    continue
                h, w = frame.shape[:2]
                x1, y1 = max(0, tlx), max(0, tly)
                x2, y2 = min(w, tlx + size), min(h, tly + size)
                if x2 <= x1 or y2 <= y1:
                    n_failed += 1
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(out_path), frame[y1:y2, x1:x2]):
                    n_failed += 1
                    continue
                n_cropped += 1

            x1_norm, y1_norm = max(0.0, min(1.0, tlx / width)), max(0.0, min(1.0, tly / height))
            x2_norm, y2_norm = max(0.0, min(1.0, (tlx + size) / width)), max(0.0, min(1.0, (tly + size) / height))
            label = "SPEAKING_AUDIBLE" if speak == 1 else "NOT_SPEAKING"
            csv_rows.append([
                COLUMBIA_VIDEO_YOUTUBE_ID, round(frame_timestamp, 2), x1_norm, y1_norm, x2_norm, y2_norm,
                label, speaker, speak, speaker,
            ])
    finally:
        cap.release()
    return csv_rows, n_cropped, n_failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Columbia ASD dataset into data/raw_columbia/")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only process the first N (of 6) speakers — overrides nothing in config, just a smoke test",
    )
    args = parser.parse_args()
    speakers = COLUMBIA_SPEAKER_NAMES[: args.limit] if args.limit is not None else COLUMBIA_SPEAKER_NAMES
    speaker_splits = assign_speaker_splits()

    with tempfile.TemporaryDirectory(prefix="columbia_dl_") as tmp:
        tmp_dir = Path(tmp)
        print("downloading video...")
        video_path = download_video(tmp_dir)
        print("downloading labels...")
        labels_dir = download_labels(tmp_dir)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        print(f"video: fps={fps:.3f} width={width} height={height}")

        speaker_rows = {speaker: parse_speaker_rows(labels_dir, speaker) for speaker in speakers}

        n_cropped_total = n_failed_total = 0
        rows_by_split: dict[str, list[list]] = {"train": [], "val": []}
        with ThreadPoolExecutor(max_workers=NUM_COLUMBIA_CROP_WORKERS) as pool:
            futures = {
                pool.submit(
                    crop_speaker_frames, video_path, fps, width, height, speaker, rows,
                    COLUMBIA_RAW_CLIPS_VIDEOS_DIR / speaker_splits[speaker],
                ): speaker
                for speaker, rows in speaker_rows.items()
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="speakers"):
                speaker = futures[future]
                rows, n_cropped, n_failed = future.result()
                rows_by_split[speaker_splits[speaker]].extend(rows)
                n_cropped_total += n_cropped
                n_failed_total += n_failed

    COLUMBIA_RAW_CSV_DIR.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        rows.sort(key=lambda r: (r[7], r[1]))  # entity_id, frame_timestamp — stable, readable ordering
        with open(COLUMBIA_RAW_CSV_DIR / f"{split}_orig.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
            writer.writerows(rows)

    processed_splits = [speaker_splits[speaker] for speaker in speakers]
    print(
        f"{len(speakers)} speakers processed ({processed_splits.count('train')} train / "
        f"{processed_splits.count('val')} val), {n_cropped_total} frames cropped, "
        f"{n_failed_total} frames failed (decode/box error) -> wrote "
        f"{len(rows_by_split['train'])} train rows, {len(rows_by_split['val'])} val rows to {COLUMBIA_RAW_CSV_DIR}"
    )


if __name__ == "__main__":
    main()
