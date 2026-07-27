"""Downloads a subset (config.AVSPEECH_TARGET_HOURS) of AVSpeech (Google
Research, "Looking to Listen at the Cocktail Party," CC-BY-4.0) and reshapes
it into the data/raw/ layout UniTalk-ASD already uses:
    data/raw_avspeech/clips_videos/<split>/<video_id>/spk/<frame_timestamp:.2f>.jpg
    data/raw_avspeech/csv/<split>_orig.csv
so scripts/features/process_dataset.py's DatasetProcessor can run over it
unmodified, just pointed at config.AVSPEECH_RAW_CLIPS_VIDEOS_DIR /
config.AVSPEECH_RAW_CSV_DIR (see scripts/features/process_avspeech_dataset.py).

Added specifically to grow the SPEAKING class, the minority (~30%) across
the other four sources combined (see project memory). Every AVSpeech
segment is curated by its own authors so the visible person is the sole
active speaker for the segment's ENTIRE 3-10s duration — so every extracted
frame gets label_id=1 (SPEAKING_AUDIBLE) unconditionally. This source never
contributes NOT_SPEAKING rows, by design.

Unlike UniTalk-ASD/AVA/WASD/Columbia, AVSpeech does NOT ship a bounding box
at all — its CSV gives only (youtube_id, start_sec, end_sec, x_norm,
y_norm), a single face-CENTER hint at the segment's start frame. This is
the first source needing this project's own face detection instead of
cropping an already-provided box, via batch-face's RetinaFace (GPU-batched;
see config.RETINAFACE_GPU_ID's comment for why this library was chosen).

The documented official CSV link (storage.cloud.google.com/avspeech-files/
...) requires an authenticated Google account (confirmed via `curl -I`:
403 / login redirect) — not actually anonymously public despite being "the"
documented link. config.AVSPEECH_TRAIN_CSV_URL points at a verified working
anonymous Hugging Face mirror instead.

Source videos span ~290K arbitrary YouTube uploads with wildly varying
native frame rates (unlike WASD/Columbia's more predictable sources), so
every fetched subclip is resampled to a constant config.DATASET_FPS (25) via
ffmpeg — same technique download_ava_subset.py uses — rather than probing
each video's native fps. frame_timestamp = frame_index / DATASET_FPS is
then exact for every source video, no per-video fps handling needed.

Pipeline per selected source (YouTube) video:
  1. One yt-dlp invocation requests every selected segment from that video
     in a single call (repeated --download-sections) — same source-video
     batching download_wasd_subset.py uses, for the same reason (resolves
     the video's manifest once, not once per segment).
  2. Each fetched subclip is resampled to DATASET_FPS via ffmpeg into a
     sequence of JPGs.
  3. All of a clip's frames are decoded and run through RetinaFace in ONE
     batched GPU forward pass (batch-face's intended usage), not
     frame-by-frame.
  4. The target face is tracked frame-to-frame by nearest box-center
     distance (frame 0 anchored to the CSV's (x_norm, y_norm) hint instead,
     since RetinaFace has no built-in identity tracking and multiple people
     can appear in a clip even though only one is speaking). If no detected
     face in a frame is within config.RETINAFACE_MAX_TRACK_DISTANCE_PX of
     the previous frame's chosen center, tracking is considered lost and
     the rest of that clip is skipped — never silently jump to a different
     person's face.
  5. Writes data/raw_avspeech/csv/{train,val}_orig.csv in CSV_COLUMNS shape
     — label/label_id are always SPEAKING_AUDIBLE/1 (see module docstring).
     entity_id/instance_id are set to the clip's own video_id (not a shared
     constant like "spk") — downstream code (process_dataset.py's
     completed_ids resume manifest, split_val_test.py's track splitting,
     src/dataset.py's LandmarkSequenceDataset) all identify a track by
     person_id ALONE, with no video_id involved, because for every other
     source (UniTalk-ASD/AVA/WASD/Columbia) the entity/track id is already
     globally unique. AVSpeech has exactly one speaker per clip, so a
     shared constant here silently collapses every video into one giant
     fake "track" downstream instead — discovered via split_avspeech_val_test.py
     producing a single 1-track val split on 2026-07-27. Using video_id
     keeps person_id globally unique without touching that shared code.

No audio is downloaded — this project never trains on it.

Resumable and CUMULATIVE (added 2026-07-27): config.AVSPEECH_TARGET_HOURS
is the total desired selection, not an increment — since selection is a
fixed-seed shuffle + cumulative-duration cutoff, raising it and rerunning
naturally extends into new segments beyond whatever a previous run already
covered. Every attempted segment (success OR failure) is recorded in
config.AVSPEECH_ATTEMPTED_MANIFEST and skipped on rerun — failures are
skipped too (not just successes), since most are real, persistent YouTube
availability issues, not worth re-attempting. {train,val}_orig.csv are
merged with (not overwritten by) a previous run's rows.

Usage:
    python scripts/dataset/download_avspeech_subset.py
    python scripts/dataset/download_avspeech_subset.py --limit 3   # smoke test: 3 source videos
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from batch_face import RetinaFace
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (
    AVSPEECH_ATTEMPTED_MANIFEST,
    AVSPEECH_DATA_RAW_DIR,
    AVSPEECH_RAW_CLIPS_VIDEOS_DIR,
    AVSPEECH_RAW_CSV_DIR,
    AVSPEECH_STAGING_DIR,
    AVSPEECH_TARGET_HOURS,
    AVSPEECH_TRAIN_CSV_URL,
    AVSPEECH_VAL_FRACTION,
    CSV_COLUMNS,
    DATASET_FPS,
    NUM_AVSPEECH_DOWNLOAD_WORKERS,
    RETINAFACE_GPU_ID,
    RETINAFACE_MAX_TRACK_DISTANCE_PX,
    YOUTUBE_FORMAT,
    YOUTUBE_URL_TEMPLATE,
)

_SELECTION_SEED = 42
Segment = tuple[str, float, float, float, float]  # youtube_id, start, end, x_norm, y_norm


def download_csv(tmp_dir: Path) -> pd.DataFrame:
    csv_path = tmp_dir / "avspeech_train.csv"
    if not csv_path.exists():
        urllib.request.urlretrieve(AVSPEECH_TRAIN_CSV_URL, csv_path)
    df = pd.read_csv(csv_path, header=None, names=["youtube_id", "start", "end", "x_norm", "y_norm"])
    print(f"parsed {len(df)} total AVSpeech segments, {df['youtube_id'].nunique()} unique source videos")
    return df


def compute_realized_hours() -> float:
    """Actual usable hours currently persisted in {train,val}_orig.csv —
    the ground truth this script's auto-loop targets, as opposed to raw
    (attempted, pre-attrition) hours. See select_next_batch()'s docstring.
    """
    total_rows = 0
    for split in ("train", "val"):
        path = AVSPEECH_RAW_CSV_DIR / f"{split}_orig.csv"
        if path.exists():
            with open(path) as f:
                total_rows += sum(1 for _ in f) - 1  # header
    return total_rows / (DATASET_FPS * 3600)


def select_next_batch(df: pd.DataFrame, attempted_ids: set[str], raw_hours: float, seed: int = _SELECTION_SEED) -> pd.DataFrame:
    """Fixed-seed shuffled (same shuffle every call — deterministic, stable
    ordering across repeated calls, not re-randomized) greedy accumulation
    of NOT-YET-ATTEMPTED segments until their combined RAW duration reaches
    raw_hours. Returns an empty DataFrame once every segment in `df` has
    been attempted (the whole AVSpeech CSV is exhausted).

    "Raw" hours here is distinct from "realized" hours (compute_realized_hours())
    — real-world YouTube attrition (private/deleted videos, lost face
    tracking) means only a fraction of raw-selected duration survives to
    become actual cropped, labeled frames (observed ~46.7% on this source's
    first full run, 2026-07-26/27 — far lower than an early small-sample
    estimate suggested). main()'s loop selects successive raw batches sized
    from the OBSERVED realized/raw ratio so far, re-measured after each
    batch, until the REALIZED total actually reaches the target — instead
    of this project's original one-shot design, which picked a single raw
    batch and left it to the user to notice and manually top up the
    shortfall (which is exactly what happened the first time this source
    was added).
    """
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    video_ids = shuffled["youtube_id"] + "_" + shuffled["start"].map("{:.2f}".format) + "-" + shuffled["end"].map("{:.2f}".format)
    remaining = shuffled[~video_ids.isin(attempted_ids)]
    if remaining.empty:
        return remaining
    durations = (remaining["end"] - remaining["start"]).to_numpy()
    cum = np.cumsum(durations)
    target_seconds = raw_hours * 3600
    n = int(np.searchsorted(cum, target_seconds)) + 1
    return remaining.iloc[: min(n, len(remaining))].copy()


def assign_video_splits(video_ids: list[str], seed: int = _SELECTION_SEED) -> dict[str, str]:
    """Fixed-seed shuffle of source video_ids -> {video_id: "train"|"val"},
    config.AVSPEECH_VAL_FRACTION of them to "val" — this project's own
    train/val split, since AVSpeech ships no partition relevant to a
    selected subset (same rationale as config.COLUMBIA_NUM_VAL_SPEAKERS).
    """
    ids = list(video_ids)
    random.Random(seed).shuffle(ids)
    n_val = round(len(ids) * AVSPEECH_VAL_FRACTION)
    val_ids = set(ids[:n_val])
    return {vid: ("val" if vid in val_ids else "train") for vid in ids}


def load_attempted_ids() -> set[str]:
    """video_ids already attempted (success OR failure) in a previous run —
    see config.AVSPEECH_ATTEMPTED_MANIFEST's comment for why failures are
    included, not just successes.
    """
    if not AVSPEECH_ATTEMPTED_MANIFEST.exists():
        return set()
    return {line.strip() for line in AVSPEECH_ATTEMPTED_MANIFEST.read_text().splitlines() if line.strip()}


def load_existing_csv_rows(split: str) -> list[list]:
    """Rows already written by a previous run's {split}_orig.csv, so a
    rerun merges with (not overwrites) prior results. frame_timestamp
    (column 1) is parsed back to float so it sorts consistently against
    this run's freshly-computed float values.
    """
    path = AVSPEECH_RAW_CSV_DIR / f"{split}_orig.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        rows = []
        for row in reader:
            row[1] = float(row[1])
            rows.append(row)
        return rows


def fetch_subclips_for_source(source_id: str, specs: list[tuple[str, float, float]], out_dir: Path) -> dict[str, Path]:
    """specs: list of (video_id, start, end) all belonging to source_id. One
    yt-dlp call requests every section at once — same pattern as
    download_wasd_subset.py's function of the same name.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["yt-dlp", "--quiet", "--no-warnings"]
    for _video_id, start, end in specs:
        cmd += ["--download-sections", f"*{start:.3f}-{end:.3f}"]
    cmd += [
        "-f", YOUTUBE_FORMAT,
        "-o", str(out_dir / f"{source_id}_%(section_start)s-%(section_end)s.%(ext)s"),
        YOUTUBE_URL_TEMPLATE.format(video_id=source_id),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        pass  # fall through — map whatever sections did finish before the timeout

    # yt-dlp's %(section_start)s/%(section_end)s use Python's default
    # float-to-str formatting, which strips trailing zeros (30.200 -> "30.2")
    # — verified directly (2026-07-26), so the exact output filename can't be
    # predicted by re-formatting our own requested start/end. Instead, parse
    # the actual produced filenames back into numbers and match numerically.
    produced: list[tuple[float, float, Path]] = []
    prefix = f"{source_id}_"
    for path in out_dir.glob(f"{source_id}_*.*"):
        remainder = path.stem[len(prefix):] if path.stem.startswith(prefix) else None
        if remainder is None or "-" not in remainder:
            continue
        start_str, end_str = remainder.split("-", 1)
        try:
            produced.append((float(start_str), float(end_str), path))
        except ValueError:
            continue

    result = {}
    for video_id, start, end in specs:
        match = next((p for s, e, p in produced if abs(s - start) < 0.01 and abs(e - end) < 0.01), None)
        if match is not None:
            result[video_id] = match
    return result


def resample_to_dataset_fps(subclip_path: Path, frames_dir: Path) -> bool:
    """ffmpeg-resamples subclip_path to a constant DATASET_FPS, writing
    frame_%08d.jpg into frames_dir. Returns False on ffmpeg failure.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(subclip_path),
        "-r", str(DATASET_FPS), "-q:v", "2",
        str(frames_dir / "f_%08d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.returncode == 0


def track_target_face(
    detector: RetinaFace, frame_paths: list[Path], hint_x_norm: float, hint_y_norm: float
) -> dict[int, tuple[float, float, float, float]]:
    """Returns {frame_index: (x1,y1,x2,y2)} for the tracked target face,
    only for frames where tracking succeeded — see module docstring for the
    nearest-center tracking + max-distance-loss algorithm.
    """
    images = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in frame_paths]
    all_faces = detector(images, threshold=0.9)

    tracked: dict[int, tuple[float, float, float, float]] = {}
    prev_center: tuple[float, float] | None = None
    for idx, faces in enumerate(all_faces):
        if not faces:
            break  # no face at all this frame — treat as lost, stop (never skip-and-resume)

        if prev_center is None:
            h, w = images[idx].shape[:2]
            target = (hint_x_norm * w, hint_y_norm * h)
        else:
            target = prev_center

        def center_of(box) -> tuple[float, float]:
            x1, y1, x2, y2 = box
            return ((x1 + x2) / 2, (y1 + y2) / 2)

        best_box, best_dist = None, float("inf")
        for box, _kps, _score in faces:
            cx, cy = center_of(box)
            dist = ((cx - target[0]) ** 2 + (cy - target[1]) ** 2) ** 0.5
            if dist < best_dist:
                best_box, best_dist = box, dist

        if prev_center is not None and best_dist > RETINAFACE_MAX_TRACK_DISTANCE_PX:
            break  # tracking lost — stop rather than risk jumping to a different person

        tracked[idx] = tuple(best_box)
        prev_center = center_of(best_box)

    return tracked


def process_source_video(
    source_id: str, entries: list[tuple[str, str, float, float, float, float]], detector: RetinaFace
) -> tuple[list[list], int, int]:
    """entries: list of (video_id, split, start, end, x_norm, y_norm) all
    belonging to source_id. Returns (csv_rows, n_cropped, n_failed).
    """
    subclips_dir = AVSPEECH_STAGING_DIR / "subclips"
    specs = [(video_id, start, end) for video_id, _split, start, end, _x, _y in entries]
    subclip_paths = fetch_subclips_for_source(source_id, specs, subclips_dir)

    csv_rows, n_cropped, n_failed = [], 0, 0
    for video_id, split, start, end, x_norm, y_norm in entries:
        subclip_path = subclip_paths.get(video_id)
        if subclip_path is None:
            n_failed += 1
            continue
        try:
            with tempfile.TemporaryDirectory(prefix=f"avspeech_frames_{video_id}_") as frames_dir:
                frames_dir = Path(frames_dir)
                if not resample_to_dataset_fps(subclip_path, frames_dir):
                    n_failed += 1
                    continue
                frame_paths = sorted(frames_dir.glob("f_*.jpg"))
                if not frame_paths:
                    n_failed += 1
                    continue

                tracked = track_target_face(detector, frame_paths, x_norm, y_norm)
                # video_id doubles as the entity/person id here — see module
                # docstring for why a shared constant like "spk" would collide
                # across every clip downstream instead.
                out_dir = AVSPEECH_RAW_CLIPS_VIDEOS_DIR / split / video_id / video_id
                for idx, box in tracked.items():
                    frame_timestamp = idx / DATASET_FPS
                    img = cv2.imread(str(frame_paths[idx]))
                    h, w = img.shape[:2]
                    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
                    x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
                    if x2 <= x1 or y2 <= y1:
                        n_failed += 1
                        continue
                    out_path = out_dir / f"{frame_timestamp:.2f}.jpg"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(out_path), img[y1:y2, x1:x2]):
                        n_failed += 1
                        continue
                    n_cropped += 1
                    csv_rows.append([
                        video_id, round(frame_timestamp, 2),
                        x1 / w, y1 / h, x2 / w, y2 / h,
                        "SPEAKING_AUDIBLE", video_id, 1, video_id,
                    ])
                n_failed += len(frame_paths) - len(tracked)
        finally:
            subclip_path.unlink(missing_ok=True)
    return csv_rows, n_cropped, n_failed


def process_batch(batch: pd.DataFrame, detector: RetinaFace, limit: int | None) -> tuple[int, int, int]:
    """Fetches, tracks, and crops every segment in `batch` (already filtered
    to not-yet-attempted by the caller's select_next_batch() call), merges
    results into {train,val}_orig.csv, and returns
    (n_source_videos_processed, n_cropped, n_failed).
    """
    splits = assign_video_splits(list(batch["youtube_id"].unique()))

    entries_by_source: dict[str, list] = defaultdict(list)
    for row in batch.itertuples(index=False):
        video_id = f"{row.youtube_id}_{row.start:.2f}-{row.end:.2f}"
        entries_by_source[row.youtube_id].append(
            (video_id, splits[row.youtube_id], row.start, row.end, row.x_norm, row.y_norm)
        )

    # Defensive double-check — select_next_batch() already excludes
    # attempted segments, but re-filtering here is cheap and guards against
    # any caller that passes an unfiltered batch directly.
    attempted = load_attempted_ids()
    for sid in list(entries_by_source):
        remaining = [e for e in entries_by_source[sid] if e[0] not in attempted]
        if remaining:
            entries_by_source[sid] = remaining
        else:
            del entries_by_source[sid]

    source_ids = list(entries_by_source)
    random.Random(_SELECTION_SEED).shuffle(source_ids)
    if limit is not None:
        source_ids = source_ids[:limit]

    rows_by_split: dict[str, list[list]] = {"train": [], "val": []}
    n_cropped_total = n_failed_total = 0
    # ThreadPoolExecutor here (not ProcessPoolExecutor): RetinaFace/CUDA
    # context isn't fork-safe across processes, and each worker's actual
    # bottleneck is the yt-dlp/ffmpeg subprocess (releases the GIL) with a
    # single shared GPU detector doing the batched inference — matches
    # download_wasd_subset.py's own I/O-bound threading rationale.
    with ThreadPoolExecutor(max_workers=NUM_AVSPEECH_DOWNLOAD_WORKERS) as pool, \
            open(AVSPEECH_ATTEMPTED_MANIFEST, "a") as manifest_file:
        futures = {
            pool.submit(process_source_video, sid, entries_by_source[sid], detector): sid for sid in source_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="source videos"):
            sid = futures[future]
            rows, n_cropped, n_failed = future.result()
            n_cropped_total += n_cropped
            n_failed_total += n_failed
            for row in rows:
                # row[0] is video_id "{youtube_id}_{start}-{end}"; recover youtube_id's split.
                # rsplit(maxsplit=1) is safe even if youtube_id itself contains
                # underscores — it only splits at the one we appended.
                youtube_id = row[0].rsplit("_", 1)[0]
                rows_by_split[splits[youtube_id]].append(row)
            # Mark every entry for this source video as attempted (success OR
            # failure) only now that it's actually finished — a crash before
            # this point just means a harmless re-attempt next run.
            for video_id, *_rest in entries_by_source[sid]:
                manifest_file.write(video_id + "\n")
            manifest_file.flush()
            os.fsync(manifest_file.fileno())

    AVSPEECH_RAW_CSV_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        rows = load_existing_csv_rows(split) + rows_by_split[split]
        rows.sort(key=lambda r: (r[0], r[1]))
        with open(AVSPEECH_RAW_CSV_DIR / f"{split}_orig.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
            writer.writerows(rows)
        rows_by_split[split] = rows  # for the summary line below, reflects the full merged total

    print(
        f"{len(source_ids)} new source videos processed, {n_cropped_total} frames cropped, "
        f"{n_failed_total} frames failed (tracking lost/decode/box error) -> "
        f"{len(rows_by_split['train'])} total train rows, {len(rows_by_split['val'])} total val rows "
        f"(including prior runs) in {AVSPEECH_RAW_CSV_DIR}"
    )
    return len(source_ids), n_cropped_total, n_failed_total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download an AVSpeech subset into data/raw_avspeech/ — "
        "auto-loops, selecting progressively more raw segments, until the "
        "REALIZED (usable, post-attrition) hours target is actually met."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="smoke test: process a single small batch capped at N source videos, then stop (ignores --hours)",
    )
    parser.add_argument(
        "--hours", type=float, default=AVSPEECH_TARGET_HOURS,
        help=f"target REALIZED hours desired in {AVSPEECH_RAW_CSV_DIR} "
        f"(default: config.AVSPEECH_TARGET_HOURS={AVSPEECH_TARGET_HOURS})",
    )
    parser.add_argument(
        "--batch-hours", type=float, default=30.0,
        help="raw hours to attempt per loop iteration before re-measuring the realized/raw "
        "success rate and sizing the next batch",
    )
    args = parser.parse_args()

    realized = compute_realized_hours()
    print(f"currently realized: {realized:.2f}h, target: {args.hours:.2f}h")
    if realized >= args.hours and args.limit is None:
        print("target already met — nothing to do")
        return

    AVSPEECH_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    AVSPEECH_DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    detector = RetinaFace(gpu_id=RETINAFACE_GPU_ID)

    with tempfile.TemporaryDirectory(prefix="avspeech_dl_") as tmp:
        df = download_csv(Path(tmp))
        avg_duration_h = float((df["end"] - df["start"]).mean()) / 3600

        # Seed the realized/raw success-rate estimate from this project's
        # own real history (attempted manifest vs. already-realized hours)
        # rather than an optimistic guess — an early small-sample estimate
        # on this source suggested ~84.2%, but the true rate observed once
        # a full run actually completed was ~46.7%. Falls back to 50% only
        # when there's no history yet (manifest empty, first run ever).
        attempted = load_attempted_ids()
        raw_attempted_hours_est = len(attempted) * avg_duration_h
        success_rate = max(0.05, realized / raw_attempted_hours_est) if raw_attempted_hours_est > 0 else 0.5

        iteration = 0
        while True:
            realized = compute_realized_hours()
            if args.limit is None:
                remaining_needed = args.hours - realized
                if remaining_needed <= 0:
                    print(f"target reached: {realized:.2f}h realized >= {args.hours:.2f}h target")
                    break
                batch_raw_hours = min(args.batch_hours, (remaining_needed / success_rate) * 1.15)
            else:
                batch_raw_hours = 5.0  # small candidate pool; --limit below caps actual work

            attempted = load_attempted_ids()
            batch = select_next_batch(df, attempted, batch_raw_hours)
            if batch.empty:
                print(f"AVSpeech source data exhausted — stopped at {realized:.2f}h realized, short of {args.hours:.2f}h target")
                break

            iteration += 1
            batch_raw_h = float((batch["end"] - batch["start"]).sum()) / 3600
            print(
                f"[batch {iteration}] {len(batch)} segments (~{batch_raw_h:.2f}h raw, success-rate estimate "
                f"{success_rate:.1%}) — realized so far {realized:.2f}h / {args.hours:.2f}h target"
            )
            process_batch(batch, detector, args.limit)

            new_realized = compute_realized_hours()
            batch_realized_h = new_realized - realized
            success_rate = max(0.05, batch_realized_h / batch_raw_h)  # re-measured, sizes the next batch
            print(
                f"[batch {iteration}] realized {batch_realized_h:.2f}h from {batch_raw_h:.2f}h raw "
                f"({success_rate:.1%} success) — total now {new_realized:.2f}h"
            )

            if args.limit is not None:
                break  # smoke test — one batch only

    if AVSPEECH_STAGING_DIR.exists():
        shutil.rmtree(AVSPEECH_STAGING_DIR)
        print(f"removed staging dir {AVSPEECH_STAGING_DIR}")


if __name__ == "__main__":
    main()
