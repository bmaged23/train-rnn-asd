"""Downloads SpeakingFaces' silent-session (Session 1) subset and reshapes
it into the data/raw/ layout UniTalk-ASD already uses:
    data/raw_speakingfaces/clips_videos/<split>/<video_id>/<person_id>/<frame_timestamp:.2f>.png
    data/raw_speakingfaces/csv/{train,val}_orig.csv
so scripts/features/process_dataset.py's DatasetProcessor can run over it
unmodified, just pointed at config.SPEAKINGFACES_RAW_CLIPS_VIDEOS_DIR /
config.SPEAKINGFACES_RAW_CSV_DIR (see scripts/features/process_speakingfaces_dataset.py).

Added to grow the NOT_SPEAKING class — the opposite gap from AVSpeech (which
is 100% SPEAKING by construction). SpeakingFaces (ISSAI, "SpeakingFaces: A
Large-Scale Multimodal Dataset of Voice Commands with Visual and Thermal
Video Streams", CC-BY-4.0) is a lab-recorded dataset: 142 subjects, each
doing 2 trials of 2 sessions across 9 fixed camera angles — Session 1
subjects are silent and still, Session 2 reads voice commands off a screen.

Only Session 1 is used here. Session 2's public release
(image_audio/sub_{N}_ia.zip) was investigated and rejected (2026-07-28
pilot): it only ships ~4-6 sparse, non-consecutive frames per spoken
command (verified via a frame-index gap-check — e.g. one command's kept
frames were [74, 601, 905, 1029, 1055, 1208], nowhere close to
consecutive), not continuous video, so it can't support this project's
50-frame windowed model. Session 1, by contrast, was verified fully
continuous — 900/900 frames present with zero gaps for every camera
position checked — so every frame here gets label_id=0 (NOT_SPEAKING)
unconditionally, the mirror image of AVSpeech's constant-label approach.

Hosted directly on HuggingFace as one zip per subject (image_only/sub_{N}
_io.zip) — no YouTube re-fetch, no throttling risk, unlike WASD/AVSpeech.
sub_2_io.zip is a known-corrupt 0-byte file on the source repo itself
(confirmed via the HF API's own file-size metadata) and is skipped
automatically (also simply absent from metadata/subjects.csv, so nothing
special has to detect it).

Each subject's zip nests 3 image "modalities" per position under
trial_{T}/rgb_image_aligned/ (thermal/raw/aligned-rgb, filename's last field
== config.SPEAKINGFACES_ALIGNED_RGB_MODALITY for the visual-thermal-aligned
stream this script extracts) plus duplicate copies elsewhere in the zip
that are ignored entirely. Frames are natively 28fps; resampled via
nearest-frame selection onto a 25fps grid (config.DATASET_FPS) rather than
left at native rate — see config.py's SPEAKINGFACES_NATIVE_FPS comment for
why this deviates from download_columbia_subset.py's native-fps exception
(a 50-frame window must mean the same 2 real seconds across every combined
source). PNGs are kept as-is, not re-encoded to JPG (config.IMAGE_EXTENSIONS
already accepts them, and process_dataset.py never assumes .jpg) — avoids a
wasted lossy recompression pass over ~2.3M images.

person_id = f"sub{{N}}_trial{{T}}_pos{{P}}" — a globally unique key, even
though the on-disk layout nests position under a f"sub{{N}}_trial{{T}}"
video_id folder. Learned from the AVSpeech integration bug: any shared
per-video person_id constant collides across every downstream consumer
that groups by person_id alone (process_dataset.py's completed_ids resume
manifest, split_val_test.py's track splitting, src/dataset.py's
LandmarkSequenceDataset) — never repeat that mistake.

Train/"val" partitioning uses SpeakingFaces' own official per-subject
Train/Valid/Test column (metadata/subjects.csv) instead of inventing a new
shuffle — Valid+Test subjects both fold into this project's raw "val" pool,
which scripts/splits/split_speakingfaces_val_test.py then re-splits into
val_split/test_split the same stratified way as every other source.

Resumable: a subject is only ever appended to
config.SPEAKINGFACES_COMPLETED_SUBJECTS_MANIFEST once every one of its
frames is durably written, so a rerun just skips already-done subjects
(same append-only-after-durable-write pattern process_dataset.py itself
uses, and the opposite of the download-side attempted_video_ids.txt bug
that briefly corrupted AVSpeech's own manifest).

Usage:
    python scripts/dataset/download_speakingfaces_subset.py
    python scripts/dataset/download_speakingfaces_subset.py --limit 3   # smoke test: 3 subjects
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (
    CSV_COLUMNS,
    DATASET_FPS,
    SPEAKINGFACES_ALIGNED_RGB_MODALITY,
    SPEAKINGFACES_COMPLETED_SUBJECTS_MANIFEST,
    SPEAKINGFACES_DATA_RAW_DIR,
    SPEAKINGFACES_FRAMES_PER_POSITION,
    SPEAKINGFACES_HF_REPO_ID,
    SPEAKINGFACES_NATIVE_FPS,
    SPEAKINGFACES_NUM_POSITIONS,
    SPEAKINGFACES_NUM_TRIALS,
    SPEAKINGFACES_RAW_CLIPS_VIDEOS_DIR,
    SPEAKINGFACES_RAW_CSV_DIR,
    SPEAKINGFACES_STAGING_DIR,
)

_NOT_SPEAKING_LABEL = "NOT_SPEAKING"
_NOT_SPEAKING_LABEL_ID = 0


def load_subject_splits() -> dict[int, str]:
    """{subject_id: "train"|"val"} from the dataset's own official Split
    column — "Train" -> "train", "Valid"/"Test" both -> "val" (this
    project's raw "val" pool, later re-split by
    split_speakingfaces_val_test.py). Subject 2 is simply absent from this
    file (matches its known-corrupt 0-byte zip), so nothing special has to
    filter it out.
    """
    path = hf_hub_download(repo_id=SPEAKINGFACES_HF_REPO_ID, repo_type="dataset", filename="metadata/subjects.csv")
    df = pd.read_csv(path)
    return {
        int(row.Sub_ID): ("train" if row.Split == "Train" else "val")
        for row in df.itertuples(index=False)
    }


def resample_frame_indices() -> list[int]:
    """Nearest-frame selection mapping config.SPEAKINGFACES_FRAMES_PER_POSITION
    native-rate frames (1-indexed) onto a config.DATASET_FPS grid — e.g. 900
    frames over 32.14s at 28fps resamples to ~804 frames at 25fps. Returns
    the ORIGINAL (native) 1-indexed frame numbers to keep, one per output
    tick, in order (duplicates possible only if DATASET_FPS > native fps,
    not the case here).
    """
    duration_s = SPEAKINGFACES_FRAMES_PER_POSITION / SPEAKINGFACES_NATIVE_FPS
    n_out = round(duration_s * DATASET_FPS)
    out_ticks_s = np.arange(n_out) / DATASET_FPS
    native_frame_times_s = (np.arange(SPEAKINGFACES_FRAMES_PER_POSITION) + 1) / SPEAKINGFACES_NATIVE_FPS
    nearest_idx = np.searchsorted(native_frame_times_s, out_ticks_s)
    nearest_idx = np.clip(nearest_idx, 0, SPEAKINGFACES_FRAMES_PER_POSITION - 1)
    return [int(i) + 1 for i in nearest_idx]  # back to 1-indexed native frame numbers


def load_completed_subjects() -> set[int]:
    if not SPEAKINGFACES_COMPLETED_SUBJECTS_MANIFEST.exists():
        return set()
    return {int(line.strip()) for line in SPEAKINGFACES_COMPLETED_SUBJECTS_MANIFEST.read_text().splitlines() if line.strip()}


def process_subject(sub_id: int, split: str, resample_map: list[int]) -> tuple[list[list], int, int]:
    """Downloads, extracts, and reshapes one subject's silent-session frames.
    Returns (csv_rows, n_written, n_missing).
    """
    zip_path = Path(
        hf_hub_download(
            repo_id=SPEAKINGFACES_HF_REPO_ID,
            repo_type="dataset",
            filename=f"image_only/sub_{sub_id}_io.zip",
            local_dir=SPEAKINGFACES_STAGING_DIR,
        )
    )
    csv_rows: list[list] = []
    n_written = n_missing = 0
    try:
        try:
            zf = zipfile.ZipFile(zip_path)
        except zipfile.BadZipFile:
            # Known issue: at least sub_2_io.zip is a 0-byte corrupt file on
            # the source repo itself (confirmed via the HF API's own
            # file-size metadata) — treated as a permanent, non-retryable
            # failure for this subject, not a crash. Any other subject
            # hitting this is logged the same way rather than assumed.
            print(f"[sub{sub_id}] BadZipFile — source zip is corrupt, skipping this subject entirely")
            return csv_rows, n_written, n_missing
        with zf as z:
            for trial in range(1, SPEAKINGFACES_NUM_TRIALS + 1):
                video_id = f"sub{sub_id}_trial{trial}"
                for pos in range(1, SPEAKINGFACES_NUM_POSITIONS + 1):
                    person_id = f"{video_id}_pos{pos}"
                    out_dir = SPEAKINGFACES_RAW_CLIPS_VIDEOS_DIR / split / video_id / person_id
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for out_frame_idx, native_frame in enumerate(resample_map):
                        member = (
                            f"sub_{sub_id}_io/trial_{trial}/rgb_image_aligned/"
                            f"{sub_id}_{trial}_1_{pos}_{native_frame}_{SPEAKINGFACES_ALIGNED_RGB_MODALITY}.png"
                        )
                        frame_timestamp = out_frame_idx / DATASET_FPS
                        out_path = out_dir / f"{frame_timestamp:.2f}.png"
                        if not out_path.exists():
                            # Only reached again for a subject an earlier,
                            # interrupted run never finished (never-completed
                            # subjects are the only ones reprocessed) — still
                            # need the CSV row below even when the file itself
                            # already exists, or this frame's metadata is
                            # silently lost.
                            try:
                                data = z.read(member)
                            except KeyError:
                                n_missing += 1
                                continue
                            out_path.write_bytes(data)
                        n_written += 1
                        csv_rows.append([
                            video_id, round(frame_timestamp, 2),
                            0.0, 0.0, 1.0, 1.0,  # full, uncropped frame — see module docstring
                            _NOT_SPEAKING_LABEL, person_id, _NOT_SPEAKING_LABEL_ID, person_id,
                        ])
    finally:
        zip_path.unlink(missing_ok=True)
    return csv_rows, n_written, n_missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download SpeakingFaces' silent-session subset into data/raw_speakingfaces/"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only process the first N subjects (of the ~141 usable) — smoke test",
    )
    args = parser.parse_args()

    SPEAKINGFACES_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    SPEAKINGFACES_RAW_CSV_DIR.mkdir(parents=True, exist_ok=True)

    subject_splits = load_subject_splits()
    subject_ids = sorted(subject_splits)
    if args.limit is not None:
        subject_ids = subject_ids[: args.limit]

    completed = load_completed_subjects()
    resample_map = resample_frame_indices()
    print(f"resampling {SPEAKINGFACES_FRAMES_PER_POSITION} frames/position -> {len(resample_map)} frames/position (28fps -> {DATASET_FPS}fps)")

    n_written_total = n_missing_total = n_skipped = 0
    n_rows_total = {"train": 0, "val": 0}

    # Both CSVs opened in append mode for the whole run and a row written
    # per-subject IMMEDIATELY (flushed+fsynced) before that subject is
    # marked complete below — never batched till the end. A batch-at-the-
    # end design would lose already-durable subjects' rows entirely on a
    # mid-run crash, since the completed-subjects manifest (which a rerun
    # trusts to skip already-done work) would already list them.
    csv_files, csv_writers = {}, {}
    for split in ("train", "val"):
        csv_path = SPEAKINGFACES_RAW_CSV_DIR / f"{split}_orig.csv"
        write_header = not csv_path.exists()
        csv_files[split] = open(csv_path, "a", newline="")
        csv_writers[split] = csv.writer(csv_files[split])
        if write_header:
            csv_writers[split].writerow(CSV_COLUMNS)
            csv_files[split].flush()

    try:
        with open(SPEAKINGFACES_COMPLETED_SUBJECTS_MANIFEST, "a") as manifest_file:
            for sub_id in tqdm(subject_ids, desc="subjects"):
                if sub_id in completed:
                    n_skipped += 1
                    continue
                split = subject_splits[sub_id]
                rows, n_written, n_missing = process_subject(sub_id, split, resample_map)

                csv_writers[split].writerows(rows)
                csv_files[split].flush()
                os.fsync(csv_files[split].fileno())
                n_rows_total[split] += len(rows)
                n_written_total += n_written
                n_missing_total += n_missing

                # Only mark done AFTER this subject's rows are durably on
                # disk — a crash mid-subject just means a harmless re-attempt.
                manifest_file.write(f"{sub_id}\n")
                manifest_file.flush()
    finally:
        for f in csv_files.values():
            f.close()

    if SPEAKINGFACES_STAGING_DIR.exists():
        shutil.rmtree(SPEAKINGFACES_STAGING_DIR)

    print(
        f"{len(subject_ids) - n_skipped} subjects processed ({n_skipped} already done), "
        f"{n_written_total} frames written, {n_missing_total} frames missing from source zip -> "
        f"{n_rows_total['train']} new train rows, {n_rows_total['val']} new val rows"
    )


if __name__ == "__main__":
    main()
