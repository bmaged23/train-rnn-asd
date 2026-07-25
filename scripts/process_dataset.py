"""
Batch-process every person folder in a split (train/val), in parallel.

No per-person files are written — PersonLandmarkExtractor runs with
save_json=False, save_annotated=False, so extraction is purely in-memory per
person. Every frame from every person is joined with its CSV label and
appended as one row to a single combined CSV for the split — a flat numeric
table (one column per landmark coordinate), directly loadable for training
with no parsing step, and meaningfully smaller on disk than JSON for this much
dense numeric data (no repeated key names / bracket punctuation per frame).

Extraction runs across NUM_EXTRACTION_WORKERS worker processes (config.py), each
building its own PersonLandmarkExtractor using MEDIAPIPE_DELEGATE (config.py —
see that constant's comment for the CPU-vs-GPU benchmark on this server: a small
GPU-delegate worker pool measured faster in aggregate than a full-core CPU pool,
so that's the default; NUM_EXTRACTION_WORKERS is sized accordingly and should be
raised back toward the core count if MEDIAPIPE_DELEGATE is switched to "CPU").
Joining with labels and writing the combined CSV stays single-threaded in the
main process (fast, I/O-only), so there's no concurrent-write risk.

Resumability is NOT based on scanning the CSV (a crash mid-row-write for one
person would leave a partial, but individually well-formed, set of rows that a
"person_id already appears in the CSV" check couldn't distinguish from a
complete one). Instead, a person_id is appended to a separate completed_ids.txt
manifest only after that person's CSV rows are flushed + fsynced — the manifest
is the actual source of truth for what's done.

Output layout (PROCESSED_DATASET_DIR, config.py — data/processed/):
    data/processed/<split>/combined_landmarks.csv   (one row per frame)
    data/processed/<split>/completed_ids.txt          (resume manifest, one person_id per line)

CSV columns: person_id, video_id, image_name, detected, mar, target, target_index,
face_landmark_0_x, face_landmark_0_y, face_landmark_0_z, ... (NUM_FACE_LANDMARKS).
mouth_landmark_* is NOT stored — it's a fixed subset of face_landmark_* (config's
MOUTH_LANDMARK_INDICES), reconstruct it at load time instead of duplicating it on
disk. target/target_index come from the CSV's label/label_id (null if no matching
CSV row was found for that frame); landmark/mar values are formatted to
CSV_FLOAT_DECIMALS (config.py) rather than full float precision — landmark columns
are empty when detected=False.

Usage:
    python scripts/process_dataset.py
"""
from __future__ import annotations

import csv
import fcntl
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    COMBINED_LANDMARKS_FILENAME,
    COMPLETED_IDS_FILENAME,
    CSV_FLOAT_DECIMALS,
    NUM_EXTRACTION_WORKERS,
    NUM_FACE_LANDMARKS,
    PROCESSED_DATASET_DIR,
    RAW_CLIPS_VIDEOS_DIR,
    RAW_CSV_DIR,
    SPLITS,
)
from extract_landmarks import PersonLandmarkExtractor

# mouth_landmark_* columns are deliberately NOT stored — they're a fixed subset
# of face_landmark_* (see config.MOUTH_LANDMARK_INDICES) and would just duplicate
# data already in the row; reconstruct them at load time instead.
FACE_LANDMARK_COLUMNS = [f"face_landmark_{i}_{axis}" for i in range(NUM_FACE_LANDMARKS) for axis in "xyz"]
CSV_HEADER = (
    ["person_id", "video_id", "image_name", "detected", "mar", "target", "target_index"]
    + FACE_LANDMARK_COLUMNS
)

# Built once per worker process via _init_worker, reused for every person that
# worker handles — a fresh PersonLandmarkExtractor (and its FaceLandmarker) is
# expensive to construct, so we don't want to pay that cost per person.
_worker_extractor: PersonLandmarkExtractor | None = None


def _init_worker(extractor_kwargs: dict) -> None:
    global _worker_extractor
    _worker_extractor = PersonLandmarkExtractor(**extractor_kwargs)


def _extract_worker(person_dir: Path, person_id: str) -> dict:
    return _worker_extractor.process_person(person_dir, person_id=person_id)


class DatasetProcessor:
    """Drives PersonLandmarkExtractor (in parallel) over every person folder in
    a split and writes one combined, label-joined CSV per split.
    """

    def __init__(
        self,
        extractor_kwargs: dict,
        num_workers: int = NUM_EXTRACTION_WORKERS,
        landmarks_dir: Path = PROCESSED_DATASET_DIR,
        clips_videos_dir: Path = RAW_CLIPS_VIDEOS_DIR,
        raw_csv_dir: Path = RAW_CSV_DIR,
    ):
        self.extractor_kwargs = extractor_kwargs
        self.num_workers = num_workers
        self.landmarks_dir = Path(landmarks_dir)
        # Overridable so the same class can process a different raw/ layout
        # (e.g. data/raw_ava/) without any other logic changes — see
        # scripts/process_ava_dataset.py.
        self.clips_videos_dir = Path(clips_videos_dir)
        self.raw_csv_dir = Path(raw_csv_dir)

    # ---- public API ----

    def process_split(self, split: str) -> None:
        clips_split_dir = self.clips_videos_dir / split
        split_output_dir = self.landmarks_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = split_output_dir / COMBINED_LANDMARKS_FILENAME
        manifest_path = split_output_dir / COMPLETED_IDS_FILENAME

        # Guard against two invocations processing the same split at once — with
        # only one of them holding this lock, a second concurrent run fails fast
        # instead of racing writes against the same combined CSV / manifest.
        lock_file = open(split_output_dir / ".lock", "w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                f"[{split}] another process_dataset.py run already holds the lock for "
                f"this split ({split_output_dir}) — refusing to run two extractions "
                f"against the same output folder concurrently."
            )

        try:
            self._process_split_locked(split, clips_split_dir, csv_path, manifest_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()

    def _process_split_locked(
        self, split: str, clips_split_dir: Path, csv_path: Path, manifest_path: Path
    ) -> None:
        completed_ids = self._load_completed_ids(manifest_path)
        person_dirs = sorted(clips_split_dir.glob("*/*"))
        to_extract = [p for p in person_dirs if p.name not in completed_ids]
        # Only needed by combine_and_write, which only runs if to_extract is
        # non-empty — skips a multi-second pandas read+index of the labels CSV
        # on a no-op rerun of an already-fully-processed split.
        labels = self._load_labels(split, self.raw_csv_dir) if to_extract else {}

        write_header = not csv_path.exists()
        unmatched_frames = 0
        with open(csv_path, "a", newline="") as csv_file, open(manifest_path, "a") as manifest_file:
            writer = csv.writer(csv_file)
            if write_header:
                writer.writerow(CSV_HEADER)
                csv_file.flush()

            def combine_and_write(person_dir: Path, record: dict) -> None:
                nonlocal unmatched_frames
                person_id, video_id = person_dir.name, person_dir.parent.name
                assert record["person_id"] == person_id, (
                    f"record/folder mismatch: got {record['person_id']!r} for folder {person_id!r} "
                    "— would corrupt the combined CSV if written"
                )

                entity_labels = labels.get(video_id, {}).get(person_id, {})
                n_unmatched = 0
                rows = []
                for image in record["images"]:
                    ts_key = Path(image["image_name"]).stem
                    label, label_id = entity_labels.get(ts_key, (None, None))
                    if label is None:
                        n_unmatched += 1
                    rows.append(self._image_to_row(person_id, video_id, image, label, label_id))
                unmatched_frames += n_unmatched

                writer.writerows(rows)
                csv_file.flush()
                os.fsync(csv_file.fileno())

                # Only mark this person done AFTER their rows are durably on disk —
                # if a crash happens between the two writes, at worst this person
                # gets re-extracted and duplicated on the next run, never silently
                # lost or half-written.
                manifest_file.write(person_id + "\n")
                manifest_file.flush()
                os.fsync(manifest_file.fileno())
                completed_ids.add(person_id)

            if to_extract:
                with ProcessPoolExecutor(
                    max_workers=self.num_workers,
                    initializer=_init_worker,
                    initargs=(self.extractor_kwargs,),
                ) as pool:
                    futures = {pool.submit(_extract_worker, person_dir, person_dir.name): person_dir for person_dir in to_extract}
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"[{split}] extracting"):
                        combine_and_write(futures[future], future.result())

        print(
            f"[{split}] {len(person_dirs)} people "
            f"({len(person_dirs) - len(to_extract)} already done, {len(to_extract)} extracted), "
            f"{unmatched_frames} frames with no CSV label match"
        )

    # ---- internals ----

    @staticmethod
    def _fmt_float(value: float | None) -> str:
        return "" if value is None else f"{value:.{CSV_FLOAT_DECIMALS}f}"

    @classmethod
    def _image_to_row(cls, person_id: str, video_id: str, image: dict, label: str | None, label_id: int | None) -> list:
        row = [
            person_id, video_id, image["image_name"], image["detected"],
            cls._fmt_float(image["mar"]), label, label_id,
        ]
        face = image["face_landmarks"] or [[None, None, None]] * NUM_FACE_LANDMARKS
        for point in face:
            row.extend(cls._fmt_float(c) for c in point)
        return row

    @staticmethod
    def _load_labels(split: str, raw_csv_dir: Path) -> dict[str, dict[str, dict[str, tuple[str, int]]]]:
        df = pd.read_csv(raw_csv_dir / f"{split}_orig.csv")
        ts_keys = df["frame_timestamp"].map(lambda t: f"{t:.2f}")

        labels: dict[str, dict[str, dict[str, tuple[str, int]]]] = {}
        for video_id, entity_id, ts_key, label, label_id in zip(
            df["video_id"], df["entity_id"], ts_keys, df["label"], df["label_id"]
        ):
            labels.setdefault(video_id, {}).setdefault(entity_id, {})[ts_key] = (label, int(label_id))
        return labels

    @staticmethod
    def _load_completed_ids(manifest_path: Path) -> set[str]:
        if not manifest_path.exists():
            return set()
        with open(manifest_path) as f:
            return {line.strip() for line in f if line.strip()}


if __name__ == "__main__":
    extractor_kwargs = dict(normalize=True, save_annotated=False, save_json=False)
    processor = DatasetProcessor(extractor_kwargs)
    for split in SPLITS:
        processor.process_split(split)
