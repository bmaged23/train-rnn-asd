"""PyTorch Dataset/DataLoader over process_dataset.py's combined_landmarks.csv.

One sample = one person_id track (a single face/entity within one video, the
same unit scripts/features/process_dataset.py extracts per-person). Each track is a
variable-length sequence of per-frame feature vectors, a per-frame validity
mask, and per-frame SPEAKING_AUDIBLE/NOT_SPEAKING labels. The feature vector
is (landmarks, optionally + MAR) — which landmark set and whether MAR is
included are both config.py toggles (config.USE_MOUTH_LANDMARKS_ONLY,
config.USE_MAR), applied identically across train/val/test so a saved
checkpoint's input_size always matches what get_dataloader() produces.

Two distinct masking decisions, both driven by MIN_DETECTED_FRAMES_RATIO
(config.py):
  - Track-level: a track whose overall detected-frame ratio falls below the
    threshold is dropped from the dataset index entirely — not deleted from
    disk, just never surfaced by LandmarkSequenceDataset.
  - Frame-level: a track that clears the threshold still keeps every frame
    (to preserve real temporal spacing), but any individual undetected frame
    within it has its feature vector zero-filled and is flagged 0 in `mask`.
    Batch padding (variable-length tracks -> a fixed-length batch tensor)
    reuses the same `mask` convention, so downstream code only has to check
    one tensor to know which timesteps are real, detected input.

There are three logical splits — "train", "val", "test". "train" reads
data/processed/train/combined_landmarks.csv directly; "val" and "test" read
data/processed/val/{val_split,test_split}.csv, the stratified split of the
raw "val" folder produced by scripts/splits/split_val_test.py (run that first — see
its docstring for how the split is balanced).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    AVA_PROCESSED_DATASET_DIR,
    AVSPEECH_PROCESSED_DATASET_DIR,
    BALANCED_BATCHES,
    BATCH_SIZE,
    COLUMBIA_PROCESSED_DATASET_DIR,
    COMBINED_LANDMARKS_FILENAME,
    DATA_PROCESSED_DIR,
    MIN_DETECTED_FRAMES_RATIO,
    MOUTH_LANDMARK_INDICES,
    NUM_FACE_LANDMARKS,
    TEST_SPLIT_FILENAME,
    USE_MAR,
    USE_MOUTH_LANDMARKS_ONLY,
    VAL_SPLIT_FILENAME,
    WASD_PROCESSED_DATASET_DIR,
)

# config.USE_MOUTH_LANDMARKS_ONLY/USE_MAR are a single source of truth applied
# identically to train/val/test — not per-call overrides, since the model's
# input_size is derived from NUM_INPUT_FEATURES below and a checkpoint
# trained with one combination won't load into a model built with another.
_LANDMARK_INDICES = MOUTH_LANDMARK_INDICES if USE_MOUTH_LANDMARKS_ONLY else range(NUM_FACE_LANDMARKS)
LANDMARK_FEATURE_COLUMNS = [f"face_landmark_{i}_{axis}" for i in _LANDMARK_INDICES for axis in "xyz"]
FEATURE_COLUMNS = LANDMARK_FEATURE_COLUMNS + (["mar"] if USE_MAR else [])
NUM_INPUT_FEATURES = len(FEATURE_COLUMNS)

_METADATA_COLUMNS = ["person_id", "image_name", "detected", "target_index"]

# Logical splits the DataLoader exposes, distinct from config.SPLITS (the raw
# HuggingFace/data-on-disk splits, "train"/"val" — see module docstring).
LOGICAL_SPLITS = ("train", "val", "test")


def split_csv_path(processed_dir: Path, split: str) -> Path:
    """Same train/val/test -> filename convention scripts/splits/split_val_test.py
    and process_dataset.py already establish, generalized over the processed
    dir root so a second dataset sharing this exact schema (e.g. AVA-
    ActiveSpeaker, rooted at config.AVA_PROCESSED_DATASET_DIR — see
    ava_source() below) can reuse it instead of duplicating the mapping.
    """
    if split == "train":
        return processed_dir / "train" / COMBINED_LANDMARKS_FILENAME
    if split == "val":
        return processed_dir / "val" / VAL_SPLIT_FILENAME
    return processed_dir / "val" / TEST_SPLIT_FILENAME  # split == "test"


_SPLIT_CSV_PATH = {split: split_csv_path(DATA_PROCESSED_DIR, split) for split in LOGICAL_SPLITS}


def ava_source(split: str) -> tuple[Path, str]:
    """One `extra_sources` entry (see LandmarkSequenceDataset/get_dataloader
    and their windowed counterparts) that points at the AVA-ActiveSpeaker
    subset (scripts/dataset/download_ava_subset.py + scripts/features/process_ava_dataset.py's
    output) for the given logical split — same combined_landmarks.csv schema
    as UniTalk-ASD, just rooted at config.AVA_PROCESSED_DATASET_DIR. Shared by
    both the per-frame and windowed pipelines (src/windowed_dataset.py
    re-exports this rather than duplicating it) since it's pure path
    resolution with no windowing-specific logic.
    """
    return split_csv_path(AVA_PROCESSED_DATASET_DIR, split), "ava"


def wasd_source(split: str) -> tuple[Path, str]:
    """One `extra_sources` entry (see LandmarkSequenceDataset/get_dataloader
    and their windowed counterparts) that points at the WASD (Wilder Active
    Speaker Detection) subset (scripts/dataset/download_wasd_subset.py +
    scripts/features/process_wasd_dataset.py's output) for the given logical split —
    same combined_landmarks.csv schema as UniTalk-ASD/AVA, just rooted at
    config.WASD_PROCESSED_DATASET_DIR. Shared by both the per-frame and
    windowed pipelines the same way ava_source() is.
    """
    return split_csv_path(WASD_PROCESSED_DATASET_DIR, split), "wasd"


def columbia_source(split: str) -> tuple[Path, str]:
    """One `extra_sources` entry (see LandmarkSequenceDataset/get_dataloader
    and their windowed counterparts) that points at the Columbia Active
    Speaker Detection subset (scripts/dataset/download_columbia_subset.py +
    scripts/features/process_columbia_dataset.py's output) for the given
    logical split — same combined_landmarks.csv schema as the other three
    sources, just rooted at config.COLUMBIA_PROCESSED_DATASET_DIR.

    Same train/val/test shape as ava_source()/wasd_source(), just far
    smaller — Columbia has only 6 discrete speaker tracks total, split
    config.COLUMBIA_NUM_VAL_SPEAKERS/rest between val/train (see
    download_columbia_subset.py's module docstring), so every logical split
    has some real data, just much less of it than the other three sources.
    """
    return split_csv_path(COLUMBIA_PROCESSED_DATASET_DIR, split), "columbia"


def avspeech_source(split: str) -> tuple[Path, str]:
    """One `extra_sources` entry (see LandmarkSequenceDataset/get_dataloader
    and their windowed counterparts) that points at the AVSpeech subset
    (scripts/dataset/download_avspeech_subset.py + scripts/features/process_avspeech_dataset.py's
    output) for the given logical split — same combined_landmarks.csv schema
    as the other four sources, just rooted at config.AVSPEECH_PROCESSED_DATASET_DIR.

    Every AVSpeech track is 100% SPEAKING_AUDIBLE by construction (see
    download_avspeech_subset.py's module docstring) — added specifically to
    grow the SPEAKING class, the minority across the other four sources
    combined, so it never contributes NOT_SPEAKING rows.
    """
    return split_csv_path(AVSPEECH_PROCESSED_DATASET_DIR, split), "avspeech"


def _frame_timestamp(image_name: str) -> float:
    # image_name stems are the frame_timestamp CSV field formatted as "%.2f"
    # (see process_dataset.py) — parse back to float for chronological sort.
    return float(Path(image_name).stem)


def _load_frame_table(split: str, csv_path: Path | None = None) -> pd.DataFrame:
    """csv_path overrides the default UniTalk-ASD path for `split` — used to
    load a second dataset (e.g. AVA-ActiveSpeaker, see
    src/combined_dataset.py) sharing the same combined_landmarks.csv schema.
    """
    if csv_path is None:
        csv_path = _SPLIT_CSV_PATH[split]
    if not csv_path.exists():
        how = (
            "run scripts/features/process_dataset.py first"
            if split == "train"
            else "run scripts/splits/split_val_test.py first"
        )
        raise FileNotFoundError(f"{csv_path} not found — {how}")
    df = pd.read_csv(csv_path, usecols=_METADATA_COLUMNS + FEATURE_COLUMNS, engine="pyarrow")
    df["frame_ts"] = df["image_name"].map(_frame_timestamp)
    # stable sort so within-person_id rows land in chronological order
    df.sort_values(["person_id", "frame_ts"], inplace=True, kind="mergesort")
    return df


class LandmarkSequenceDataset(Dataset):
    """Loads one split's combined_landmarks.csv fully into memory (only
    FEATURE_COLUMNS + metadata, not the full file — a few GB with the default
    mouth-only + MAR config; substantially more with
    config.USE_MOUTH_LANDMARKS_ONLY=False, since that's ~11.8x more landmark
    columns — see notebooks/03_explore_processed_dataset.ipynb for the full
    file sizes) and exposes one variable-length track per index.
    """

    def __init__(
        self,
        split: str,
        min_detected_ratio: float = MIN_DETECTED_FRAMES_RATIO,
        csv_path: Path | None = None,
        source_label: str | None = None,
        extra_sources: list[tuple[Path, str]] = (),
    ):
        """extra_sources: additional (csv_path, source_label) pairs — each is
        loaded the same way as the primary source and appended into this same
        dataset's person_ids/features/masks/labels, for training on multiple
        datasets combined (see ava_source() above). Empty by default —
        existing callers (scripts/modeling/train/frames.py, scripts/modeling/evaluate/frames.py) are
        unaffected unless they opt in.
        """
        if split not in LOGICAL_SPLITS:
            raise ValueError(f"split must be one of {LOGICAL_SPLITS}, got {split!r}")
        self.split = split
        self.min_detected_ratio = min_detected_ratio
        self.source_label = source_label or split

        self.person_ids: list[str] = []
        self.features: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []

        self._load_source(split, csv_path, self.source_label, min_detected_ratio)
        for extra_csv_path, extra_label in extra_sources:
            self._load_source(split, extra_csv_path, extra_label, min_detected_ratio)

    def _load_source(
        self, split: str, csv_path: Path | None, source_label: str, min_detected_ratio: float
    ) -> None:
        df = _load_frame_table(split, csv_path)
        n_rejected = 0
        n_before = len(self.person_ids)

        for person_id, group in df.groupby("person_id", sort=False):
            detected = group["detected"].to_numpy()
            if detected.mean() < min_detected_ratio:
                n_rejected += 1
                continue

            feats = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
            feats[~detected] = 0.0  # zero-fill undetected frames' feature values

            self.person_ids.append(person_id)
            self.features.append(feats)
            self.masks.append(detected.astype(np.float32))
            self.labels.append(group["target_index"].to_numpy(dtype=np.int64, copy=True))

        n_kept = len(self.person_ids) - n_before
        print(
            f"[{source_label}] LandmarkSequenceDataset: {n_kept} tracks kept, {n_rejected} rejected "
            f"(< {min_detected_ratio:.0%} detected frames), out of {n_kept + n_rejected} total"
        )

    def __len__(self) -> int:
        return len(self.person_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": torch.from_numpy(self.features[idx]),
            "mask": torch.from_numpy(self.masks[idx]),
            "labels": torch.from_numpy(self.labels[idx]),
            "length": self.features[idx].shape[0],
            "person_id": self.person_ids[idx],
        }


class StratifiedBatchSampler(Sampler[list[int]]):
    """Deals tracks into batches so each batch's SPEAKING_AUDIBLE frame
    percentage (masked, real-detected frames only) tracks the dataset's
    overall percentage far more tightly than plain random batching — plain
    shuffling was measured (2026-07-16) at per-batch stdev ~9.9pp (range
    14%-75%) around a ~38% train-split average.

    Not oversampling/undersampling, and not 50/50 balancing: every track is
    used exactly once per epoch, same as plain sampling, so the class prior
    the model actually trains on is unchanged — only which tracks land in
    the same batch changes. Tracks are sorted by their own speaking fraction
    and dealt round-robin across `n_batches` "hands" (like dealing cards),
    so each batch gets a representative spread across the full range of
    track speaking fractions rather than a random clump from one end.
    """

    def __init__(self, dataset: "LandmarkSequenceDataset", batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        speaking_fraction = np.array([
            labels[mask.astype(bool)].mean() if mask.any() else 0.0
            for labels, mask in zip(dataset.labels, dataset.masks)
        ])
        self.n = len(speaking_fraction)
        self.n_batches = math.ceil(self.n / batch_size) if self.n else 0
        self._sorted_order = np.argsort(speaking_fraction, kind="stable")

    def __iter__(self):
        batches: list[list[int]] = [[] for _ in range(self.n_batches)]
        for rank, idx in enumerate(self._sorted_order):
            batches[rank % self.n_batches].append(int(idx))
        if self.shuffle:
            np.random.shuffle(batches)
            for b in batches:
                np.random.shuffle(b)
        yield from batches

    def __len__(self) -> int:
        return self.n_batches


def collate_sequences(batch: list[dict]) -> dict:
    """Zero-pads a batch of variable-length tracks to the batch's own max
    length. Padding positions get mask=0 — the same value used for
    undetected-within-track frames — so any downstream masked mean/loss only
    needs to look at one tensor.
    """
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    max_len = int(lengths.max())
    batch_size = len(batch)

    features = torch.zeros((batch_size, max_len, NUM_INPUT_FEATURES), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    labels = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, b in enumerate(batch):
        t = b["length"]
        features[i, :t] = b["features"]
        mask[i, :t] = b["mask"]
        labels[i, :t] = b["labels"]

    return {
        "features": features,      # (B, T, NUM_INPUT_FEATURES)
        "mask": mask,               # (B, T) — 1.0 = real detected frame, 0.0 = undetected or padding
        "labels": labels,           # (B, T) — target_index (0=NOT_SPEAKING, 1=SPEAKING_AUDIBLE)
        "lengths": lengths,         # (B,)   — true track length (detected + undetected, excl. padding)
        "person_id": [b["person_id"] for b in batch],
    }


def get_dataloader(
    split: str,
    batch_size: int = BATCH_SIZE,
    shuffle: bool | None = None,
    num_workers: int = 0,
    extra_sources: list[tuple[Path, str]] = (),
) -> DataLoader:
    """shuffle defaults to True for the train split and False otherwise.
    When config.BALANCED_BATCHES is True, batches are built via
    StratifiedBatchSampler instead of plain random/sequential batching —
    see that class's docstring.

    extra_sources: passed straight through to LandmarkSequenceDataset — e.g.
    get_dataloader(split, extra_sources=[ava_source(split)]) trains on
    UniTalk-ASD + AVA combined instead of UniTalk-ASD alone.
    """
    dataset = LandmarkSequenceDataset(split, extra_sources=extra_sources)
    if shuffle is None:
        shuffle = split == "train"
    if BALANCED_BATCHES:
        batch_sampler = StratifiedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_sequences, num_workers=num_workers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_sequences,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    for split in LOGICAL_SPLITS:
        loader = get_dataloader(split)
        batch = next(iter(loader))
        print(
            f"[{split}] batch: features={tuple(batch['features'].shape)} "
            f"mask={tuple(batch['mask'].shape)} labels={tuple(batch['labels'].shape)} "
            f"mean(mask)={batch['mask'].mean():.3f}"
        )
