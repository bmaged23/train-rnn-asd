"""PyTorch Dataset/DataLoader for the windowed many-to-one variant of the ASD RNN.

Alongside (not replacing) src/dataset.py's per-frame many-to-many pipeline —
see config.py's "windowed" section comment for the full rationale. Wraps
LandmarkSequenceDataset (reusing its CSV loading, feature-selection toggles,
and track-level MIN_DETECTED_FRAMES_RATIO filtering) and chops each kept
track into fixed-length, non-overlapping windows of config.WINDOW_SIZE
frames (= config.WINDOW_SECONDS * config.DATASET_FPS — window duration is
configured in seconds, not hardcoded frames). Each window becomes one
sample with ONE label — majority vote
over that window's masked (real, detected) frames, ties going to
NOT_SPEAKING (0), matching the project's >0.5 threshold convention used
elsewhere — rather than one sample per frame.

Windows are built strictly after LandmarkSequenceDataset's split boundaries,
so this inherits the same train/val/test separation with no additional
leakage risk (whole tracks — and therefore all of their windows — stay
within one split).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    BALANCED_BATCHES,
    BATCH_SIZE,
    WINDOW_MIN_DETECTED_FRAMES_RATIO,
    WINDOW_PAD_MIN_LENGTH_RATIO,
    WINDOW_PAD_SHORT_TRACKS,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    WINDOW_TRAIN_STRIDE,
)

# ava_source()/wasd_source() now live in dataset.py (shared with the
# per-frame pipeline — they're pure path resolution, nothing
# windowing-specific) and are re-exported here so existing
# `from windowed_dataset import ava_source` call sites
# (scripts/modeling/train/windowed_combined.py, scripts/modeling/evaluate/windowed_combined.py)
# keep working unchanged.
from dataset import LOGICAL_SPLITS, LandmarkSequenceDataset, ava_source, split_csv_path, wasd_source


class WindowedLandmarkSequenceDataset(Dataset):
    """One sample = one fixed-length (config.WINDOW_SIZE) window of frames
    from a single track, with one majority-vote label. Built by slicing
    LandmarkSequenceDataset's already-loaded, already-filtered tracks —
    doesn't re-read the CSV.

    When `pad_short_tracks` is True (config.WINDOW_PAD_SHORT_TRACKS), tracks
    shorter than `window_size` — but at least `pad_min_length_ratio` of it
    (config.WINDOW_PAD_MIN_LENGTH_RATIO) — get ONE zero-padded window each
    instead of being dropped; tracks below that floor are still dropped, same
    as when padding is off entirely (see config.py's comments for the
    rationale). Such windows carry a `length` < `window_size`;
    WindowedSpeakingDetectorRNN uses this (via pack_padded_sequence) so
    padding never influences the model's output. Full-length windows always
    report `length == window_size`, so this is a no-op when the flag is off
    or a track is long enough anyway.
    """

    def __init__(
        self,
        split: str,
        window_size: int = WINDOW_SIZE,
        window_stride: int = WINDOW_STRIDE,
        min_detected_ratio: float = WINDOW_MIN_DETECTED_FRAMES_RATIO,
        pad_short_tracks: bool = WINDOW_PAD_SHORT_TRACKS,
        pad_min_length_ratio: float = WINDOW_PAD_MIN_LENGTH_RATIO,
        extra_sources: list[tuple[Path, str]] = (),
    ):
        """extra_sources: additional (csv_path, source_label) pairs — each
        loaded as its own LandmarkSequenceDataset and windowed alongside the
        default UniTalk-ASD source, for training on multiple datasets
        combined (see ava_source() above). Empty by default — existing
        callers (scripts/modeling/train/windowed.py, scripts/modeling/evaluate/windowed.py) are
        unaffected unless they opt in.
        """
        if split not in LOGICAL_SPLITS:
            raise ValueError(f"split must be one of {LOGICAL_SPLITS}, got {split!r}")
        self.split = split
        self.window_size = window_size

        bases = [LandmarkSequenceDataset(split)]
        for csv_path, source_label in extra_sources:
            bases.append(LandmarkSequenceDataset(split, csv_path=csv_path, source_label=source_label))
        min_length_to_pad = window_size * pad_min_length_ratio

        self.features: list[np.ndarray] = []  # each (window_size, NUM_INPUT_FEATURES), zero-padded past `length`
        self.lengths: list[int] = []           # true (unpadded) frame count per window
        self.labels: list[int] = []            # one int (0/1) per window
        self.person_ids: list[str] = []        # source track, informational only
        n_dropped_short_track = 0
        n_padded_short_track = 0
        n_dropped_low_detection = 0

        for base in bases:
            for person_id, feats, mask, labels in zip(base.person_ids, base.features, base.masks, base.labels):
                track_len = feats.shape[0]
                if track_len < window_size:
                    if not pad_short_tracks or track_len < min_length_to_pad:
                        n_dropped_short_track += 1
                        continue

                    window_mask = mask.astype(bool)
                    if window_mask.mean() < min_detected_ratio:
                        n_dropped_low_detection += 1
                        continue

                    window_labels = labels[window_mask]
                    window_label = int(window_labels.mean() > 0.5)  # majority vote, ties -> 0 (NOT_SPEAKING)

                    padded_feats = np.zeros((window_size, feats.shape[1]), dtype=feats.dtype)
                    padded_feats[:track_len] = feats

                    self.features.append(padded_feats)
                    self.lengths.append(track_len)
                    self.labels.append(window_label)
                    self.person_ids.append(person_id)
                    n_padded_short_track += 1
                    continue

                for start in range(0, track_len - window_size + 1, window_stride):
                    end = start + window_size
                    window_mask = mask[start:end].astype(bool)
                    if window_mask.mean() < min_detected_ratio:
                        n_dropped_low_detection += 1
                        continue

                    window_labels = labels[start:end][window_mask]
                    window_label = int(window_labels.mean() > 0.5)  # majority vote, ties -> 0 (NOT_SPEAKING)

                    self.features.append(feats[start:end])
                    self.lengths.append(window_size)
                    self.labels.append(window_label)
                    self.person_ids.append(person_id)

        n_kept = len(self.labels)
        n_speaking = int(sum(self.labels))
        source_tag = split if not extra_sources else f"{split}+{'+'.join(s for _, s in extra_sources)}"
        print(
            f"[{source_tag}] WindowedLandmarkSequenceDataset: {n_kept} windows kept "
            f"({n_speaking} speaking / {n_kept - n_speaking} not-speaking), "
            f"{n_dropped_low_detection} windows dropped (< {min_detected_ratio:.0%} detected frames), "
            f"{n_padded_short_track} short tracks padded into one window each "
            f"(>= {min_length_to_pad:.0f}/{window_size} real frames required), "
            f"{n_dropped_short_track} tracks dropped as too short "
            f"(window_size={window_size}, pad_short_tracks={pad_short_tracks}, "
            f"pad_min_length_ratio={pad_min_length_ratio})"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": torch.from_numpy(self.features[idx]),
            "length": self.lengths[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "person_id": self.person_ids[idx],
        }


class WindowedStratifiedBatchSampler(Sampler[list[int]]):
    """Deals windows into batches so each batch's SPEAKING_AUDIBLE fraction
    tracks the dataset's overall fraction, instead of swinging batch-to-batch
    under plain random sampling — same rationale and "deal sorted items
    round-robin across batches" technique as src/dataset.py's
    StratifiedBatchSampler (see [[dataloader-design]] in project memory),
    but simpler here: each window already has a single scalar 0/1 label, so
    there's no need to first compute a per-track fraction the way the
    per-frame version does for variable-length mixed tracks. Not
    oversampling/undersampling — every window is still used exactly once
    per epoch, only reordered, so the class prior is unchanged.

    Kept as a separate implementation rather than sharing code with
    src/dataset.py's version — that file is deliberately untouched by the
    windowed pipeline (see [[windowed-approach]]).
    """

    def __init__(self, dataset: "WindowedLandmarkSequenceDataset", batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        labels = np.array(dataset.labels)
        self.n = len(labels)
        self.n_batches = math.ceil(self.n / batch_size) if self.n else 0
        # Stable sort on a binary array groups into a 0-block then a 1-block;
        # dealing that round-robin across n_batches "hands" distributes both
        # blocks proportionally into every batch — the same effect as
        # proportional interleaving, without needing separate bookkeeping.
        self._sorted_order = np.argsort(labels, kind="stable")

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


def collate_windows(batch: list[dict]) -> dict:
    """Features are already fixed-length (window_size) — real windows filled
    entirely, padded short-track windows zero-filled past `length` — so this
    is just a stack, no per-batch padding logic needed. `lengths` lets
    WindowedSpeakingDetectorRNN ignore the padding via pack_padded_sequence.
    """
    return {
        "features": torch.stack([b["features"] for b in batch]),          # (B, window_size, NUM_INPUT_FEATURES)
        "lengths": torch.tensor([b["length"] for b in batch], dtype=torch.long),  # (B,)
        "labels": torch.stack([b["label"] for b in batch]),                # (B,)
        "person_id": [b["person_id"] for b in batch],
    }


def get_windowed_dataloader(
    split: str,
    batch_size: int = BATCH_SIZE,
    shuffle: bool | None = None,
    num_workers: int = 0,
    extra_sources: list[tuple[Path, str]] = (),
) -> DataLoader:
    """shuffle defaults to True for the train split and False otherwise.
    When config.BALANCED_BATCHES is True, batches are built via
    WindowedStratifiedBatchSampler instead of plain random/sequential
    batching — see that class's docstring. Shared with the per-frame
    pipeline's get_dataloader() via the same config flag.

    "train" uses config.WINDOW_TRAIN_STRIDE (overlapping windows, more
    training examples per track); "val"/"test" always use the non-overlapping
    config.WINDOW_STRIDE, so early-stopping/evaluation signal stays honest —
    see config.py's WINDOW_TRAIN_STRIDE comment for why.

    extra_sources: passed straight through to WindowedLandmarkSequenceDataset
    — e.g. get_windowed_dataloader(split, extra_sources=[ava_source(split)])
    trains on UniTalk-ASD + AVA combined instead of UniTalk-ASD alone.
    """
    window_stride = WINDOW_TRAIN_STRIDE if split == "train" else WINDOW_STRIDE
    dataset = WindowedLandmarkSequenceDataset(split, window_stride=window_stride, extra_sources=extra_sources)
    if shuffle is None:
        shuffle = split == "train"
    if BALANCED_BATCHES:
        batch_sampler = WindowedStratifiedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_windows, num_workers=num_workers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_windows,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    for split in LOGICAL_SPLITS:
        loader = get_windowed_dataloader(split)
        batch = next(iter(loader))
        print(
            f"[{split}] batch: features={tuple(batch['features'].shape)} "
            f"labels={tuple(batch['labels'].shape)} mean(labels)={batch['labels'].float().mean():.3f}"
        )
