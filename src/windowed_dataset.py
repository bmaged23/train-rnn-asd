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

import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    AUGMENT_COPIES_PER_WINDOW,
    AUGMENT_LANDMARKS,
    AUGMENT_ON_THE_FLY,
    AUGMENT_ROTATION_MAX_DEGREES,
    AUGMENT_SCALE_RANGE,
    BALANCED_BATCHES,
    BALANCED_SOURCE_BATCHES,
    BATCH_SIZE,
    MIN_DETECTED_FRAMES_RATIO,
    USE_MAR,
    USE_MOUTH_GEOMETRY_FEATURES,
    USE_MOUTH_LANDMARKS_ONLY,
    WINDOW_MIN_DETECTED_FRAMES_RATIO,
    WINDOW_PAD_MIN_LENGTH_RATIO,
    WINDOW_PAD_SHORT_TRACKS,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    WINDOW_TRAIN_STRIDE,
    WINDOWED_CACHE_DIR,
    WINDOWED_SEED,
)

# ava_source()/wasd_source()/columbia_source()/avspeech_source()/speakingfaces_source() now live in dataset.py
# (shared with the per-frame pipeline — they're pure path resolution,
# nothing windowing-specific) and are re-exported here so existing
# `from windowed_dataset import ava_source` call sites
# (scripts/modeling/train/windowed_combined.py, scripts/modeling/evaluate/windowed_combined.py)
# keep working unchanged.
from dataset import (
    LOGICAL_SPLITS,
    NUM_INPUT_FEATURES,
    NUM_MOUTH_GEOMETRY_FEATURES,
    LandmarkSequenceDataset,
    ava_source,
    avspeech_source,
    columbia_source,
    compute_mouth_geometry_features,
    speakingfaces_source,
    split_csv_path,
    wasd_source,
)

_NUM_LANDMARKS = (NUM_INPUT_FEATURES - (1 if USE_MAR else 0) - NUM_MOUTH_GEOMETRY_FEATURES) // 3
# Offset of the geometry-feature block within a frame's feature vector —
# always the very last NUM_MOUTH_GEOMETRY_FEATURES columns (landmarks, then
# MAR if present, then geometry — see src/dataset.py's NUM_INPUT_FEATURES
# comment for this fixed column order).
_GEOMETRY_START = _NUM_LANDMARKS * 3 + (1 if USE_MAR else 0)


def _random_rotation_matrix(max_degrees: float) -> np.ndarray:
    """One small random 3D rotation — independent pitch/yaw/roll angles,
    uniform in [-max_degrees, max_degrees] — simulating mild camera-angle
    jitter. NOT a physically exact reprojection (see config.AUGMENT_LANDMARKS'
    comment for why that's an inherent limitation of augmenting already-2D-
    projected landmark points, not a bug here).
    """
    rx, ry, rz = np.radians(np.random.uniform(-max_degrees, max_degrees, size=3))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (rot_z @ rot_y @ rot_x).astype(np.float32)


def augment_landmarks(feats: np.ndarray) -> np.ndarray:
    """One fresh random rotation + scale applied to every frame's landmark
    (x, y, z) triples in `feats` (window_size, NUM_INPUT_FEATURES) — the
    SAME transform across all frames in the window (one random camera-angle
    draw per window, not per frame, since a window represents one
    continuous capture). MAR (if USE_MAR) is left untouched — see
    config.AUGMENT_LANDMARKS' comment. config.USE_MOUTH_GEOMETRY_FEATURES'
    18+18 distance features are the OPPOSITE case — deliberately
    RECOMPUTED from the rotated/scaled points below, not left untouched,
    since (unlike MAR, a self-normalizing ratio) they're raw distances that
    genuinely change under a scale/rotation jitter — see that config
    comment for why. Zero-padded/undetected frames are unaffected either
    way, since rotating or scaling the zero vector is still zero, and the
    geometry distance between two zero points is also zero — no
    special-casing needed for either.
    """
    landmark_cols = _NUM_LANDMARKS * 3
    points = feats[:, :landmark_cols].reshape(-1, _NUM_LANDMARKS, 3)
    rotation = _random_rotation_matrix(AUGMENT_ROTATION_MAX_DEGREES)
    scale = np.random.uniform(*AUGMENT_SCALE_RANGE)
    points = (points @ rotation.T) * scale

    augmented = feats.copy()
    augmented[:, :landmark_cols] = points.reshape(feats.shape[0], landmark_cols)

    if USE_MOUTH_GEOMETRY_FEATURES:
        geometry = compute_mouth_geometry_features(points).astype(np.float32)
        augmented[:, _GEOMETRY_START:_GEOMETRY_START + NUM_MOUTH_GEOMETRY_FEATURES] = geometry

    return augmented


def _cache_key(
    split: str,
    window_size: int,
    window_stride: int,
    min_detected_ratio: float,
    pad_short_tracks: bool,
    pad_min_length_ratio: float,
    extra_source_labels: list[str],
    materialize: bool,
) -> str:
    """Short, stable hash over every setting that actually affects the built
    arrays — a cache file only ever matches if ALL of these are identical to
    when it was built. Anything not listed here (e.g. BATCH_SIZE) doesn't
    affect WindowedLandmarkSequenceDataset's own contents, only how a
    DataLoader later consumes them, so it's deliberately excluded.

    Keyed on `materialize` (whether extra copies get baked into the cached
    arrays), NOT on whether on-the-fly augmentation is active — on-the-fly
    transforms happen in __getitem__, on top of whatever's cached, and never
    change what's stored. So AUGMENT_LANDMARKS=False and
    AUGMENT_LANDMARKS=True+AUGMENT_ON_THE_FLY=True correctly share the exact
    same cache (both store real-only windows); only AUGMENT_ON_THE_FLY=False
    (materialized mode) needs its own separate, larger cache.
    """
    payload = {
        "split": split,
        "window_size": window_size,
        "window_stride": window_stride,
        "min_detected_ratio": min_detected_ratio,
        "pad_short_tracks": pad_short_tracks,
        "pad_min_length_ratio": pad_min_length_ratio,
        "sources": sorted(["unitalk", *extra_source_labels]),
        "use_mouth_landmarks_only": USE_MOUTH_LANDMARKS_ONLY,
        "use_mar": USE_MAR,
        "use_mouth_geometry_features": USE_MOUTH_GEOMETRY_FEATURES,
        "min_detected_frames_ratio": MIN_DETECTED_FRAMES_RATIO,
        "materialize": materialize,
        "augment_rotation_max_degrees": AUGMENT_ROTATION_MAX_DEGREES if materialize else None,
        "augment_scale_range": list(AUGMENT_SCALE_RANGE) if materialize else None,
        "augment_copies_per_window": AUGMENT_COPIES_PER_WINDOW if materialize else None,
        "windowed_seed": WINDOWED_SEED,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return f"{split}_{digest}"


def _cache_path(cache_key: str) -> Path:
    return WINDOWED_CACHE_DIR / f"{cache_key}.pkl"


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
        # config.AUGMENT_LANDMARKS/AUGMENT_ON_THE_FLY, train-only — see that
        # config comment for the two modes. `materialize` gates the extra-
        # copies-at-construction-time path below; `self.augment_on_the_fly`
        # (checked in __getitem__) gates the fresh-transform-per-access path.
        # Mutually exclusive by construction — exactly one is ever True.
        augment = AUGMENT_LANDMARKS and split == "train"
        materialize = augment and not AUGMENT_ON_THE_FLY
        self.augment_on_the_fly = augment and AUGMENT_ON_THE_FLY

        # config.WINDOWED_CACHE_DIR — skip the entire CSV-load/window/augment
        # pipeline below if a matching cache from a previous run exists (see
        # that config comment for exactly what "matching" is keyed on).
        # TRAIN ONLY, by explicit user request — val/test are cheap enough
        # to rebuild (tens of seconds, vs. minutes for train) and are never
        # augmented anyway, so there's little to gain, and always rebuilding
        # them from the CSVs directly is one less thing to reason about
        # being stale.
        cache_key = _cache_key(
            split, window_size, window_stride, min_detected_ratio,
            pad_short_tracks, pad_min_length_ratio,
            [label for _, label in extra_sources], materialize,
        )
        cache_path = _cache_path(cache_key)
        use_cache = split == "train"
        if use_cache and cache_path.exists():
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            self.features = cached["features"]
            self.lengths = cached["lengths"]
            self.labels = cached["labels"]
            self.person_ids = cached["person_ids"]
            self.source_labels = cached["source_labels"]
            print(f"[{cache_key}] WindowedLandmarkSequenceDataset: loaded {len(self.labels)} windows from cache ({cache_path})")
            return

        # Explicit "unitalk" label (rather than defaulting to just `split`,
        # e.g. "train") so every base has an unambiguous, human-readable
        # source tag — needed by WindowedSourceBalancedBatchSampler below to
        # group windows by dataset, not just by split.
        bases = [LandmarkSequenceDataset(split, source_label="unitalk")]
        for csv_path, source_label in extra_sources:
            bases.append(LandmarkSequenceDataset(split, csv_path=csv_path, source_label=source_label))
        min_length_to_pad = window_size * pad_min_length_ratio

        self.features: list[np.ndarray] = []  # each (window_size, NUM_INPUT_FEATURES), zero-padded past `length`
        self.lengths: list[int] = []           # true (unpadded) frame count per window
        self.labels: list[int] = []            # one int (0/1) per window
        self.person_ids: list[str] = []        # source track, informational only
        self.source_labels: list[str] = []     # which dataset this window came from (e.g. "wasd", "avspeech")
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
                    self.source_labels.append(base.source_label)
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
                    self.source_labels.append(base.source_label)

        n_real = len(self.labels)
        n_augmented = 0
        if materialize and AUGMENT_COPIES_PER_WINDOW > 0:
            # Snapshot the real (pre-augmentation) window count/order before
            # appending — iterating range(n_real) rather than the live lists
            # so newly-appended augmented copies are never themselves
            # re-augmented into further copies.
            for i in range(n_real):
                for _ in range(AUGMENT_COPIES_PER_WINDOW):
                    self.features.append(augment_landmarks(self.features[i]))
                    self.lengths.append(self.lengths[i])
                    self.labels.append(self.labels[i])
                    self.person_ids.append(self.person_ids[i])
                    self.source_labels.append(self.source_labels[i])
            n_augmented = len(self.labels) - n_real

        n_kept = len(self.labels)
        n_speaking = int(sum(self.labels))
        source_tag = split if not extra_sources else f"{split}+{'+'.join(s for _, s in extra_sources)}"
        print(
            f"[{source_tag}] WindowedLandmarkSequenceDataset: {n_kept} windows kept "
            f"({n_real} real + {n_augmented} augmented), "
            f"({n_speaking} speaking / {n_kept - n_speaking} not-speaking), "
            f"{n_dropped_low_detection} windows dropped (< {min_detected_ratio:.0%} detected frames), "
            f"{n_padded_short_track} short tracks padded into one window each "
            f"(>= {min_length_to_pad:.0f}/{window_size} real frames required), "
            f"{n_dropped_short_track} tracks dropped as too short "
            f"(window_size={window_size}, pad_short_tracks={pad_short_tracks}, "
            f"pad_min_length_ratio={pad_min_length_ratio})"
        )

        if use_cache:
            WINDOWED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "features": self.features,
                        "lengths": self.lengths,
                        "labels": self.labels,
                        "person_ids": self.person_ids,
                        "source_labels": self.source_labels,
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            print(f"[{cache_key}] wrote cache -> {cache_path}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        # self.features[idx] itself is never modified — on-the-fly mode
        # (config.AUGMENT_ON_THE_FLY=True, the default) applies a fresh
        # random rotation/scale on every access instead, so the same window
        # gets a different perturbation each epoch. In materialized mode
        # self.augment_on_the_fly is False and any augmented copies already
        # live in self.features as their own separate rows from construction
        # time — nothing sample-specific happens here for those.
        feats = augment_landmarks(self.features[idx]) if self.augment_on_the_fly else self.features[idx]
        return {
            "features": torch.from_numpy(feats),
            "length": self.lengths[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "person_id": self.person_ids[idx],
            "source_label": self.source_labels[idx],
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


def _water_fill(target: int, capacities: dict[str, int]) -> dict[str, int]:
    """Distributes `target` units as equally as possible across the keys in
    `capacities`, never exceeding any key's own capacity. Any shortfall from
    a key hitting its cap is redistributed equally among the remaining
    not-yet-capped keys (repeats until either the target is fully placed or
    every key is capped) — standard water-filling / max-min fair allocation.
    """
    remaining = {k: c for k, c in capacities.items() if c > 0}
    allocated = {k: 0 for k in capacities}
    left = min(target, sum(remaining.values()))
    while left > 0 and remaining:
        share, extra = divmod(left, len(remaining))
        capped = []
        for i, k in enumerate(list(remaining)):
            want = share + (1 if i < extra else 0)
            give = min(want, remaining[k])
            allocated[k] += give
            remaining[k] -= give
            left -= give
            if remaining[k] == 0:
                capped.append(k)
        for k in capped:
            del remaining[k]
    return allocated


class WindowedSourceBalancedBatchSampler(Sampler[list[int]]):
    """Like WindowedStratifiedBatchSampler (each batch's SPEAKING_AUDIBLE
    fraction tracks the dataset's overall fraction), but ALSO — WITHIN each
    class — gives every contributing dataset source as close to an EQUAL
    share of that class's slots per batch as real data allows, instead of a
    share proportional to that source's size. Added 2026-07-29
    (config.BALANCED_SOURCE_BATCHES) once the combined dataset reached 6
    sources of wildly different sizes; plain WindowedStratifiedBatchSampler
    was verified to make each source's per-batch presence merely
    proportional (Columbia's ~0.65% of train windows meant it barely
    appeared in most batches at all).

    Every real window is used AT MOST once per epoch — same no-repeat
    guarantee as WindowedStratifiedBatchSampler and every other sampler in
    this project. An earlier version (2026-07-29) instead oversampled small
    sources with an infinite reshuffle-and-repeat generator to force a
    perfectly equal split every single batch; rejected 2026-07-30 (explicit
    user request) once it became clear that repeating Columbia's tiny pool
    ~30x/epoch risked the model just memorizing that handful of tracks
    rather than generalizing, on top of already being an unusual departure
    from this project's otherwise-universal exactly-once-per-epoch rule.
    Now: once a source's real supply for a class runs out, it simply stops
    contributing for the rest of the epoch — see _water_fill() — and its
    share is redistributed among whichever sources for that class still
    have data left, exactly as requested ("if the data finished it's okay,
    don't repeat it, just use the rest"). Composition is only PERFECTLY
    equal-per-source in the early batches, before the smallest sources
    (Columbia first) run out; later batches lean increasingly on whichever
    large sources (WASD/AVSpeech) still have real data remaining — this is
    the expected, intentional shape, not a bug.

    Per-class target uses a fractional carry/accumulator across batches
    (fixed 2026-08-23, real bug found + verified against VLM training:
    batch_size=1 with e.g. a 58%/42% class split made the OLD
    per-batch-independent `round(batch_size * label_fraction)` always
    resolve to 1 for the majority class and 0 for the minority — EVERY
    single batch, all epoch, since round() of a fixed <0.5 fraction never
    crosses over. Once the majority class ran out, that iteration's batch
    was empty for both classes and the loop broke immediately, having
    drawn zero minority-class windows the entire epoch — confirmed via a
    real-ratio simulation: 1,722/3,000 windows consumed, 0 of the minority
    class). Now each label accumulates `batch_size * label_fraction[label]`
    every iteration it still has capacity; the integer part is drawn (and
    subtracted back out) once it reaches >=1, so a label owed e.g. 0.42
    windows/batch still gets drawn roughly 42% of the time instead of
    never. An iteration can therefore legitimately draw nothing yet from
    either class while both accumulators are still <1 — that no longer
    ends the epoch; only both labels' real capacity running out does (see
    `any_caps` in __iter__). At batch_size >= ~3 this converges to
    essentially the same per-batch composition the old round()-based
    version produced, so the fix is only user-visible at small batch
    sizes like the VLM pipeline's batch_size=1.
    """

    def __init__(self, dataset: "WindowedLandmarkSequenceDataset", batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        labels = np.asarray(dataset.labels)
        sources = np.asarray(dataset.source_labels)
        self.n = len(labels)
        self.n_batches = math.ceil(self.n / batch_size) if self.n else 0

        # {0: {"unitalk": [idx, ...], "wasd": [...], ...}, 1: {...}}
        self._cells: dict[int, dict[str, list[int]]] = {0: {}, 1: {}}
        for idx, (label, source) in enumerate(zip(labels.tolist(), sources.tolist())):
            self._cells[label].setdefault(source, []).append(idx)

        n0 = sum(len(idxs) for idxs in self._cells[0].values())
        n1 = sum(len(idxs) for idxs in self._cells[1].values())
        self._label_fraction = {0: n0 / self.n, 1: n1 / self.n} if self.n else {0: 0.5, 1: 0.5}

    def __iter__(self):
        # Shuffled, non-repeating per-(label, source) queues — popped from
        # the front (via an index cursor, not actual list mutation) as each
        # batch draws from them. Reshuffled fresh every __iter__ call (i.e.
        # every epoch), same as every other sampler here.
        pools: dict[int, dict[str, list[int]]] = {0: {}, 1: {}}
        cursors: dict[int, dict[str, int]] = {0: {}, 1: {}}
        for label in (0, 1):
            for source, idxs in self._cells[label].items():
                order = list(idxs)
                if self.shuffle:
                    np.random.shuffle(order)
                pools[label][source] = order
                cursors[label][source] = 0

        def remaining(label: int) -> dict[str, int]:
            return {
                source: len(pools[label][source]) - cursors[label][source]
                for source in pools[label]
                if len(pools[label][source]) - cursors[label][source] > 0
            }

        # Batches are collected first, THEN shuffled as a whole (not just
        # yielded in construction order) — otherwise every epoch would have
        # the exact same shape regardless of shuffling within a batch: early
        # batches diverse across all 6 sources, late batches narrowing down
        # to whichever large source (WASD/AVSpeech) hasn't run out yet. That
        # would mean the model trains on a systematically drifting mix
        # within every single epoch (broad early, narrow late) — shuffling
        # batch ORDER, not just batch CONTENTS, breaks the correlation
        # between a batch's position in the epoch and which sources are
        # still available, spreading the "narrow" batches randomly
        # throughout instead of concentrating them at the end.
        # Fractional carry per label — see class docstring's "fixed
        # 2026-08-23" note. Reset fresh every __iter__ call (every epoch),
        # same as pools/cursors above.
        accumulator = {0: 0.0, 1: 0.0}

        batches: list[list[int]] = []
        total_remaining = self.n
        while total_remaining > 0:
            batch: list[int] = []
            any_caps = False
            for label in (0, 1):
                caps = remaining(label)
                if not caps:
                    continue
                any_caps = True
                accumulator[label] += self.batch_size * self._label_fraction[label]
                target = int(accumulator[label])
                accumulator[label] -= target
                if target == 0:
                    continue
                allocation = _water_fill(target, caps)
                for source, count in allocation.items():
                    start = cursors[label][source]
                    batch.extend(pools[label][source][start : start + count])
                    cursors[label][source] += count
                    total_remaining -= count
            if not any_caps:
                break  # both labels' real capacity exhausted — epoch complete
            if not batch:
                continue  # both accumulators still <1 this round — keep accruing, not an end condition
            if self.shuffle:
                np.random.shuffle(batch)
            batches.append(batch)
        if self.shuffle:
            np.random.shuffle(batches)
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
        "source_label": [b["source_label"] for b in batch],
    }


def get_windowed_dataloader(
    split: str,
    batch_size: int = BATCH_SIZE,
    shuffle: bool | None = None,
    num_workers: int = 0,
    extra_sources: list[tuple[Path, str]] = (),
) -> DataLoader:
    """shuffle defaults to True for the train split and False otherwise.
    When config.BALANCED_SOURCE_BATCHES is True, batches are built via
    WindowedSourceBalancedBatchSampler — class-balanced like
    WindowedStratifiedBatchSampler below, but ALSO equal-per-dataset-source
    within each class (see that class's docstring for why this needed a
    dedicated sampler once the combined dataset reached 6 very
    differently-sized sources). Otherwise, when config.BALANCED_BATCHES is
    True, batches are built via WindowedStratifiedBatchSampler instead of
    plain random/sequential batching — see that class's docstring. Shared
    with the per-frame pipeline's get_dataloader() via the same
    BALANCED_BATCHES flag (BALANCED_SOURCE_BATCHES is windowed-only).

    "train" uses config.WINDOW_TRAIN_STRIDE (overlapping windows, more
    training examples per track); "val"/"test" always use the non-overlapping
    config.WINDOW_STRIDE, so early-stopping/evaluation signal stays honest —
    see config.py's WINDOW_TRAIN_STRIDE comment for why.

    BALANCED_SOURCE_BATCHES is applied to "train" ONLY, same train-only
    principle as WINDOW_TRAIN_STRIDE above and for the same reason: unlike
    WindowedStratifiedBatchSampler (reorders every real window exactly
    once, safe anywhere), WindowedSourceBalancedBatchSampler genuinely
    oversamples small sources and skips some real windows from large ones
    each pass to hit equal per-batch quotas — appropriate for shaping what
    the model trains on, but it would make val/test measure a distorted,
    non-representative resample instead of the true held-out data. Caught
    2026-07-30 after a training run's val loader was silently distorted
    this way (identical per-source sample counts in a test-time diagnostic
    that should never match — the tell that oversampling had leaked into
    eval) and produced a worse, noisier result that couldn't be trusted.

    extra_sources: passed straight through to WindowedLandmarkSequenceDataset
    — e.g. get_windowed_dataloader(split, extra_sources=[ava_source(split)])
    trains on UniTalk-ASD + AVA combined instead of UniTalk-ASD alone.
    """
    window_stride = WINDOW_TRAIN_STRIDE if split == "train" else WINDOW_STRIDE
    dataset = WindowedLandmarkSequenceDataset(split, window_stride=window_stride, extra_sources=extra_sources)
    if shuffle is None:
        shuffle = split == "train"
    if BALANCED_SOURCE_BATCHES and split == "train":
        batch_sampler = WindowedSourceBalancedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_windows, num_workers=num_workers)
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
