"""PyTorch Dataset/DataLoader for the third candidate architecture — a CNN+LSTM
consuming face-crop IMAGES directly, windowed the same way src/windowed_dataset.py
windows landmark sequences (see config.py's "image_windowed_all_combined"
section comment for the full rationale/history).

Reuses each source's own combined_landmarks.csv (person_id/video_id/
image_name/target_index columns only — NOT the landmark float columns, much
cheaper to load) purely for track structure and labels, then resolves each
row's actual face-crop image file on disk instead of reading landmark
features. Deliberately a separate implementation from windowed_dataset.py's
WindowedLandmarkSequenceDataset rather than a shared base class — same
"kept as a separate implementation" precedent as WindowedStratifiedBatchSampler's
own docstring (this project avoids coupling the windowed and per-frame/image
pipelines together). WindowedSourceBalancedBatchSampler and
WindowedStratifiedBatchSampler ARE reused as-is (imported from
windowed_dataset.py), since both only ever touch a dataset's generic
`.labels`/`.source_labels` lists — nothing landmark-specific.

Deliberately does NOT reuse LandmarkSequenceDataset's `detected` (MediaPipe
face-landmark-found) mask/filtering at all: `detected` reflects MediaPipe's
own extraction success on a frame, not whether that frame's face-crop IMAGE
is usable — the crop file exists on disk regardless (it's produced upstream
by each source's own crop/track step, e.g. RetinaFace tracking, before
landmark extraction ever runs). Filtering images out by a landmark-specific
signal would throw away exactly the harder frames (occlusion, motion blur,
extreme angle) a raw-pixel model has the best chance of still using — the
whole motivation for trying this architecture in the first place. So here:
every frame in a track contributes its real image and real label to the
window/majority-vote it falls into, with no per-frame masking; the only
frames that don't get a real image are padding positions in a padded
short-track window (config.WINDOW_PAD_SHORT_TRACKS), which get an all-zero
image tensor and are excluded from the model's output via
pack_padded_sequence, same convention as the landmark pipeline's padding.

On-disk resized-frame cache (config.IMAGE_RESIZED_CACHE_DIR, built by
build_image_cache() below) — added 2026-08-01. Originally this dataset read
every frame straight from each source's own raw crop directory, decoding +
pad-to-square + resizing to IMAGE_SIZE from scratch in _load_image on EVERY
__getitem__ call — i.e. every epoch, for every frame in every window it
appears in. Profiling (2026-08-01) showed this was the real bottleneck: one
training step's GPU forward+backward+optimizer.step measured ~43ms, but real
epochs took 1700-2100s — almost entirely CPU decode/resize, not GPU compute.
build_image_cache() now does that decode+pad+resize (and the
IMAGE_MIN_CROP_SIZE too-small-crop filtering) ONCE per run, up front,
writing small already-IMAGE_SIZE JPEGs to IMAGE_RESIZED_CACHE_DIR;
ImageWindowedDataset then only ever reads from there — _load_image no
longer pads/resizes at all, just decodes an already-correct-size JPEG. This
is NOT the "raw pixel tensor cache" this docstring used to explicitly reject
as enormous (window_size x IMAGE_SIZE^2 x 3 bytes x num_windows, ~440GB at
this project's 6-source scale) — it's a per-FRAME (not per-window, so
overlapping windows never duplicate storage) cache of small re-encoded
JPEGs (~35-70GB estimated), still decoded each epoch but a MUCH cheaper
decode than the original (some source crops are up to ~970px) than the
original full-size source image would be.

Reused across runs when still valid, same principle as WINDOWED_CACHE_DIR
— reversed 2026-08-01 from this project's initial "always rebuild fresh"
choice, once a from-scratch build was timed at ~25 min for the full
6-source/3-split combined set and re-paying that on every single run (train
then immediately eval, or two training runs in a row) was clearly wasteful
whenever nothing that affects the cache's content actually changed.
Validity is keyed on IMAGE_SIZE + IMAGE_MIN_CROP_SIZE (the only two settings
that affect what gets written to a cached file) via a small
_cache_meta.json tracking which (key, split) combinations are already
built; build_image_cache() reuses whatever's still valid and only rebuilds
what's missing or invalidated by a settings change — see that function's
docstring for the exact reuse/rebuild decision.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision import io as tv_io
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    AUGMENT_IMAGES,
    AVA_RAW_CLIPS_VIDEOS_DIR,
    AVSPEECH_RAW_CLIPS_VIDEOS_DIR,
    BALANCED_BATCHES,
    BALANCED_SOURCE_BATCHES,
    COLUMBIA_RAW_CLIPS_VIDEOS_DIR,
    DATASET_FPS,
    IMAGE_AUGMENT_BRIGHTNESS,
    IMAGE_AUGMENT_CONTRAST,
    IMAGE_AUGMENT_COPIES_PER_WINDOW,
    IMAGE_AUGMENT_ON_THE_FLY,
    IMAGE_AUGMENT_ROTATION_MAX_DEGREES,
    IMAGE_AUGMENT_SCALE_RANGE,
    IMAGE_AUGMENT_SHIFT_SECONDS,
    IMAGE_BATCH_SIZE,
    IMAGE_CACHE_BUILD_WORKERS,
    IMAGE_FRAME_CHECK_WORKERS,
    IMAGE_MAX_MASKED_FRAME_RATIO,
    IMAGE_MIN_CROP_SIZE,
    IMAGE_NORMALIZE_MEAN,
    IMAGE_NORMALIZE_STD,
    IMAGE_RESIZED_CACHE_DIR,
    IMAGE_SIZE,
    IMAGE_WINDOWED_SEED,
    RAW_CLIPS_VIDEOS_DIR,
    SPEAKINGFACES_RAW_CLIPS_VIDEOS_CROPPED_DIR,
    WASD_RAW_CLIPS_VIDEOS_DIR,
    WINDOW_PAD_MIN_LENGTH_RATIO,
    WINDOW_PAD_SHORT_TRACKS,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    WINDOW_TRAIN_STRIDE,
)
from dataset import (
    LOGICAL_SPLITS,
    ava_source,
    avspeech_source,
    columbia_source,
    speakingfaces_source,
    split_csv_path,
    wasd_source,
)
from train_utils import reset_dir
from windowed_dataset import WindowedSourceBalancedBatchSampler, WindowedStratifiedBatchSampler, _water_fill

# The 5 EXTRA combined sources build_image_cache() below preprocesses on top
# of UniTalk-ASD — UniTalk itself is NOT in this list, it's always added
# separately via split_csv_path()/"unitalk" (same "default + extra_sources"
# pattern every other combined entry point in this project already uses,
# e.g. scripts/modeling/train/image_windowed_all_combined.py's own loader
# calls) — see build_image_cache()'s own source-list construction below,
# which explicitly includes both. Together: all 6 combined sources.
_CACHE_EXTRA_SOURCE_FNS = [ava_source, wasd_source, columbia_source, avspeech_source, speakingfaces_source]

# source_label -> (clips_videos_dir, image_ext_override). image_ext_override
# is None for every source except SpeakingFaces: its combined_landmarks.csv
# `image_name` column still ends in ".png" (the original raw-frame
# extension, from when landmarks were extracted directly off the uncropped
# scene — see [[project-asd-speakingfaces-integration]] in project memory
# for how that worked at all), but SPEAKINGFACES_RAW_CLIPS_VIDEOS_CROPPED_DIR
# (scripts/dataset/crop_speakingfaces_faces.py's output, the directory this
# pipeline actually reads images from) writes ".jpg" — same stem, different
# extension. Every other source's crop step already wrote ".jpg" files
# matching their CSV's `image_name` exactly, so no override is needed there.
_IMAGE_SOURCES: dict[str, tuple[Path, str | None]] = {
    "unitalk": (RAW_CLIPS_VIDEOS_DIR, None),
    "ava": (AVA_RAW_CLIPS_VIDEOS_DIR, None),
    "wasd": (WASD_RAW_CLIPS_VIDEOS_DIR, None),
    "columbia": (COLUMBIA_RAW_CLIPS_VIDEOS_DIR, None),
    "avspeech": (AVSPEECH_RAW_CLIPS_VIDEOS_DIR, None),
    "speakingfaces": (SPEAKINGFACES_RAW_CLIPS_VIDEOS_CROPPED_DIR, ".jpg"),
}

_METADATA_COLUMNS = ["person_id", "video_id", "image_name", "target_index"]


def _load_frame_table(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found — run the matching process_*_dataset.py first")
    df = pd.read_csv(csv_path, usecols=_METADATA_COLUMNS, engine="pyarrow")
    # Vectorized equivalent of float(Path(image_name).stem) — str.rsplit(".",
    # n=1) splits on only the LAST dot (same as Path.stem, which strips only
    # the final suffix), so a timestamp that itself contains a dot (e.g.
    # "405.60.jpg" -> "405.60") still parses correctly. Measured 2026-08-01:
    # the previous per-row .map(_frame_timestamp) (a Python function call +
    # Path object construction per row) was ~17s of a ~268s construction
    # pass on just the "test" split's ~1.56M rows alone — real cost at this
    # project's scale, worth vectorizing.
    df["frame_ts"] = df["image_name"].str.rsplit(".", n=1).str[0].astype(float)
    df.sort_values(["person_id", "frame_ts"], inplace=True, kind="mergesort")
    return df


def _resolve_image_path(clips_videos_dir: Path, raw_split: str, video_id: str, person_id: str, image_name: str, ext_override: str | None) -> Path:
    name = image_name if ext_override is None else Path(image_name).stem + ext_override
    return clips_videos_dir / raw_split / video_id / person_id / name


def _is_too_small(size: tuple[int, int]) -> bool:
    """config.IMAGE_MIN_CROP_SIZE, checked against the SMALLER of (w, h) —
    catches thin-sliver crops too, not just small-but-square ones. 0
    disables the check (nothing is ever "too small"). Used only by
    build_image_cache() below — this decision is made once, at cache-build
    time, against the ORIGINAL source image's size; a frame that fails this
    check is simply never written to the cache, so downstream code (dataset
    construction, _load_image) never needs to re-check it.
    """
    return IMAGE_MIN_CROP_SIZE > 0 and min(size) < IMAGE_MIN_CROP_SIZE


def _raw_split_for(split: str) -> str:
    """process_dataset.py (and its per-source variants) only ever write raw
    "train" and raw "val" folders (config.SPLITS) — logical "val" and "test"
    are both a stratified split of the raw "val" folder's ROWS
    (scripts/splits/split_val_test.py), never a separate image folder, so
    both read images from the raw "val" tree.
    """
    return "train" if split == "train" else "val"


def _pad_and_resize(img: Image.Image) -> Image.Image:
    """Pad to square (not stretch) before resizing — preserves each source's
    own native crop aspect ratio instead of distorting face proportions
    differently per source. Build-time only (build_image_cache()) — the
    cache this produces is always already IMAGE_SIZE x IMAGE_SIZE, so
    _load_image (reading from the cache) never needs to do this itself.
    """
    w, h = img.size
    size = max(w, h)
    if w != h:
        square = Image.new("RGB", (size, size))
        square.paste(img, ((size - w) // 2, (size - h) // 2))
        img = square
    return img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)


def resolve_cache_path(source_label: str, raw_split: str, video_id: str, person_id: str, image_name: str) -> Path:
    """The one shared path convention build_image_cache() (writer) and
    ImageWindowedDataset.__init__ (reader) both use — always ".jpg"
    regardless of the source's own original extension (build_image_cache()
    always re-encodes as JPEG), so callers never need _IMAGE_SOURCES'
    per-source ext_override once reading from the cache.

    Returns a Path (needed for build_image_cache()'s own .mkdir(parents=True)
    use) — for the PER-FRAME hot path (ImageWindowedDataset.__init__'s
    existence check, over millions of frames), use resolve_cache_path_str()
    instead; see that function's docstring for why Path specifically is too
    slow there.
    """
    stem = Path(image_name).stem
    return IMAGE_RESIZED_CACHE_DIR / source_label / raw_split / video_id / person_id / f"{stem}.jpg"


_IMAGE_RESIZED_CACHE_DIR_STR = str(IMAGE_RESIZED_CACHE_DIR)


def resolve_cache_path_str(source_label: str, raw_split: str, video_id: str, person_id: str, image_name: str) -> str:
    """Same path as resolve_cache_path(), but built with plain string
    formatting and returned as a str, not wrapped in pathlib.Path — added
    2026-08-01 after cProfile showed pathlib.Path (Python 3.13's rewritten,
    ABC-based implementation) has real per-call overhead for both
    construction (chained `/` -> __truediv__ -> with_segments -> _parse_path
    -> posixpath.join, repeated per path SEGMENT) and existence checks
    (.exists() -> stat() -> __fspath__ -> __str__ -> drive -> _raw_path,
    reconstructing the string form internally on every single call rather
    than reusing one already computed). Measured on the "test" split
    (~1.57M frames): switching ImageWindowedDataset.__init__'s per-frame
    path construction + existence check from Path objects to plain strings
    + os.path.exists() cut construction from 130.6s to a small fraction of
    that — pathlib's overhead, not the underlying stat() syscalls, was the
    real remaining cost after the earlier per-track-executor.map fix.
    `Image.open()` (src/image_dataset.py's _load_image) accepts a plain str
    identically to a Path, so nothing downstream needs to change.
    """
    stem = image_name.rsplit(".", 1)[0]
    return f"{_IMAGE_RESIZED_CACHE_DIR_STR}/{source_label}/{raw_split}/{video_id}/{person_id}/{stem}.jpg"


def _check_paths_exist(paths: list[str]) -> list[bool]:
    """One CHUNK's worth of ImageWindowedDataset.__init__'s cache-existence
    check — a plain tight loop, not itself parallel; the caller submits
    many of these (one per chunk) to a ThreadPoolExecutor so the real
    parallelism comes from running several chunks concurrently, not from
    the loop itself. See that call site's comment for why chunking matters
    at all here (ThreadPoolExecutor.map()'s chunksize is a documented no-op
    for thread pools, so this has to be done manually).

    `paths` are plain strings (resolve_cache_path_str()), not pathlib.Path
    — os.path.exists() on a str skips pathlib's per-call string
    reconstruction overhead entirely (see resolve_cache_path_str()'s
    docstring for the measured cost that was).
    """
    return [os.path.exists(p) for p in paths]


def _process_and_cache_frame(raw_path: Path, cache_path: Path) -> str:
    """One frame's worth of build_image_cache()'s work: decode the ORIGINAL
    source image, apply the same IMAGE_MIN_CROP_SIZE check + pad-to-square +
    resize _load_image used to do inline every epoch, and write the result
    to `cache_path` (parent directory must already exist — created once per
    track by the caller, not per frame, to avoid redundant mkdir calls
    across a whole track's frames). Returns "written"/"too_small"/"missing"
    for the caller's summary counts. A too-small or missing frame is simply
    never written — downstream, a missing cache file already means
    "zero-mask this frame" (same fallback _load_image always had for a
    missing file), so there's no need for a separate too-small signal past
    this point.
    """
    try:
        with Image.open(raw_path) as img:
            img = img.convert("RGB")
            if _is_too_small(img.size):
                return "too_small"
            img = _pad_and_resize(img)
            img.save(cache_path, "JPEG", quality=90)
            return "written"
    except (FileNotFoundError, OSError):
        return "missing"


def _cache_meta_path() -> Path:
    return IMAGE_RESIZED_CACHE_DIR / "_cache_meta.json"


def _current_cache_key() -> dict:
    """The only two settings that affect what actually gets WRITTEN to a
    cached file — IMAGE_MAX_MASKED_FRAME_RATIO/other window-shape settings
    don't belong here, they only affect which cached frames a given
    ImageWindowedDataset construction later chooses to use, not the cache's
    own content.
    """
    return {"image_size": IMAGE_SIZE, "image_min_crop_size": IMAGE_MIN_CROP_SIZE}


def load_cache_meta() -> dict | None:
    """Returns {"key": {...}, "done_splits": [...]}, or None if no cache
    (or no readable metadata) exists yet. Shared by build_image_cache() and
    ImageWindowedDataset.__init__'s own validity check — both need to agree
    on exactly the same notion of "is split X still validly cached."
    """
    meta_path = _cache_meta_path()
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache_meta(meta: dict) -> None:
    _cache_meta_path().write_text(json.dumps(meta, indent=2))


def build_image_cache(splits: tuple[str, ...] = LOGICAL_SPLITS, workers: int = IMAGE_CACHE_BUILD_WORKERS) -> None:
    """Explicit preprocessing pass — see this module's docstring for the
    full "why" (moves decode+pad+resize out of the per-epoch hot path).

    Reuses config.IMAGE_RESIZED_CACHE_DIR across calls/runs when still
    valid: a split is skipped (not rebuilt) if _cache_meta.json already
    records it as done under the CURRENT _current_cache_key() (IMAGE_SIZE +
    IMAGE_MIN_CROP_SIZE — the only settings that affect what a cached file
    actually contains). If the key on disk doesn't match the current one
    (either setting changed since the cache was last built), the whole
    cache is wiped and rebuilt from scratch for every requested split —
    a stale cache built under different settings is never partially reused.
    Reversed 2026-08-01 from this project's initial "always rebuild fresh
    every run" choice — see this module's docstring for why.

    Always preprocesses all 6 combined sources (UniTalk-ASD + the 5 in
    _CACHE_EXTRA_SOURCE_FNS) for every split actually built — this
    project's "all_combined" train/evaluate entry points always use exactly
    this dataset, so there's no per-instance extra_sources param here the
    way ImageWindowedDataset itself has one.

    Uses ProcessPoolExecutor, not ThreadPoolExecutor — see
    config.IMAGE_CACHE_BUILD_WORKERS' comment for the measured reason
    (threading this specific decode/resize/encode work is SLOWER than
    single-threaded, not faster; Pillow's resize/save calls don't release
    the GIL well enough here).
    """
    current_key = _current_cache_key()
    meta = load_cache_meta()
    if meta is not None and meta.get("key") == current_key:
        done_splits = set(meta.get("done_splits", []))
    else:
        # No cache yet, or it was built under different settings — a
        # partially-matching reuse isn't safe (e.g. IMAGE_SIZE changed:
        # old files are the WRONG size), so start completely fresh.
        reset_dir(IMAGE_RESIZED_CACHE_DIR)
        done_splits = set()
        meta = {"key": current_key, "done_splits": []}

    splits_to_build = [s for s in splits if s not in done_splits]
    already_reused = [s for s in splits if s in done_splits]
    if already_reused:
        print(f"image cache: reusing already-cached splits {already_reused} (key={current_key}) -> {IMAGE_RESIZED_CACHE_DIR}")
    if not splits_to_build:
        print(f"image cache: all requested splits {list(splits)} already cached and valid, nothing to build")
        return

    t_start = time.time()
    for split in splits_to_build:
        raw_split = _raw_split_for(split)
        default_csv = split_csv_path(_default_processed_dir(), split)
        sources = [(default_csv, "unitalk")] + [fn(split) for fn in _CACHE_EXTRA_SOURCE_FNS]

        raw_paths: list[Path] = []
        cache_paths: list[Path] = []
        for csv_path, source_label in sources:
            clips_videos_dir, ext_override = _IMAGE_SOURCES[source_label]
            df = _load_frame_table(csv_path)
            for person_id, group in df.groupby("person_id", sort=False):
                video_id = group["video_id"].iloc[0]
                image_names = group["image_name"].to_numpy()
                out_dir = resolve_cache_path(source_label, raw_split, video_id, person_id, image_names[0]).parent
                out_dir.mkdir(parents=True, exist_ok=True)
                for name in image_names:
                    raw_paths.append(_resolve_image_path(clips_videos_dir, raw_split, video_id, person_id, name, ext_override))
                    cache_paths.append(resolve_cache_path(source_label, raw_split, video_id, person_id, name))

        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_process_and_cache_frame, raw_paths, cache_paths, chunksize=100))
        n_written = results.count("written")
        n_too_small = results.count("too_small")
        n_missing = results.count("missing")
        print(
            f"[{split}] image cache: {len(results)} frames processed in {time.time() - t0:.1f}s "
            f"({n_written} written, {n_too_small} too-small (IMAGE_MIN_CROP_SIZE={IMAGE_MIN_CROP_SIZE}), "
            f"{n_missing} missing source files)"
        )
        # Saved after EACH split (not once at the very end) so a build
        # interrupted partway through still leaves already-finished splits
        # correctly marked as reusable, instead of losing that progress.
        done_splits.add(split)
        meta["done_splits"] = sorted(done_splits)
        _save_cache_meta(meta)
    print(f"image cache build complete in {time.time() - t_start:.1f}s -> {IMAGE_RESIZED_CACHE_DIR}")


def _sample_augment_params() -> dict:
    """One fresh random draw — used for every frame in a single window (see
    module/config comment on config.AUGMENT_IMAGES for why the SAME draw is
    reused across a whole window rather than resampled per frame).
    """
    return {
        "angle": float(np.random.uniform(-IMAGE_AUGMENT_ROTATION_MAX_DEGREES, IMAGE_AUGMENT_ROTATION_MAX_DEGREES)),
        "scale": float(np.random.uniform(*IMAGE_AUGMENT_SCALE_RANGE)),
        "brightness": float(np.random.uniform(1 - IMAGE_AUGMENT_BRIGHTNESS, 1 + IMAGE_AUGMENT_BRIGHTNESS)),
        "contrast": float(np.random.uniform(1 - IMAGE_AUGMENT_CONTRAST, 1 + IMAGE_AUGMENT_CONTRAST)),
    }


def _apply_augment(img: Image.Image, params: dict) -> Image.Image:
    """Applies one (angle, scale, brightness, contrast) draw to a single
    already-square, already-resized PIL image. Rotation is around the
    center, no canvas resize (`expand=False`) — corners left empty by the
    rotation are filled black, same convention as this dataset's other
    zero-padding. Scale zooms in (center-crop back to size) or out
    (pads back to size with black) rather than resizing the canvas, so
    every frame stays IMAGE_SIZE x IMAGE_SIZE going into ToTensor/Normalize.
    """
    w, h = img.size
    img = img.rotate(params["angle"], resample=Image.BILINEAR)

    scale = params["scale"]
    if scale != 1.0:
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        if scale >= 1.0:
            left, top = (new_w - w) // 2, (new_h - h) // 2
            img = img.crop((left, top, left + w, top + h))
        else:
            canvas = Image.new("RGB", (w, h))
            canvas.paste(img, ((w - new_w) // 2, (h - new_h) // 2))
            img = canvas

    img = ImageEnhance.Brightness(img).enhance(params["brightness"])
    img = ImageEnhance.Contrast(img).enhance(params["contrast"])
    return img


# config.IMAGE_AUGMENT_SHIFT_SECONDS converted to frames, PLUS half a
# WINDOW_TRAIN_STRIDE — the added half-stride guarantees this is never a
# whole multiple of WINDOW_TRAIN_STRIDE itself, so shifting by this amount
# can never land exactly back on the grid real windows already occupy (see
# that config comment for the full "why" — a plain whole-stride-multiple
# shift would silently reproduce an already-included real window verbatim).
_SHIFT_FRAMES = round(IMAGE_AUGMENT_SHIFT_SECONDS * DATASET_FPS) + WINDOW_TRAIN_STRIDE // 2


def _choose_shifted_start(start: int, window_size: int, track_len: int, real_starts: set[int]) -> tuple[int, bool]:
    """Picks a start position for a materialized augmented copy's SOURCE
    frames: `start` shifted by `_SHIFT_FRAMES`, so the copy's underlying
    footage is continuous but genuinely different from the real window it's
    a copy of. Tries forward first, then backward; falls back to `start`
    itself (same frames — only the pixel transform differs) if neither
    fits within the track or would land exactly on another already-
    enumerated real window's start (both blocked only for tracks with
    little to no room to shift in, e.g. near `window_size` long). Returns
    (chosen_start, was_shifted) — `was_shifted` only for the summary counts
    printed at the end of __init__.
    """
    forward = start + _SHIFT_FRAMES
    if forward + window_size <= track_len and forward not in real_starts:
        return forward, True
    backward = start - _SHIFT_FRAMES
    if backward >= 0 and backward not in real_starts:
        return backward, True
    return start, False


class ImageWindowedDataset(Dataset):
    """One sample = one fixed-length (config.WINDOW_SIZE) window of face-crop
    images from a single track, majority-vote label over every real frame in
    that window (see module docstring for why this majority vote is
    unmasked, unlike WindowedLandmarkSequenceDataset's). Exposes
    `.labels`/`.source_labels` in the same shape WindowedLandmarkSequenceDataset
    does, so WindowedStratifiedBatchSampler/WindowedSourceBalancedBatchSampler
    work unmodified.
    """

    def __init__(
        self,
        split: str,
        window_size: int = WINDOW_SIZE,
        window_stride: int = WINDOW_STRIDE,
        pad_short_tracks: bool = WINDOW_PAD_SHORT_TRACKS,
        pad_min_length_ratio: float = WINDOW_PAD_MIN_LENGTH_RATIO,
        extra_sources: list[tuple[Path, str]] = (),
        sample_size_per_source: int | None = None,
        sample_seed: int = IMAGE_WINDOWED_SEED,
    ):
        """extra_sources: additional (csv_path, source_label) pairs, same
        shape as windowed_dataset.py's — e.g. pass dataset.ava_source(split)
        to add AVA-ActiveSpeaker alongside the default UniTalk-ASD source.
        `source_label` must be a key of `_IMAGE_SOURCES` above (every
        combined-dataset source function in dataset.py already returns one
        of those labels).

        `sample_size_per_source` (added 2026-08-23, explicit user request —
        a quick "train test"/dry-run mode): None (default) uses every real
        window this split/sources combination produces, unchanged from
        before this param existed. A number N instead means: after building
        the FULL window list as normal, randomly keep at most N windows per
        source (seeded via `sample_seed`, without replacement) — e.g. N=500
        across 6 sources gives 3,000 total windows instead of the full
        ~488K. Does NOT reduce how long __init__ itself takes (every
        source's CSV is still loaded and windowed in full first — the
        selection happens after), only how many windows the resulting
        dataset/dataloader actually iterates over.

        The N windows kept per source are CLASS-BALANCED (added 2026-08-23,
        explicit user request), not a plain random draw across both
        classes: aims for N//2 SPEAKING_AUDIBLE + N//2 NOT_SPEAKING per
        source (e.g. N=500 -> 250/250). A source short on one class
        donates that shortfall to the other class instead of leaving the
        source under N (e.g. a source that's purely one class just gets N
        windows of that class) — see `_subsample_per_source`'s docstring.
        Logged whenever a source can't reach the full N across both
        classes combined, so a real shortfall is visible rather than
        silent.
        """
        if split not in LOGICAL_SPLITS:
            raise ValueError(f"split must be one of {LOGICAL_SPLITS}, got {split!r}")
        # Checks the SPECIFIC split against build_image_cache()'s own
        # metadata (not just "does the cache directory have SOMETHING in
        # it") — a blind non-empty check would silently pass for a cache
        # that has e.g. only "test" built, then this "train" construction
        # would find every frame_ok False and every window get dropped,
        # producing a silently EMPTY dataset instead of a clear error.
        cache_meta = load_cache_meta()
        if (
            cache_meta is None
            or cache_meta.get("key") != _current_cache_key()
            or split not in cache_meta.get("done_splits", [])
        ):
            raise FileNotFoundError(
                f"{IMAGE_RESIZED_CACHE_DIR} has no valid cache for split={split!r} under the "
                f"current IMAGE_SIZE/IMAGE_MIN_CROP_SIZE settings — run build_image_cache() first "
                f"(scripts/features/build_image_resized_cache.py standalone, or "
                f"scripts/modeling/train/image_windowed_all_combined.py, which calls it "
                f"automatically at the start of every run — evaluate scripts do NOT rebuild it "
                f"themselves, they expect a training run's cache to still be on disk)"
            )
        self.split = split
        self.window_size = window_size
        self.raw_split = _raw_split_for(split)
        # config.AUGMENT_IMAGES, train-split only — see that config comment
        # and _sample_augment_params()/_apply_augment() above for what gets
        # applied. Checked once here (not re-derived in __getitem__), same
        # convention as WindowedLandmarkSequenceDataset.augment_on_the_fly/
        # .materialize below.
        augment = AUGMENT_IMAGES and split == "train"
        self.materialize = augment and not IMAGE_AUGMENT_ON_THE_FLY
        self.augment_on_the_fly = augment and IMAGE_AUGMENT_ON_THE_FLY

        self.window_paths: list[list[str | None]] = []  # each: window_size entries (plain strings, see resolve_cache_path_str()), None = padding
        self.window_augment_params: list[dict | None] = []  # materialized-copy frozen draw, else None
        self.lengths: list[int] = []
        self.labels: list[int] = []
        self.person_ids: list[str] = []
        self.source_labels: list[str] = []

        default_csv = split_csv_path(_default_processed_dir(), split)
        all_sources = [(default_csv, "unitalk"), *extra_sources]

        min_length_to_pad = window_size * pad_min_length_ratio
        n_dropped_short_track = 0
        n_padded_short_track = 0
        n_materialized_shifted = 0
        n_materialized_fallback = 0
        n_dropped_masked = 0

        def append_window(paths_slice, length, label, person_id, source_label, augment_params=None):
            self.window_paths.append(paths_slice)
            self.window_augment_params.append(augment_params)
            self.lengths.append(length)
            self.labels.append(label)
            self.person_ids.append(person_id)
            self.source_labels.append(source_label)

        # config.IMAGE_MAX_MASKED_FRAME_RATIO — a window is dropped (not
        # appended at all) if more than this fraction of its WINDOW_SIZE
        # frames would come back zero-masked at __getitem__ time (missing
        # from the cache — either the original file was missing, or
        # build_image_cache() dropped it for being too small — or
        # short-track padding). Checked via ONE big Path.exists() sweep
        # across every track in this split combined (see the two-pass
        # structure below), not lazily, since a batch sampler needs a
        # fixed, final window count up front. `masked_ratio()` below only
        # counts real (non-padding) frames the check found missing from the
        # cache; padding frames are counted separately by the caller, since
        # `frame_ok` only ever covers a track's real entries.
        check_masking = IMAGE_MAX_MASKED_FRAME_RATIO < 1.0

        def masked_ratio(frame_ok: list[bool] | None, start: int, end: int, n_padding: int = 0) -> float:
            n_masked = n_padding
            if frame_ok is not None:
                n_masked += sum(1 for ok in frame_ok[start:end] if not ok)
            return n_masked / window_size

        # Pass 1/2 (single-threaded): build every track's path list and
        # collect them all into ONE flat list, deferring the actual
        # existence check. Then ONE executor.map() call checks the WHOLE
        # split's real frames at once. Measured 2026-08-01: calling
        # executor.map() separately PER TRACK (~10K tracks in just the
        # smallest split) cost 220s of dispatch overhead alone — the actual
        # Path.exists() work was never the bottleneck, the per-call
        # overhead of ~10K separate ThreadPoolExecutor.map() invocations
        # was. Batching into one call eliminates that overhead entirely.
        track_records: list[tuple] = []  # (person_id, source_label, paths, labels, track_len, path_offset)
        all_paths: list[str] = []
        for csv_path, source_label in all_sources:
            df = _load_frame_table(csv_path)
            for person_id, group in df.groupby("person_id", sort=False):
                video_id = group["video_id"].iloc[0]
                image_names = group["image_name"].to_numpy()
                labels = group["target_index"].to_numpy(dtype=np.int64)
                track_len = len(group)
                # Plain strings, not Path objects — see resolve_cache_path_str()'s
                # docstring for the measured pathlib overhead this avoids at
                # this project's per-frame scale (millions of calls).
                paths = [
                    resolve_cache_path_str(source_label, self.raw_split, video_id, person_id, name)
                    for name in image_names
                ]
                path_offset = len(all_paths)
                all_paths.extend(paths)
                track_records.append((person_id, source_label, paths, labels, track_len, path_offset))

        if check_masking:
            # Manually chunked (NOT executor.map(fn, all_paths) directly) —
            # measured 2026-08-01: even ONE big executor.map() call still
            # submits one Future PER ITEM (ThreadPoolExecutor.map()'s
            # `chunksize` param is a no-op for thread pools, unlike process
            # pools — see the stdlib docs), so a ~1.56M-item split only
            # dropped construction from 222.9s to 182.2s, not the
            # near-elimination expected from fixing the per-track dispatch
            # problem alone. Splitting into a small number of large chunks
            # (each chunk's Path.exists() calls run in a tight loop on one
            # worker thread — still real parallelism, since each syscall
            # releases the GIL) cuts submitted Futures from ~1.56M to
            # roughly IMAGE_FRAME_CHECK_WORKERS x 4.
            n = len(all_paths)
            chunk_size = max(1, math.ceil(n / (IMAGE_FRAME_CHECK_WORKERS * 4)))
            chunks = [all_paths[i:i + chunk_size] for i in range(0, n, chunk_size)]
            with ThreadPoolExecutor(max_workers=IMAGE_FRAME_CHECK_WORKERS) as executor:
                chunk_results = list(executor.map(_check_paths_exist, chunks))
            all_frame_ok = [ok for chunk in chunk_results for ok in chunk]
        else:
            all_frame_ok = None

        # Pass 3: build windows per track, reusing the precomputed
        # existence results (a plain list slice — no more per-track
        # executor calls).
        for person_id, source_label, paths, labels, track_len, path_offset in track_records:
            frame_ok = all_frame_ok[path_offset:path_offset + track_len] if all_frame_ok is not None else None
            real_starts: list[int] = []  # this track's own real-window start positions only

            if track_len < window_size:
                if not pad_short_tracks or track_len < min_length_to_pad:
                    n_dropped_short_track += 1
                    continue
                if masked_ratio(frame_ok, 0, track_len, window_size - track_len) > IMAGE_MAX_MASKED_FRAME_RATIO:
                    n_dropped_masked += 1
                    continue
                window_label = int(labels.mean() > 0.5)
                padded_paths: list[str | None] = list(paths) + [None] * (window_size - track_len)
                append_window(padded_paths, track_len, window_label, person_id, source_label)
                real_starts.append(0)
                n_padded_short_track += 1
            else:
                for start in range(0, track_len - window_size + 1, window_stride):
                    end = start + window_size
                    if masked_ratio(frame_ok, start, end) > IMAGE_MAX_MASKED_FRAME_RATIO:
                        n_dropped_masked += 1
                        continue
                    window_label = int(labels[start:end].mean() > 0.5)  # majority vote, ties -> 0 (NOT_SPEAKING)
                    append_window(paths[start:end], window_size, window_label, person_id, source_label)
                    real_starts.append(start)

            # Materialized augmented copies (config.IMAGE_AUGMENT_SHIFT_SECONDS's
            # comment has the full "why a shift, not a same-frame
            # duplicate" rationale) — built right here, per track, while
            # `paths`/`labels`/`track_len`/`real_starts` are still in
            # scope, one copy per real window per IMAGE_AUGMENT_COPIES_PER_WINDOW.
            if self.materialize and real_starts:
                real_starts_set = set(real_starts)
                for start in real_starts:
                    for _ in range(IMAGE_AUGMENT_COPIES_PER_WINDOW):
                        if track_len < window_size:
                            # Padded short track — the single window already
                            # spans the whole track, so there's no room to
                            # shift into at all; the copy differs only by
                            # its (frozen) pixel transform.
                            copy_paths = list(paths) + [None] * (window_size - track_len)
                            copy_label = int(labels.mean() > 0.5)
                            copy_length = track_len
                            n_materialized_fallback += 1
                        else:
                            copy_start, shifted = _choose_shifted_start(start, window_size, track_len, real_starts_set)
                            copy_end = copy_start + window_size
                            if masked_ratio(frame_ok, copy_start, copy_end) > IMAGE_MAX_MASKED_FRAME_RATIO:
                                n_dropped_masked += 1
                                continue
                            copy_paths = paths[copy_start:copy_end]
                            copy_label = int(labels[copy_start:copy_end].mean() > 0.5)
                            copy_length = window_size
                            n_materialized_shifted += int(shifted)
                            n_materialized_fallback += int(not shifted)
                        append_window(
                            copy_paths, copy_length, copy_label, person_id, source_label,
                            augment_params=_sample_augment_params(),
                        )

        n_kept = len(self.labels)
        n_speaking = int(sum(self.labels))
        source_tag = split if not extra_sources else f"{split}+{'+'.join(s for _, s in extra_sources)}"
        print(
            f"[{source_tag}] ImageWindowedDataset: {n_kept} windows kept "
            f"({n_speaking} speaking / {n_kept - n_speaking} not-speaking), "
            f"{n_padded_short_track} short tracks padded into one window each "
            f"(>= {min_length_to_pad:.0f}/{window_size} real frames required), "
            f"{n_dropped_short_track} tracks dropped as too short "
            f"(window_size={window_size}, pad_short_tracks={pad_short_tracks}, "
            f"pad_min_length_ratio={pad_min_length_ratio}), "
            f"{n_dropped_masked} windows dropped for exceeding "
            f"IMAGE_MAX_MASKED_FRAME_RATIO={IMAGE_MAX_MASKED_FRAME_RATIO} "
            f"(missing/too-small/padded frames, IMAGE_MIN_CROP_SIZE={IMAGE_MIN_CROP_SIZE})"
            + (
                f", {n_materialized_shifted} materialized copies used shifted (genuinely different) "
                f"frames, {n_materialized_fallback} fell back to same-frames (pixel-transform-only, "
                f"track too short to shift within)"
                if self.materialize else ""
            )
        )

        if sample_size_per_source is not None:
            self._subsample_per_source(sample_size_per_source, sample_seed)

        # PIL-based path — only used when augment_params is not None (see
        # _load_image), since _apply_augment's rotate/resize/enhance calls
        # need a PIL Image, not a raw tensor.
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD),
        ])
        # Tensor-based mean/std for the fast (non-augmented) decode path —
        # see _load_image's docstring for why this is measurably faster.
        self._normalize_mean = torch.tensor(IMAGE_NORMALIZE_MEAN).view(3, 1, 1)
        self._normalize_std = torch.tensor(IMAGE_NORMALIZE_STD).view(3, 1, 1)
        self._n_missing_files = 0

    def _subsample_per_source(self, n_per_source: int, seed: int) -> None:
        """Deterministically (seeded) keeps at most `n_per_source` random
        windows per source, class-balanced, in place, across every list
        this dataset indexes by window (window_paths/window_augment_params/
        lengths/labels/person_ids/source_labels) — see __init__'s docstring
        for the full rationale. Called once, after the full window list is
        built, never re-derives which sources exist beyond what
        self.source_labels already contains.

        Reuses windowed_dataset._water_fill (the same "distribute a target
        as evenly as possible across capped capacities, redistributing any
        shortfall to whichever capacity still has room" routine
        WindowedSourceBalancedBatchSampler already uses) with target=
        n_per_source across capacities={0: <NOT_SPEAKING count>, 1:
        <SPEAKING_AUDIBLE count>} for that source: two classes water-fills
        down to an even n_per_source//2 / n_per_source//2 split when both
        have enough, but a class short on supply donates its shortfall to
        the other rather than leaving the source under n_per_source when
        the other class could cover it (e.g. a purely single-class source
        just gets n_per_source windows of that one class).
        """
        rng = random.Random(seed)
        indices_by_source: dict[str, list[int]] = {}
        for i, source in enumerate(self.source_labels):
            indices_by_source.setdefault(source, []).append(i)

        selected: list[int] = []
        for source, indices in indices_by_source.items():
            indices_by_label: dict[int, list[int]] = {}
            for i in indices:
                indices_by_label.setdefault(self.labels[i], []).append(i)
            capacities = {label: len(idxs) for label, idxs in indices_by_label.items()}
            allocation = _water_fill(n_per_source, capacities)
            total = sum(allocation.values())
            if total < n_per_source:
                print(
                    f"[{self.split}] sample_size_per_source={n_per_source} but source={source!r} "
                    f"only has {total} windows available across both classes — using all {total} "
                    f"of them (class split: {allocation})"
                )
            for label, count in allocation.items():
                selected.extend(rng.sample(indices_by_label[label], count))
        selected.sort()  # stable, deterministic order rather than shuffled-by-dict-iteration

        self.window_paths = [self.window_paths[i] for i in selected]
        self.window_augment_params = [self.window_augment_params[i] for i in selected]
        self.lengths = [self.lengths[i] for i in selected]
        self.labels = [self.labels[i] for i in selected]
        self.person_ids = [self.person_ids[i] for i in selected]
        self.source_labels = [self.source_labels[i] for i in selected]
        print(f"[{self.split}] sample_size_per_source={n_per_source}: subsampled to {len(selected)} windows total")

    def _load_image(self, path: str | None, augment_params: dict | None = None) -> torch.Tensor:
        """`path` (when not None) is a plain string (resolve_cache_path_str())
        always pointing into config.IMAGE_RESIZED_CACHE_DIR — already
        pad-to-square'd, resized to IMAGE_SIZE, and RGB (build_image_cache()'s
        _process_and_cache_frame did img.convert("RGB") once, at build time,
        before saving), so no pad/resize/too-small-check/convert work
        happens here anymore, just decode + optional augment + normalize.

        Two decode paths, chosen by whether augmentation is needed:
        - augment_params is None (the common case — AUGMENT_IMAGES is off
          by default): torchvision.io.decode_jpeg (libjpeg-turbo backend),
          straight to a tensor. Measured 2026-08-01 on this project's own
          cached 96x96 JPEGs: ~1.8x faster than the PIL path below
          (4041 vs 7293 img/s on a 5000-file sample) — worth the dual-path
          complexity given this runs on every single frame of every window,
          every epoch.
        - augment_params is not None: PIL-based (Image.open + self._transform),
          since _apply_augment's rotate/resize/ImageEnhance calls need a
          real PIL Image, not a raw tensor — no fast path exists for this
          branch.

        A missing cache file (build_image_cache() never wrote one — original
        was missing OR too small) falls back to a zero tensor either way,
        same convention as a genuinely padded window position.
        """
        if path is None:
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        if augment_params is None:
            try:
                data = tv_io.read_file(path)
                img = tv_io.decode_jpeg(data, mode=tv_io.ImageReadMode.RGB)
                return img.float().div_(255.0).sub_(self._normalize_mean).div_(self._normalize_std)
            except (FileNotFoundError, OSError, RuntimeError):
                self._n_missing_files += 1
                return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        try:
            with Image.open(path) as img:
                img = _apply_augment(img, augment_params)
                return self._transform(img)
        except (FileNotFoundError, OSError):
            self._n_missing_files += 1
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        # On-the-fly mode: fresh draw every access (None -> no-op for real
        # windows, since self.augment_on_the_fly is only True on the train
        # split when config.AUGMENT_IMAGES is on). Materialized mode: the
        # frozen draw sampled once at construction time (self.window_augment_params[idx]),
        # None for the real (non-copy) windows sitting alongside the copies.
        # Reused identically for every frame in the window either way (see
        # _sample_augment_params' docstring).
        if self.augment_on_the_fly:
            augment_params = _sample_augment_params()
        else:
            augment_params = self.window_augment_params[idx]
        images = torch.stack([self._load_image(p, augment_params) for p in self.window_paths[idx]])  # (window_size, 3, H, W)
        return {
            "features": images,
            "length": self.lengths[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "person_id": self.person_ids[idx],
            "source_label": self.source_labels[idx],
        }


def _default_processed_dir() -> Path:
    import config
    return config.DATA_PROCESSED_DIR


def collate_images(batch: list[dict]) -> dict:
    return {
        "features": torch.stack([b["features"] for b in batch]),          # (B, window_size, 3, H, W)
        "lengths": torch.tensor([b["length"] for b in batch], dtype=torch.long),  # (B,)
        "labels": torch.stack([b["label"] for b in batch]),                # (B,)
        "person_id": [b["person_id"] for b in batch],
        "source_label": [b["source_label"] for b in batch],
    }


def get_image_windowed_dataloader(
    split: str,
    batch_size: int = IMAGE_BATCH_SIZE,
    shuffle: bool | None = None,
    num_workers: int = 0,
    extra_sources: list[tuple[Path, str]] = (),
    sample_size_per_source: int | None = None,
) -> DataLoader:
    """Same balancing/shuffling rules as windowed_dataset.get_windowed_dataloader
    (see that function's docstring) — BALANCED_SOURCE_BATCHES/BALANCED_BATCHES
    apply identically, reusing the exact same sampler classes. `num_workers`
    matters far more here than for the landmark pipeline: each __getitem__
    call still decodes window_size real (already pre-resized by
    build_image_cache()) image files from disk (not just an in-RAM array
    lookup), so a real multi-process value (e.g. 4-8) is worth setting
    explicitly when calling this, unlike the landmark pipeline's default of
    0 — even though decoding an already-IMAGE_SIZE JPEG is far cheaper than
    the original pre-cache pipeline's decode+pad+resize of the full-size
    source image.

    "train" uses config.WINDOW_TRAIN_STRIDE (overlapping windows); "val"/
    "test" always use the non-overlapping config.WINDOW_STRIDE — same
    train-only rationale as the landmark pipeline (see config.WINDOW_TRAIN_STRIDE).

    `num_workers > 0` also sets `persistent_workers=True` and
    `pin_memory=True` — the default DataLoader behavior (persistent_workers
    off) tears down and re-spawns every worker process (re-importing torch/
    torchvision/PIL in each one) on every single `for batch in loader:` call,
    i.e. every epoch, for both the train AND val loaders — real, needless
    overhead measured directly on a real run (2026-07-31): epoch time didn't
    match pure GPU-compute time even after fixing the earlier CPU-decode
    bottleneck (IMAGE_NUM_DATALOADER_WORKERS 8->64), pointing at per-epoch
    worker respawn + un-pinned host->GPU transfer as the next-biggest cost.
    persistent_workers keeps the same worker pool alive across the whole
    training run instead.

    `sample_size_per_source`: passed straight through to
    ImageWindowedDataset — see that class's __init__ docstring.
    """
    window_stride = WINDOW_TRAIN_STRIDE if split == "train" else WINDOW_STRIDE
    dataset = ImageWindowedDataset(
        split, window_stride=window_stride, extra_sources=extra_sources,
        sample_size_per_source=sample_size_per_source,
    )
    if shuffle is None:
        shuffle = split == "train"
    loader_kwargs = dict(
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=num_workers > 0,
    )
    if BALANCED_SOURCE_BATCHES and split == "train":
        batch_sampler = WindowedSourceBalancedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_images, **loader_kwargs)
    if BALANCED_BATCHES:
        batch_sampler = WindowedStratifiedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_images, **loader_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_images,
        **loader_kwargs,
    )


if __name__ == "__main__":
    build_image_cache()
    for split in LOGICAL_SPLITS:
        loader = get_image_windowed_dataloader(split)
        batch = next(iter(loader))
        print(
            f"[{split}] batch: features={tuple(batch['features'].shape)} "
            f"labels={tuple(batch['labels'].shape)} mean(labels)={batch['labels'].float().mean():.3f}"
        )
