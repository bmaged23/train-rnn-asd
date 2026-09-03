"""Produces a face-only-cropped MIRROR of config.SPEAKINGFACES_RAW_CLIPS_VIDEOS_DIR
at config.SPEAKINGFACES_RAW_CLIPS_VIDEOS_CROPPED_DIR — for the candidate
raw-pixel CNN+LSTM approach (see [[project-asd-windowed-sampler-and-augmentation]]
in project memory), which needs consistent face framing across all 6
sources unlike the existing landmark pipeline (MediaPipe does its own
detection + normalizes away framing differences, so it never needed this —
see config.SPEAKINGFACES_RAW_CLIPS_VIDEOS_CROPPED_DIR's comment).

NEVER touches SPEAKINGFACES_RAW_CLIPS_VIDEOS_DIR itself — reads from it,
writes an entirely separate directory tree, same shape
(<split>/<video_id>/<person_id>/<frame_timestamp>.jpg).

Per track (one person_id folder = one continuous 804-frame, single-subject
capture — no multi-person ambiguity, unlike download_avspeech_subset.py's
YouTube clips): batched RetinaFace detection over every frame at once (same
batch_face usage as download_avspeech_subset.py's track_target_face), then
nearest-center tracking frame-to-frame. Deliberately more lenient than that
script's "break on any lost/missing detection" — SpeakingFaces has exactly
one real candidate face per frame by construction (a single subject, fixed
camera), so there's no risk of drifting onto a different person the way a
YouTube clip has; a frame with zero detected faces is just skipped (kept
missing, like an undetected frame elsewhere in this project) rather than
aborting the rest of the track and losing everything after it.

Resumable: a track is only appended to config.SPEAKINGFACES_RAW_CLIPS_VIDEOS_CROPPED_DIR's
sibling completed-tracks manifest once every one of its frames is durably
written, so a rerun just skips already-done tracks.

Usage:
    python scripts/dataset/crop_speakingfaces_faces.py
    python scripts/dataset/crop_speakingfaces_faces.py --limit 5   # smoke test: 5 tracks
"""
from __future__ import annotations

import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch
from batch_face import RetinaFace
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (
    NUM_SPEAKINGFACES_CROP_WORKERS,
    RETINAFACE_MAX_TRACK_DISTANCE_PX,
    SPEAKINGFACES_CROP_GPU_ID,
    SPEAKINGFACES_FACE_CROP_MARGIN,
    SPEAKINGFACES_RAW_CLIPS_VIDEOS_CROPPED_DIR,
    SPEAKINGFACES_RAW_CLIPS_VIDEOS_DIR,
    SPLITS,
)

_MANIFEST_PATH = SPEAKINGFACES_RAW_CLIPS_VIDEOS_CROPPED_DIR / "completed_tracks.txt"

# One RetinaFace instance PER WORKER THREAD (lazily created on that thread's
# first task, then reused for every subsequent track it picks up) — see
# config.SPEAKINGFACES_CROP_GPU_ID's comment for the two GPU-concurrency
# attempts that failed before landing on CPU (gpu_id=-1) here, which
# sidesteps the NVML crash entirely by never touching CUDA at all.
_thread_local = threading.local()


def get_thread_detector() -> RetinaFace:
    if not hasattr(_thread_local, "detector"):
        # PyTorch's CPU backend fans a single op out across ALL cores by
        # default (torch.get_num_threads()) — with NUM_SPEAKINGFACES_CROP_WORKERS
        # threads each independently doing that, total oversubscription
        # stalls everything (measured directly 2026-07-31: 4 workers took
        # >3min for 16 tracks, vs. ~15s/track expected). Cap each thread's
        # own intra-op parallelism so the total across all workers stays
        # within the real core count instead of each one fighting for all
        # of them.
        torch.set_num_threads(max(1, (os.cpu_count() or 1) // NUM_SPEAKINGFACES_CROP_WORKERS))
        _thread_local.detector = RetinaFace(gpu_id=SPEAKINGFACES_CROP_GPU_ID)
    return _thread_local.detector


def load_completed() -> set[str]:
    if not _MANIFEST_PATH.exists():
        return set()
    return {line.strip() for line in _MANIFEST_PATH.read_text().splitlines() if line.strip()}


def track_faces(detector: RetinaFace, frame_paths: list[Path]) -> dict[int, tuple[float, float, float, float]]:
    """Returns {frame_index: (x1,y1,x2,y2)} for every frame where a face was
    found — anchored on the LARGEST detected face in the first frame that
    has any (no external position hint available, unlike AVSpeech's CSV),
    then nearest-previous-center tracking. A frame with zero faces is
    skipped (not fatal) — see module docstring for why that's safe here.
    """
    images = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in frame_paths]
    all_faces = detector(images, threshold=0.9)

    def center_of(box) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def area_of(box) -> float:
        x1, y1, x2, y2 = box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    tracked: dict[int, tuple[float, float, float, float]] = {}
    prev_center: tuple[float, float] | None = None
    for idx, faces in enumerate(all_faces):
        if not faces:
            continue  # skip this frame only, not the rest of the track

        if prev_center is None:
            best_box = max((box for box, _kps, _score in faces), key=area_of)
        else:
            best_box, best_dist = None, float("inf")
            for box, _kps, _score in faces:
                cx, cy = center_of(box)
                dist = ((cx - prev_center[0]) ** 2 + (cy - prev_center[1]) ** 2) ** 0.5
                if dist < best_dist:
                    best_box, best_dist = box, dist
            if best_dist > RETINAFACE_MAX_TRACK_DISTANCE_PX:
                continue  # this frame's nearest face is implausibly far — skip it, keep prev_center for the next frame

        tracked[idx] = tuple(best_box)
        prev_center = center_of(best_box)

    return tracked


def crop_track(track_dir: Path, out_dir: Path) -> tuple[int, int]:
    """Returns (n_cropped, n_missing). Uses this thread's own RetinaFace
    instance (see get_thread_detector()), not a shared one.
    """
    frame_paths = sorted(track_dir.glob("*.png"))
    if not frame_paths:
        return 0, 0
    tracked = track_faces(get_thread_detector(), frame_paths)

    out_dir.mkdir(parents=True, exist_ok=True)
    n_cropped = n_missing = 0
    for idx, path in enumerate(frame_paths):
        out_path = out_dir / f"{path.stem}.jpg"
        box = tracked.get(idx)
        if box is None:
            n_missing += 1
            continue
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        mx, my = bw * SPEAKINGFACES_FACE_CROP_MARGIN, bh * SPEAKINGFACES_FACE_CROP_MARGIN
        cx1, cy1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
        cx2, cy2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
        if cx2 <= cx1 or cy2 <= cy1 or not cv2.imwrite(str(out_path), img[cy1:cy2, cx1:cx2]):
            n_missing += 1
            continue
        n_cropped += 1
    return n_cropped, n_missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N tracks — smoke test")
    args = parser.parse_args()

    SPEAKINGFACES_RAW_CLIPS_VIDEOS_CROPPED_DIR.mkdir(parents=True, exist_ok=True)
    completed = load_completed()

    tracks: list[tuple[str, Path, Path]] = []  # (track_key, src_dir, out_dir)
    for split in SPLITS:
        split_dir = SPEAKINGFACES_RAW_CLIPS_VIDEOS_DIR / split
        for video_dir in sorted(split_dir.iterdir()) if split_dir.exists() else []:
            for track_dir in sorted(video_dir.iterdir()):
                track_key = f"{split}/{video_dir.name}/{track_dir.name}"
                if track_key in completed:
                    continue
                out_dir = SPEAKINGFACES_RAW_CLIPS_VIDEOS_CROPPED_DIR / split / video_dir.name / track_dir.name
                tracks.append((track_key, track_dir, out_dir))

    if args.limit is not None:
        tracks = tracks[: args.limit]
    print(f"{len(tracks)} tracks to process ({len(completed)} already done)")

    n_cropped_total = n_missing_total = 0
    # ThreadPoolExecutor, each thread lazily creating its OWN RetinaFace
    # instance via get_thread_detector() (see that function's comment for
    # why — testing whether this avoids the NVML crash a shared-instance
    # run hit at 8 workers). Manifest writes happen only in the main thread
    # as futures complete, never from within worker threads, so there's no
    # concurrent-write risk.
    with ThreadPoolExecutor(max_workers=NUM_SPEAKINGFACES_CROP_WORKERS) as pool, \
            open(_MANIFEST_PATH, "a") as manifest_file:
        futures = {pool.submit(crop_track, track_dir, out_dir): track_key for track_key, track_dir, out_dir in tracks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="tracks"):
            track_key = futures[future]
            n_cropped, n_missing = future.result()
            n_cropped_total += n_cropped
            n_missing_total += n_missing
            manifest_file.write(track_key + "\n")
            manifest_file.flush()

    print(f"{len(tracks)} tracks processed, {n_cropped_total} frames cropped, {n_missing_total} frames missing (no face found)")


if __name__ == "__main__":
    main()
