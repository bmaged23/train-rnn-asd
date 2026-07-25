"""Downloads the MediaPipe FaceLandmarker model into models/face_landmarker.task.

The project used to depend on a symlink at /data/bola/models/ pointing into
another user's directory (/data/Mona/attention/) — this fetches an
independent copy from Google's official MediaPipe model bucket instead, so
the project doesn't depend on anything outside itself. extract_landmarks.py
also calls download_if_missing() automatically the first time
PersonLandmarkExtractor needs the model, so this script is mainly for
explicit/manual re-fetching.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --force   # re-download even if already present
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import FACE_LANDMARKER_MODEL_PATH, FACE_LANDMARKER_MODEL_URL


def download_if_missing(force: bool = False) -> Path:
    if FACE_LANDMARKER_MODEL_PATH.exists() and not force:
        print(f"already present: {FACE_LANDMARKER_MODEL_PATH}")
        return FACE_LANDMARKER_MODEL_PATH

    FACE_LANDMARKER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = FACE_LANDMARKER_MODEL_PATH.with_suffix(".task.tmp")

    with tqdm(unit="B", unit_scale=True, desc="face_landmarker.task") as pbar:
        def report(block_num: int, block_size: int, total_size: int) -> None:
            if pbar.total is None and total_size > 0:
                pbar.total = total_size
            pbar.update(block_size)

        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, tmp_path, reporthook=report)

    tmp_path.rename(FACE_LANDMARKER_MODEL_PATH)
    print(f"downloaded -> {FACE_LANDMARKER_MODEL_PATH}")
    return FACE_LANDMARKER_MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if already present")
    args = parser.parse_args()
    download_if_missing(force=args.force)


if __name__ == "__main__":
    main()
