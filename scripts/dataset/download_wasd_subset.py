"""Downloads WASD (Wilder Active Speaker Detection, TBIOM 2025, CC-BY-SA-4.0)
and reshapes it into the same layout convention data/raw_ava/ already uses:
    data/raw_wasd/clips_videos/<split>/<video_id>/<entity_id>/<frame_timestamp:.2f>.jpg
    data/raw_wasd/csv/<split>_orig.csv
so scripts/features/process_dataset.py's DatasetProcessor can run over it unmodified,
just pointed at config.WASD_RAW_CLIPS_VIDEOS_DIR / config.WASD_RAW_CSV_DIR
(see scripts/features/process_wasd_dataset.py).

Unlike AVA, WASD's own train_orig.csv/val_orig.csv already ship with a header
and the exact CSV_COLUMNS column names/order, and a binary label_id (0/1, no
third class — confirmed against the real downloaded CSV), so they're copied
in unmodified here — no relabeling step like AVA_LABEL_ID_MAP is needed.

CSVs come from WASD's csv.zip on Google Drive (gdown, ~0.3GB, never hit any
quota). Video, however, does NOT come from WASD's own WASD_videos.zip
(~27GB, also Google Drive) — that hit Google's per-file daily download quota
on first attempt (2026-07-22, confirmed via the exact "too many users have
downloaded this file recently" response, not a transient/parsing error) and
Drive gives no way around it except waiting. Every CSV video_id is
"{youtube_id}_{start}-{end}" (start/end in whole seconds) — WASD's source
videos are ordinary YouTube uploads, so instead this script fetches each
annotated time range directly via yt-dlp. Verified against a real clip before
building this out: pulling from YouTube and cropping the CSV's entity_box at
its frame_timestamp landed squarely on the correct speaker's face at both a
SPEAKING_AUDIBLE and a NOT_SPEAKING timestamp for two different entities —
same alignment WASD's own zip would have given, just a different transport.

Pipeline per selected source (YouTube) video:
  1. One yt-dlp invocation requests every subclip needed from that video in
     a single call (repeated --download-sections, output named via yt-dlp's
     %(section_start)s/%(section_end)s) — resolves the video's manifest once
     instead of once per subclip (a source video has 3-50 subclips, 22.7 on
     average), which matters both for speed and for not hammering YouTube
     with repeat requests for the same video.
  2. For each subclip yt-dlp actually produced, opens one cv2.VideoCapture
     (reused across every entity in it) and crops each row's normalized
     entity_box out of the frame at that row's frame_timestamp (already
     subclip-relative, starting at 0 — confirmed against the real CSV).
  3. Deletes the temporary subclip immediately after — only the cropped
     JPGs + CSVs are meant to persist; data/raw_wasd/_staging/ is scratch
     space, removed once the whole run finishes (unless --keep-staging).
Source videos WASD's own team pulled from YouTube in 2023 may since have
gone private/deleted/re-uploaded under a new id — those are skipped and
counted, never guessed at or substituted.

No audio is downloaded/extracted — this project never trains on it.

Usage:
    python scripts/dataset/download_wasd_subset.py
    python scripts/dataset/download_wasd_subset.py --limit 3   # smoke test: 3 source videos
    python scripts/dataset/download_wasd_subset.py --keep-staging
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import gdown
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (
    CSV_COLUMNS,
    NUM_WASD_DOWNLOAD_WORKERS,
    WASD_CSV_GDRIVE_ID,
    WASD_RAW_CLIPS_VIDEOS_DIR,
    WASD_RAW_CSV_DIR,
    WASD_STAGING_DIR,
    WASD_SUBSET_VIDEOS,
    YOUTUBE_CLIP_END_PAD_SECONDS,
    YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS,
    YOUTUBE_FORMAT,
    YOUTUBE_URL_TEMPLATE,
)

_SELECTION_SEED = 42
_VIDEO_ID_RE = re.compile(r"^(?P<source>.+)_(?P<start>\d+)-(?P<end>\d+)$")


def download_csvs() -> dict[str, pd.DataFrame]:
    zip_path = WASD_STAGING_DIR / "WASD_csv.zip"
    extract_dir = WASD_STAGING_DIR / "csv_extracted"
    if not (extract_dir / "csv" / "train_orig.csv").exists():
        WASD_STAGING_DIR.mkdir(parents=True, exist_ok=True)
        gdown.download(id=WASD_CSV_GDRIVE_ID, output=str(zip_path), quiet=False, resume=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        zip_path.unlink()

    WASD_RAW_CSV_DIR.mkdir(parents=True, exist_ok=True)
    dfs: dict[str, pd.DataFrame] = {}
    for split in ("train", "val"):
        df = pd.read_csv(extract_dir / "csv" / f"{split}_orig.csv")
        if list(df.columns) != CSV_COLUMNS:
            raise ValueError(
                f"[{split}] WASD's {split}_orig.csv columns {list(df.columns)} don't match "
                f"config.CSV_COLUMNS {CSV_COLUMNS} — schema drifted, update this script before proceeding."
            )
        unknown_label_ids = sorted(set(df["label_id"].unique()) - {0, 1})
        if unknown_label_ids:
            raise ValueError(
                f"[{split}] WASD's {split}_orig.csv has unexpected label_id value(s) {unknown_label_ids} "
                "(expected only 0/1) — update this script's label handling before proceeding."
            )
        df.to_csv(WASD_RAW_CSV_DIR / f"{split}_orig.csv", index=False)
        dfs[split] = df
        print(f"[{split}] {len(df)} rows, {df['video_id'].nunique()} unique subclips")
    return dfs


def parse_video_id(video_id: str) -> tuple[str, int, int] | None:
    m = _VIDEO_ID_RE.match(video_id)
    if not m:
        return None
    return m.group("source"), int(m.group("start")), int(m.group("end"))


def fetch_subclips_for_source(source_id: str, specs: list[tuple[str, int, int]], out_dir: Path) -> dict[str, Path]:
    """specs: list of (video_id, start, end) all belonging to source_id. One
    yt-dlp call requests every section at once. Returns {video_id: path} only
    for sections yt-dlp actually produced — missing entries mean that section
    (or the whole video) failed/is unavailable, handled by the caller.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    padded_end = {video_id: end + YOUTUBE_CLIP_END_PAD_SECONDS for video_id, start, end in specs}

    cmd = ["yt-dlp", "--quiet", "--no-warnings"]
    for video_id, start, end in specs:
        cmd += ["--download-sections", f"*{start}-{padded_end[video_id]}"]
    cmd += [
        "-f", YOUTUBE_FORMAT,
        "-o", str(out_dir / f"{source_id}_%(section_start)d-%(section_end)d.%(ext)s"),
        YOUTUBE_URL_TEMPLATE.format(video_id=source_id),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=YOUTUBE_DOWNLOAD_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        pass  # fall through — map whatever sections did finish before the timeout

    return {
        video_id: p
        for video_id, start, end in specs
        if (p := out_dir / f"{source_id}_{start}-{padded_end[video_id]}.mp4").exists()
    }


def crop_entity_frame(cap: cv2.VideoCapture, row, out_path: Path) -> bool:
    if out_path.exists():
        return True  # already cropped — resumable across reruns
    cap.set(cv2.CAP_PROP_POS_MSEC, float(row.frame_timestamp) * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return False
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(row.entity_box_x1 * w)), max(0, int(row.entity_box_y1 * h))
    x2, y2 = min(w, int(row.entity_box_x2 * w)), min(h, int(row.entity_box_y2 * h))
    if x2 <= x1 or y2 <= y1:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), frame[y1:y2, x1:x2]))


Entry = tuple[str, pd.DataFrame, str, int, int]  # video_id, group, split, start, end


def process_source_video(source_id: str, entries: list[Entry]) -> tuple[int, int, int]:
    """Returns (n_cropped, n_failed, n_unfetched)."""
    out_paths_by_video: dict[str, list[Path]] = {}
    pending: list[Entry] = []
    n_already = 0
    for video_id, group, split, start, end in entries:
        paths = [
            WASD_RAW_CLIPS_VIDEOS_DIR / split / video_id / row.entity_id / f"{row.frame_timestamp:.2f}.jpg"
            for row in group.itertuples(index=False)
        ]
        out_paths_by_video[video_id] = paths
        if all(p.exists() for p in paths):
            n_already += len(paths)  # whole subclip already done — no need to refetch
        else:
            pending.append((video_id, group, split, start, end))

    if not pending:
        return n_already, 0, 0

    specs = [(video_id, start, end) for video_id, group, split, start, end in pending]
    subclip_paths = fetch_subclips_for_source(source_id, specs, WASD_STAGING_DIR / "subclips")

    n_cropped, n_failed, n_unfetched = n_already, 0, 0
    for video_id, group, split, start, end in pending:
        subclip_path = subclip_paths.get(video_id)
        if subclip_path is None:
            n_unfetched += len(group)
            continue
        try:
            cap = cv2.VideoCapture(str(subclip_path))
            try:
                for row, out_path in zip(group.itertuples(index=False), out_paths_by_video[video_id]):
                    if crop_entity_frame(cap, row, out_path):
                        n_cropped += 1
                    else:
                        n_failed += 1
            finally:
                cap.release()
        finally:
            subclip_path.unlink(missing_ok=True)
    return n_cropped, n_failed, n_unfetched


def process_all(dfs: dict[str, pd.DataFrame], limit: int | None) -> None:
    video_id_to_split: dict[str, str] = {}
    for split, df in dfs.items():
        for vid in df["video_id"].unique():
            if vid in video_id_to_split:
                raise ValueError(f"video_id {vid!r} appears in both {video_id_to_split[vid]!r} and {split!r} splits — ambiguous.")
            video_id_to_split[vid] = split

    entries_by_source: dict[str, list[Entry]] = defaultdict(list)
    skipped_bad_id = 0
    for split, df in dfs.items():
        for video_id, group in df.groupby("video_id"):
            parsed = parse_video_id(video_id)
            if parsed is None:
                skipped_bad_id += 1
                continue
            source_id, start, end = parsed
            entries_by_source[source_id].append((video_id, group, split, start, end))

    # Fixed-seed shuffle over source videos so a capped/--limit run samples a
    # representative spread of scenes rather than always the same
    # alphabetically-first source video's subclips.
    source_ids = list(entries_by_source)
    random.Random(_SELECTION_SEED).shuffle(source_ids)
    n_cap = limit if limit is not None else WASD_SUBSET_VIDEOS
    if n_cap is not None:
        source_ids = source_ids[:n_cap]

    (WASD_STAGING_DIR / "subclips").mkdir(parents=True, exist_ok=True)
    n_cropped_total = n_failed_total = n_unfetched_total = n_videos_unavailable = 0
    with ThreadPoolExecutor(max_workers=NUM_WASD_DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(process_source_video, sid, entries_by_source[sid]): sid for sid in source_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="source videos"):
            n_cropped, n_failed, n_unfetched = future.result()
            n_cropped_total += n_cropped
            n_failed_total += n_failed
            n_unfetched_total += n_unfetched
            if n_unfetched and not n_cropped and not n_failed:
                n_videos_unavailable += 1

    print(
        f"{len(source_ids)} source videos selected ({skipped_bad_id} subclips skipped — unparseable video_id), "
        f"{n_videos_unavailable} source videos entirely unavailable on YouTube, "
        f"{n_cropped_total} frames cropped, {n_failed_total} frames failed (decode/box error), "
        f"{n_unfetched_total} frames skipped (source video/section unfetchable)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WASD into data/raw_wasd/ (video pulled from YouTube via yt-dlp)")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only process the first N (shuffled) source videos — overrides config.WASD_SUBSET_VIDEOS, for a smoke test",
    )
    parser.add_argument("--keep-staging", action="store_true", help="skip deleting data/raw_wasd/_staging/ after cropping (debugging)")
    args = parser.parse_args()

    dfs = download_csvs()
    process_all(dfs, args.limit)

    if not args.keep_staging and WASD_STAGING_DIR.exists():
        shutil.rmtree(WASD_STAGING_DIR)
        print(f"removed staging dir {WASD_STAGING_DIR}")


if __name__ == "__main__":
    main()
