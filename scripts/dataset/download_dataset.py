"""
Download UniTalk-ASD from HuggingFace straight into data/raw/, matching the
csv/ + clips_videos/ + clips_audios/ layout the rest of the pipeline expects.

Video ids are discovered by listing the dataset repo itself (not a separate
video_list file), so this stays in sync with whatever HuggingFace is serving.

Usage:
    python scripts/dataset/download_dataset.py
    python scripts/dataset/download_dataset.py --splits train --limit 5   # smoke test
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (
    CSV_COLUMNS,
    HF_DATASET_REPO_ID,
    HF_REPO_TYPE,
    RAW_CLIPS_AUDIOS_DIR,
    RAW_CLIPS_VIDEOS_DIR,
    RAW_CSV_DIR,
    SPLITS,
)


def list_video_ids(api: HfApi, split: str) -> list[str]:
    prefix = f"clips_videos/{split}/"
    files = api.list_repo_files(HF_DATASET_REPO_ID, repo_type=HF_REPO_TYPE)
    return sorted(Path(f).stem for f in files if f.startswith(prefix) and f.endswith(".zip"))


def download_and_extract(repo_path: str, extract_dir: Path, tmp_dir: Path) -> None:
    local_path = hf_hub_download(
        repo_id=HF_DATASET_REPO_ID, repo_type=HF_REPO_TYPE, filename=repo_path, local_dir=tmp_dir
    )
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_path) as z:
        z.extractall(extract_dir)
    Path(local_path).unlink()


def download_csv(repo_path: str, tmp_dir: Path) -> pd.DataFrame:
    local_path = hf_hub_download(
        repo_id=HF_DATASET_REPO_ID, repo_type=HF_REPO_TYPE, filename=repo_path, local_dir=tmp_dir
    )
    df = pd.read_csv(local_path, header=None, names=CSV_COLUMNS)
    Path(local_path).unlink()
    return df


def process_split(
    api: HfApi, split: str, tmp_dir: Path, limit: int | None, skip_existing: bool
) -> None:
    video_ids = list_video_ids(api, split)
    if limit is not None:
        video_ids = video_ids[:limit]

    video_split_dir = RAW_CLIPS_VIDEOS_DIR / split
    audio_split_dir = RAW_CLIPS_AUDIOS_DIR / split
    df_list = []
    failed = []

    for video_id in tqdm(video_ids, desc=f"[{split}]"):
        try:
            already_extracted = (
                skip_existing
                and (video_split_dir / video_id).is_dir()
                and any((video_split_dir / video_id).iterdir())
                and (audio_split_dir / video_id).is_dir()
                and any((audio_split_dir / video_id).iterdir())
            )
            if not already_extracted:
                download_and_extract(f"clips_videos/{split}/{video_id}.zip", video_split_dir, tmp_dir)
                download_and_extract(f"clips_audios/{split}/{video_id}.zip", audio_split_dir, tmp_dir)
            df_list.append(download_csv(f"csv/{split}/{video_id}.csv", tmp_dir))
        except Exception as e:
            print(f"  [WARN] {split}/{video_id} failed: {e}")
            failed.append(video_id)

    if df_list:
        merged = pd.concat(df_list, ignore_index=True)
        RAW_CSV_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RAW_CSV_DIR / f"{split}_orig.csv"
        merged.to_csv(out_path, index=False)
        print(
            f"[{split}] wrote {out_path} ({len(merged)} rows, "
            f"{len(video_ids) - len(failed)}/{len(video_ids)} videos)"
        )

    if failed:
        print(f"[{split}] failed videos: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UniTalk-ASD into data/raw/")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument(
        "--limit", type=int, default=None, help="only process the first N videos per split (smoke test)"
    )
    parser.add_argument(
        "--no-skip-existing", action="store_true", help="re-download even if already extracted"
    )
    args = parser.parse_args()

    api = HfApi()
    start = time.time()
    with tempfile.TemporaryDirectory(prefix="unitalk_dl_") as tmp:
        tmp_dir = Path(tmp)
        for split in args.splits:
            process_split(api, split, tmp_dir, args.limit, skip_existing=not args.no_skip_existing)
    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
