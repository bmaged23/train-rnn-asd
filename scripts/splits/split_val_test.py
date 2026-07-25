"""Splits data/processed/val/combined_landmarks.csv into val_split.csv and
test_split.csv (same folder), by whole person_id track — no track's frames
ever end up split across both files.

"val" is used every epoch for early stopping / model selection during
training (src/dataset.py's "val" split); "test" is held out and only touched
for final evaluation — the actual check for overfitting past what early
stopping on "val" alone would catch.

A plain random 70/30 shuffle could easily land more of the already-scarce
qualifying (>= MIN_DETECTED_FRAMES_RATIO detected) tracks in one file than
the other, or skew the SPEAKING_AUDIBLE/NOT_SPEAKING balance between them —
either would make "val" and "test" measure different things. So tracks are
first bucketed into 4 strata (qualifies x speaking-majority) and each
stratum is independently shuffled + split at VAL_SPLIT_RATIO, keeping both
output files close to equal on both properties by construction.

Row data itself is copied through as raw text (no float reparsing — the
source file has no quoted/comma-containing fields, confirmed by grep), so
this is a fast single streaming pass over the 8GB source file after a first
lightweight metadata-only pass decides the per-track assignment.

Usage:
    python scripts/splits/split_val_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import (
    COMBINED_LANDMARKS_FILENAME,
    DATA_PROCESSED_DIR,
    MIN_DETECTED_FRAMES_RATIO,
    TEST_SPLIT_FILENAME,
    VAL_SPLIT_FILENAME,
    VAL_SPLIT_RATIO,
    VAL_TEST_SPLIT_SEED,
)

# Defaults are UniTalk-ASD's own paths; scripts/splits/split_ava_val_test.py calls
# main() with AVA_PROCESSED_DATASET_DIR's equivalents instead, reusing every
# function below unmodified — see that script.
SOURCE_CSV = DATA_PROCESSED_DIR / "val" / COMBINED_LANDMARKS_FILENAME
VAL_OUT = DATA_PROCESSED_DIR / "val" / VAL_SPLIT_FILENAME
TEST_OUT = DATA_PROCESSED_DIR / "val" / TEST_SPLIT_FILENAME


def _compute_track_stats(source_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(source_csv, usecols=["person_id", "detected", "target_index"], engine="pyarrow")
    stats = df.groupby("person_id").agg(
        detected_ratio=("detected", "mean"),
        speaking_frac=("target_index", "mean"),
    )
    stats["qualifies"] = stats["detected_ratio"] >= MIN_DETECTED_FRAMES_RATIO
    stats["speaking_majority"] = stats["speaking_frac"] >= 0.5
    return stats


def _assign_splits(stats: pd.DataFrame) -> pd.Series:
    """Stratified shuffle-split by (qualifies, speaking_majority), fixed seed."""
    rng = np.random.default_rng(VAL_TEST_SPLIT_SEED)
    assignment = pd.Series(index=stats.index, dtype=object)

    for _, group in stats.groupby(["qualifies", "speaking_majority"]):
        ids = rng.permutation(group.index.to_numpy())
        n_val = round(len(ids) * VAL_SPLIT_RATIO)
        assignment.loc[ids[:n_val]] = "val"
        assignment.loc[ids[n_val:]] = "test"

    return assignment


def _write_split_csvs(assignment: pd.Series, source_csv: Path, val_out: Path, test_out: Path) -> tuple[int, int]:
    assignment_map = assignment.to_dict()
    n_val = n_test = 0
    with (
        open(source_csv) as src,
        open(val_out, "w") as val_f,
        open(test_out, "w") as test_f,
    ):
        header = next(src)
        val_f.write(header)
        test_f.write(header)
        for line in src:
            person_id = line.split(",", 1)[0]
            split = assignment_map.get(person_id)
            if split == "val":
                val_f.write(line)
                n_val += 1
            elif split == "test":
                test_f.write(line)
                n_test += 1
    return n_val, n_test


def _print_balance_report(stats: pd.DataFrame, assignment: pd.Series) -> None:
    stats = stats.copy()
    stats["split"] = assignment
    report = stats.groupby("split").agg(
        n_tracks=("detected_ratio", "size"),
        pct_qualifying=("qualifies", "mean"),
        mean_detected_ratio=("detected_ratio", "mean"),
        pct_speaking_majority=("speaking_majority", "mean"),
    )
    print(report.round(4))


def main(source_csv: Path = SOURCE_CSV, val_out: Path = VAL_OUT, test_out: Path = TEST_OUT) -> None:
    stats = _compute_track_stats(source_csv)
    assignment = _assign_splits(stats)
    _print_balance_report(stats, assignment)

    n_val_rows, n_test_rows = _write_split_csvs(assignment, source_csv, val_out, test_out)
    print(f"wrote {n_val_rows:,} rows ({(assignment == 'val').sum():,} tracks) -> {val_out}")
    print(f"wrote {n_test_rows:,} rows ({(assignment == 'test').sum():,} tracks) -> {test_out}")


if __name__ == "__main__":
    main()
