"""Breaks scripts/modeling/evaluate/windowed_all_combined.py's combined test-split
metrics down PER SOURCE DATASET — same checkpoint, same combined test loader,
single forward pass, but each window's person_id is looked up against every
source's own person_id set (cheap metadata-only LandmarkSequenceDataset
loads, no windowing) to bucket predictions by source before computing
per-source confusion matrices — rather than the combined-only report
scripts/modeling/evaluate/windowed_all_combined.py writes.

This works because every source's person_id is globally unique (by
construction — see [[project-asd-avspeech-integration]]'s person_id/entity_id
collision bug and fix for why that invariant matters), so a person_id ->
source_label mapping built once up front is unambiguous.

Prints one compute_metrics()/format_report() block per source, plus the
combined block (same numbers windowed_all_combined.py itself reports, as a
cross-check). Writes nothing to disk — purely a diagnostic, not a tracked
evaluation/ output.

Usage:
    python scripts/modeling/evaluate/windowed_all_combined_per_source.py [--checkpoint best|last]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from config import (
    ALL_WINDOWED_BEST_CHECKPOINT_FILENAME,
    ALL_WINDOWED_LAST_CHECKPOINT_FILENAME,
    CHECKPOINTS_DIR,
)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))
from dataset import LandmarkSequenceDataset
from windowed_dataset import (
    ava_source,
    avspeech_source,
    columbia_source,
    get_windowed_dataloader,
    speakingfaces_source,
    wasd_source,
)
from windowed_model import WindowedSpeakingDetectorRNN
from metrics import compute_metrics, format_report

SPLIT = "test"
_CHECKPOINT_CHOICES = {
    "best": ALL_WINDOWED_BEST_CHECKPOINT_FILENAME,
    "last": ALL_WINDOWED_LAST_CHECKPOINT_FILENAME,
}
_EXTRA_SOURCES = [ava_source, wasd_source, columbia_source, avspeech_source, speakingfaces_source]


def build_person_id_to_source_map() -> dict[str, str]:
    """Metadata-only pass (no windowing) per source, just to know which
    person_ids belong to which dataset. "unitalk" is the primary source
    (csv_path=None -> config.DATA_PROCESSED_DIR's own test_split.csv).
    """
    mapping: dict[str, str] = {}
    primary = LandmarkSequenceDataset(SPLIT, source_label="unitalk")
    for pid in primary.person_ids:
        mapping[pid] = "unitalk"
    for source_fn in _EXTRA_SOURCES:
        csv_path, label = source_fn(SPLIT)
        ds = LandmarkSequenceDataset(SPLIT, csv_path=csv_path, source_label=label)
        for pid in ds.person_ids:
            mapping[pid] = label
    return mapping


@torch.no_grad()
def collect_predictions_with_source(
    model: WindowedSpeakingDetectorRNN, loader, device: torch.device, person_id_to_source: dict[str, str]
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Runs `model` over every batch, bucketing (y_true, y_prob) by source —
    returns {source_label: (y_true, y_pred, y_prob)}, plus "combined" for
    every window pooled together.
    """
    model.eval()
    labels_by_source: dict[str, list] = {}
    probs_by_source: dict[str, list] = {}

    for batch in loader:
        features = batch["features"].to(device)
        lengths = batch["lengths"]
        labels = batch["labels"].to(device)
        person_ids = batch["person_id"]

        logits = model(features, lengths)
        probs = torch.sigmoid(logits).cpu().numpy()
        labels_np = labels.cpu().numpy()

        for pid, label, prob in zip(person_ids, labels_np, probs):
            source = person_id_to_source.get(pid, "UNKNOWN")
            labels_by_source.setdefault(source, []).append(label)
            probs_by_source.setdefault(source, []).append(prob)
            labels_by_source.setdefault("combined", []).append(label)
            probs_by_source.setdefault("combined", []).append(prob)

    result = {}
    for source in labels_by_source:
        y_true = np.array(labels_by_source[source], dtype=np.int64)
        y_prob = np.array(probs_by_source[source], dtype=np.float64)
        y_pred = (y_prob > 0.5).astype(np.int64)
        result[source] = (y_true, y_pred, y_prob)
    return result


def main(checkpoint: str = "best") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print("building person_id -> source map (metadata-only, no windowing)...")
    person_id_to_source = build_person_id_to_source_map()

    checkpoint_path = CHECKPOINTS_DIR / _CHECKPOINT_CHOICES[checkpoint]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_path} not found — run scripts/modeling/train/windowed_all_combined.py first")
    print(f"checkpoint: {checkpoint_path} (split={SPLIT})")

    model = WindowedSpeakingDetectorRNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    loader = get_windowed_dataloader(
        SPLIT,
        shuffle=False,
        extra_sources=[fn(SPLIT) for fn in _EXTRA_SOURCES],
    )
    by_source = collect_predictions_with_source(model, loader, device, person_id_to_source)

    source_order = ["unitalk", "ava", "wasd", "columbia", "avspeech", "speakingfaces", "combined"]
    for source in source_order:
        if source not in by_source:
            continue
        y_true, y_pred, y_prob = by_source[source]
        metrics = compute_metrics(y_true, y_pred, y_prob)
        print(f"\n{'=' * 60}\n{source}\n{'=' * 60}")
        print(format_report(metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="best", choices=sorted(_CHECKPOINT_CHOICES))
    args = parser.parse_args()
    main(checkpoint=args.checkpoint)
