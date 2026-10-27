"""Breaks scripts/modeling/evaluate/windowed_all_combined.py's combined test-split
metrics down PER SOURCE DATASET — same checkpoint, same combined test loader,
single forward pass, using each batch's native source_label metadata. This
avoids a redundant full landmark-data load and does not assume track IDs are
globally unique across independent datasets.

Writes to its own config.ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR folder —
reset (deleted and recreated) at the start of every run, same convention as
scripts/modeling/evaluate/windowed_all_combined.py's own evaluation dir, so
stale results from a previous checkpoint never linger:
    evaluation_windowed_all_combined_per_source/{source}_metrics.json  (one per source, plus "combined")
    evaluation_windowed_all_combined_per_source/eval.log

Usage:
    python scripts/modeling/evaluate/windowed_all_combined_per_source.py [--checkpoint best|last]
"""
from __future__ import annotations

import argparse
import json
import logging
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
    ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR,
    CHECKPOINTS_DIR,
    EVAL_LOG_FILENAME,
)

from src.windowed_dataset import (
    ava_source,
    avspeech_source,
    columbia_source,
    get_windowed_dataloader,
    speakingfaces_source,
    wasd_source,
)
from src.windowed_model import WindowedSpeakingDetectorRNN
from src.metrics import compute_metrics, format_report
from src.train_utils import reset_dir

logger = logging.getLogger("evaluate_windowed_all_combined_per_source")

SPLIT = "test"
_CHECKPOINT_CHOICES = {
    "best": ALL_WINDOWED_BEST_CHECKPOINT_FILENAME,
    "last": ALL_WINDOWED_LAST_CHECKPOINT_FILENAME,
}
_EXTRA_SOURCES = [ava_source, wasd_source, columbia_source, avspeech_source, speakingfaces_source]
_SOURCE_ORDER = ["unitalk", "ava", "wasd", "columbia", "avspeech", "speakingfaces", "combined"]


def setup_logging(log_path: Path) -> None:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


@torch.no_grad()
def collect_predictions_with_source(
    model: WindowedSpeakingDetectorRNN, loader, device: torch.device
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
        sources = batch["source_label"]

        logits = model(features, lengths)
        probs = torch.sigmoid(logits).cpu().numpy()
        labels_np = labels.cpu().numpy()

        for source, label, prob in zip(sources, labels_np, probs, strict=True):
            if source not in _SOURCE_ORDER[:-1]:
                raise ValueError(f'Unrecognized source label: {source!r}')
            labels_by_source.setdefault(source, []).append(label)
            probs_by_source.setdefault(source, []).append(prob)
            labels_by_source.setdefault("combined", []).append(label)
            probs_by_source.setdefault("combined", []).append(prob)

    if not labels_by_source:
        raise ValueError('No windows were processed; cannot report an evaluation')
    result = {}
    for source in labels_by_source:
        y_true = np.array(labels_by_source[source], dtype=np.int64)
        y_prob = np.array(probs_by_source[source], dtype=np.float64)
        y_pred = (y_prob > 0.5).astype(np.int64)
        result[source] = (y_true, y_pred, y_prob)
    return result


def main(checkpoint: str = "best") -> None:
    # Every run starts from a clean slate — deletes and recreates this
    # script's own evaluation folder (see src/train_utils.py), same
    # convention as scripts/modeling/evaluate/windowed_all_combined.py.
    reset_dir(ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR)
    setup_logging(ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR / EVAL_LOG_FILENAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    checkpoint_path = CHECKPOINTS_DIR / _CHECKPOINT_CHOICES[checkpoint]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_path} not found — run scripts/modeling/train/windowed_all_combined.py first")
    logger.info(f"checkpoint: {checkpoint_path} (split={SPLIT})")

    model = WindowedSpeakingDetectorRNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    loader = get_windowed_dataloader(
        SPLIT,
        shuffle=False,
        extra_sources=[fn(SPLIT) for fn in _EXTRA_SOURCES],
    )
    by_source = collect_predictions_with_source(model, loader, device)

    for source in _SOURCE_ORDER:
        if source not in by_source:
            continue
        y_true, y_pred, y_prob = by_source[source]
        metrics = compute_metrics(y_true, y_pred, y_prob)
        logger.info(f"\n{'=' * 60}\n{source}\n{'=' * 60}\n" + format_report(metrics))

        metrics_path = ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR / f"{source}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({k: v for k, v in metrics.items() if not k.startswith("_")}, f, indent=2)
    logger.info(f"wrote per-source metrics to {ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="best", choices=sorted(_CHECKPOINT_CHOICES))
    args = parser.parse_args()
    main(checkpoint=args.checkpoint)
