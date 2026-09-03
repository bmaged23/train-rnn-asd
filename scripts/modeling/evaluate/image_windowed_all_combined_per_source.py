"""Evaluates a scripts/modeling/train/image_windowed_all_combined.py checkpoint
on the held-out "test" split, broken down PER SOURCE DATASET — same shape as
scripts/modeling/evaluate/windowed_all_combined_per_source.py's landmark-pipeline
report (full binary-classification metric suite per source, plus "combined"
for every window pooled together), just for the raw-pixel CNN+LSTM instead.

Simpler than the landmark version's per-source script: that one rebuilds a
separate person_id -> source_label map via a second, metadata-only
LandmarkSequenceDataset pass per source (written before ImageWindowedDataset's
predecessor exposed per-window source_label directly). ImageWindowedDataset
already tracks each window's source_label natively and
image_dataset.collate_images returns it straight in the batch — so this
script buckets by `batch["source_label"]` directly, no extra CSV reload
needed.

Writes to its own config.IMAGE_ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR folder —
reset (deleted and recreated) at the start of every run:
    evaluation_image_windowed_all_combined_per_source/{source}_metrics.json  (one per source, plus "combined")
    evaluation_image_windowed_all_combined_per_source/eval.log

Does NOT rebuild config.IMAGE_RESIZED_CACHE_DIR itself (unlike
scripts/modeling/train/image_windowed_all_combined.py, which always rebuilds
it fresh) — this script expects that training run's cache to still be on
disk. Run training first, or scripts/features/build_image_resized_cache.py
standalone, if evaluating without a preceding training run in the same
session; ImageWindowedDataset raises a clear error if the cache is missing.

Usage:
    python scripts/modeling/evaluate/image_windowed_all_combined_per_source.py [--checkpoint best|last]
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
# TF32 (added 2026-08-25, explicit user request — same real, essentially
# free speed lever as the training script; see that script's matching
# comment). Still a deterministic algorithm, doesn't conflict with
# use_deterministic_algorithms above.
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from config import (
    CHECKPOINTS_DIR,
    EVAL_LOG_FILENAME,
    IMAGE_ALL_WINDOWED_BEST_CHECKPOINT_FILENAME,
    IMAGE_ALL_WINDOWED_LAST_CHECKPOINT_FILENAME,
    IMAGE_ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR,
    IMAGE_EVAL_BATCH_SIZE,
    IMAGE_VAL_NUM_DATALOADER_WORKERS,
    TORCH_CPU_THREADS,
)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))
from dataset import ava_source, avspeech_source, columbia_source, speakingfaces_source, wasd_source
from image_dataset import get_image_windowed_dataloader
from image_model import ImageWindowedSpeakingDetectorRNN
from metrics import compute_metrics, format_report
from train_utils import reset_dir

if not torch.cuda.is_available():
    torch.set_num_threads(TORCH_CPU_THREADS)  # see config.py — avoids CPU thread-contention on this 64-core box

logger = logging.getLogger("evaluate_image_windowed_all_combined_per_source")

SPLIT = "test"
_CHECKPOINT_CHOICES = {
    "best": IMAGE_ALL_WINDOWED_BEST_CHECKPOINT_FILENAME,
    "last": IMAGE_ALL_WINDOWED_LAST_CHECKPOINT_FILENAME,
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
def collect_predictions_by_source(
    model: ImageWindowedSpeakingDetectorRNN, loader, device: torch.device
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Runs `model` over every batch, bucketing (y_true, y_prob) by
    batch["source_label"] (native per-window field, see module docstring) —
    returns {source_label: (y_true, y_pred, y_prob)}, plus "combined" for
    every window pooled together.

    Mixed precision (added 2026-08-25, explicit user request — same
    bfloat16 autocast as the training script; see that script's run_epoch()
    for the full rationale). Applied here purely for speed — inference
    correctness is unaffected either way, and .float() before .cpu().numpy()
    (already present below) already handles converting whatever dtype
    logits/probs end up in back to a numpy-compatible one.
    """
    model.eval()
    labels_by_source: dict[str, list] = {}
    probs_by_source: dict[str, list] = {}

    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        lengths = batch["lengths"]
        labels = batch["labels"].to(device, non_blocking=True)
        sources = batch["source_label"]

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(features, lengths)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        labels_np = labels.cpu().numpy()

        for source, label, prob in zip(sources, labels_np, probs):
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
    # Every run starts from a clean slate — deletes and recreates this
    # script's own evaluation folder (see src/train_utils.py).
    reset_dir(IMAGE_ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR)
    setup_logging(IMAGE_ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR / EVAL_LOG_FILENAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    checkpoint_path = CHECKPOINTS_DIR / _CHECKPOINT_CHOICES[checkpoint]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found — run scripts/modeling/train/image_windowed_all_combined.py first"
        )
    logger.info(f"checkpoint: {checkpoint_path} (split={SPLIT})")

    model = ImageWindowedSpeakingDetectorRNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # torch.compile AFTER load_state_dict, not before (added 2026-08-25,
    # explicit user request — same real speedup as the training script; see
    # that script's matching comment for the full rationale/benchmark).
    # torch.compile prefixes wrapped parameters' state_dict keys with
    # "_orig_mod." — compiling first would make load_state_dict's plain
    # checkpoint keys fail to match.
    model.backbone = torch.compile(model.backbone)
    model.adaface_backbone = torch.compile(model.adaface_backbone)

    loader = get_image_windowed_dataloader(
        SPLIT,
        batch_size=IMAGE_EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=IMAGE_VAL_NUM_DATALOADER_WORKERS,  # test split is similarly small to val, see that config comment
        extra_sources=[fn(SPLIT) for fn in _EXTRA_SOURCES],
    )
    by_source = collect_predictions_by_source(model, loader, device)

    for source in _SOURCE_ORDER:
        if source not in by_source:
            continue
        y_true, y_pred, y_prob = by_source[source]
        metrics = compute_metrics(y_true, y_pred, y_prob)
        logger.info(f"\n{'=' * 60}\n{source}\n{'=' * 60}\n" + format_report(metrics))

        metrics_path = IMAGE_ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR / f"{source}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({k: v for k, v in metrics.items() if not k.startswith("_")}, f, indent=2)
    logger.info(f"wrote per-source metrics to {IMAGE_ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="best", choices=sorted(_CHECKPOINT_CHOICES))
    args = parser.parse_args()
    main(checkpoint=args.checkpoint)
