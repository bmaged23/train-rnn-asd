"""Evaluates a trained WindowedSpeakingDetectorRNN checkpoint on the held-out
"test" split with the full binary-classification metric suite: accuracy,
precision, recall, F1, confusion matrix, ROC-AUC, PR-AUC. Hardcoded to
"test", same rationale as scripts/evaluate.py.

Unlike scripts/evaluate.py, there's no frame-level masking here — each
window is already exactly one sample with one label (src/windowed_dataset.py
majority-votes the label at window-build time), so every window in the test
split is scored, no filtering needed at eval time.

This is a standalone, manually-run check — scripts/train_windowed.py does
not call it. Writes to its own evaluation_windowed/ folder (not
evaluation/, which holds the per-frame model's results — a "sample" means
something different here, a window rather than a detected frame, so mixing
the two folders would be confusing):
    evaluation_windowed/eval_metrics.json     — the metrics dict (gitignored)
    evaluation_windowed/confusion_matrix.png  — row-normalized confusion matrix (gitignored)
    evaluation_windowed/roc_curve.png         — ROC curve (gitignored)
    evaluation_windowed/pr_curve.png          — precision-recall curve (gitignored)
    evaluation_windowed/eval.log              — human-readable run log (gitignored)

Usage:
    python scripts/evaluate_windowed.py [--checkpoint best|last]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Must be set before CUDA is initialized — see scripts/train_windowed.py's
# matching comment. Not strictly required here (evaluation is a no-grad
# forward pass on fixed weights, so it lacks the backward-pass atomicAdd
# non-determinism that motivated this on the training side), but kept
# consistent with train_windowed.py for full rigor.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    CHECKPOINTS_DIR,
    EVAL_CONFUSION_MATRIX_FILENAME,
    EVAL_LOG_FILENAME,
    EVAL_METRICS_FILENAME,
    EVAL_PR_CURVE_FILENAME,
    EVAL_ROC_CURVE_FILENAME,
    TORCH_CPU_THREADS,
    USE_MAR,
    USE_MOUTH_LANDMARKS_ONLY,
    WINDOW_SIZE,
    WINDOWED_BEST_CHECKPOINT_FILENAME,
    WINDOWED_EVALUATION_DIR,
    WINDOWED_LAST_CHECKPOINT_FILENAME,
)

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from dataset import NUM_INPUT_FEATURES
from windowed_dataset import get_windowed_dataloader
from windowed_model import WindowedSpeakingDetectorRNN
from metrics import compute_metrics, format_report, plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from train_utils import reset_dir

if not torch.cuda.is_available():
    torch.set_num_threads(TORCH_CPU_THREADS)  # see config.py — avoids CPU thread-contention on this 64-core box

logger = logging.getLogger("evaluate_windowed")

SPLIT = "test"  # hardcoded — see module docstring
_CHECKPOINT_CHOICES = {"best": WINDOWED_BEST_CHECKPOINT_FILENAME, "last": WINDOWED_LAST_CHECKPOINT_FILENAME}


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
def collect_predictions(
    model: WindowedSpeakingDetectorRNN, loader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs `model` over every batch in `loader` and returns flat
    (y_true, y_pred, y_prob) arrays — one entry per window, all windows
    (no masking needed, unlike scripts/evaluate.py's per-frame version).
    """
    model.eval()
    all_labels, all_probs = [], []
    for batch in loader:
        features = batch["features"].to(device)
        lengths = batch["lengths"]
        labels = batch["labels"].to(device)

        logits = model(features, lengths)
        probs = torch.sigmoid(logits)

        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels).astype(np.int64)
    y_prob = np.concatenate(all_probs).astype(np.float64)
    y_pred = (y_prob > 0.5).astype(np.int64)
    return y_true, y_pred, y_prob


def main(checkpoint: str = "best") -> dict:
    """Runs the full test-split evaluation and returns the metrics dict."""
    # Every run starts from a clean slate — deletes and recreates this
    # script's own evaluation folder (see src/train_utils.py).
    reset_dir(WINDOWED_EVALUATION_DIR)
    setup_logging(WINDOWED_EVALUATION_DIR / EVAL_LOG_FILENAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    logger.info(
        f"features: mouth_landmarks_only={USE_MOUTH_LANDMARKS_ONLY} use_mar={USE_MAR} "
        f"(NUM_INPUT_FEATURES={NUM_INPUT_FEATURES})  window_size={WINDOW_SIZE} — must match the "
        f"checkpoint's training config or model.load_state_dict below will raise a shape-mismatch error"
    )

    checkpoint_path = CHECKPOINTS_DIR / _CHECKPOINT_CHOICES[checkpoint]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_path} not found — run scripts/train_windowed.py first")
    logger.info(f"checkpoint: {checkpoint_path} (split={SPLIT})")

    model = WindowedSpeakingDetectorRNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    loader = get_windowed_dataloader(SPLIT, shuffle=False)
    y_true, y_pred, y_prob = collect_predictions(model, loader, device)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    logger.info("\n" + format_report(metrics))

    metrics_path = WINDOWED_EVALUATION_DIR / EVAL_METRICS_FILENAME
    with open(metrics_path, "w") as f:
        json.dump({k: v for k, v in metrics.items() if not k.startswith("_")}, f, indent=2)
    logger.info(f"wrote {metrics_path}")

    plot_confusion_matrix(
        metrics, split=SPLIT, save_path=WINDOWED_EVALUATION_DIR / EVAL_CONFUSION_MATRIX_FILENAME
    )
    plot_roc_curve(metrics, split=SPLIT, save_path=WINDOWED_EVALUATION_DIR / EVAL_ROC_CURVE_FILENAME)
    plot_pr_curve(metrics, split=SPLIT, save_path=WINDOWED_EVALUATION_DIR / EVAL_PR_CURVE_FILENAME)
    logger.info(f"wrote plots to {WINDOWED_EVALUATION_DIR}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="best", choices=sorted(_CHECKPOINT_CHOICES),
        help="which checkpoint to load (default: best, lowest val loss)",
    )
    args = parser.parse_args()
    main(checkpoint=args.checkpoint)  # split is hardcoded to "test" — see module docstring
