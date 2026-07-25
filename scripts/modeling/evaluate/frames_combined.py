"""Evaluates a scripts/modeling/train/frames_combined.py checkpoint on the held-out
"test" split — UniTalk-ASD + AVA-ActiveSpeaker combined (same extra_sources
the training script used), so the reported metrics reflect the same combined
distribution the model was trained/validated on. Otherwise identical to
scripts/modeling/evaluate/frames.py (full binary-classification metric suite: accuracy,
precision, recall, F1, confusion matrix, ROC-AUC, PR-AUC; only real,
detected frames count, batch["mask"] == 1).

Writes to its own evaluation_combined/ folder — kept separate from
evaluation/ (the UniTalk-only per-frame model's results):
    evaluation_combined/eval_metrics.json     (gitignored)
    evaluation_combined/confusion_matrix.png  (gitignored)
    evaluation_combined/roc_curve.png         (gitignored)
    evaluation_combined/pr_curve.png          (gitignored)
    evaluation_combined/eval.log              (gitignored)

Usage:
    python scripts/modeling/evaluate/frames_combined.py [--checkpoint best|last]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Must be set before CUDA is initialized — see scripts/modeling/train/windowed.py's
# matching comment.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from config import (
    CHECKPOINTS_DIR,
    COMBINED_BEST_CHECKPOINT_FILENAME,
    COMBINED_EVALUATION_DIR,
    COMBINED_LAST_CHECKPOINT_FILENAME,
    EVAL_CONFUSION_MATRIX_FILENAME,
    EVAL_LOG_FILENAME,
    EVAL_METRICS_FILENAME,
    EVAL_PR_CURVE_FILENAME,
    EVAL_ROC_CURVE_FILENAME,
    TORCH_CPU_THREADS,
    USE_MAR,
    USE_MOUTH_LANDMARKS_ONLY,
)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))
from dataset import NUM_INPUT_FEATURES, ava_source, get_dataloader
from metrics import compute_metrics, format_report, plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from model import SpeakingDetectorRNN
from train_utils import reset_dir

if not torch.cuda.is_available():
    torch.set_num_threads(TORCH_CPU_THREADS)  # see config.py — avoids CPU thread-contention on this 64-core box

logger = logging.getLogger("evaluate_frames_combined")

SPLIT = "test"  # hardcoded — same rationale as scripts/modeling/evaluate/frames.py
_CHECKPOINT_CHOICES = {"best": COMBINED_BEST_CHECKPOINT_FILENAME, "last": COMBINED_LAST_CHECKPOINT_FILENAME}


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
    model: SpeakingDetectorRNN, loader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Runs `model` over every batch in `loader` and returns flat
    (y_true, y_pred, y_prob) arrays over masked (real, detected) frames only.
    """
    model.eval()
    all_labels, all_probs = [], []
    for batch in loader:
        features = batch["features"].to(device)
        lengths = batch["lengths"]
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device).bool()

        logits = model(features, lengths)
        probs = torch.sigmoid(logits)

        all_labels.append(labels[mask].cpu().numpy())
        all_probs.append(probs[mask].cpu().numpy())

    y_true = np.concatenate(all_labels).astype(np.int64)
    y_prob = np.concatenate(all_probs).astype(np.float64)
    y_pred = (y_prob > 0.5).astype(np.int64)
    return y_true, y_pred, y_prob


def main(checkpoint: str = "best") -> dict:
    """Runs the full combined test-split evaluation and returns the metrics dict."""
    # Every run starts from a clean slate — deletes and recreates this
    # script's own evaluation folder (see src/train_utils.py).
    reset_dir(COMBINED_EVALUATION_DIR)
    setup_logging(COMBINED_EVALUATION_DIR / EVAL_LOG_FILENAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    logger.info(
        f"features: mouth_landmarks_only={USE_MOUTH_LANDMARKS_ONLY} use_mar={USE_MAR} "
        f"(NUM_INPUT_FEATURES={NUM_INPUT_FEATURES}) — must match the checkpoint's training config "
        f"or model.load_state_dict below will raise a shape-mismatch error"
    )
    logger.info("data sources: UniTalk-ASD + AVA-ActiveSpeaker (combined test split)")

    checkpoint_path = CHECKPOINTS_DIR / _CHECKPOINT_CHOICES[checkpoint]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_path} not found — run scripts/modeling/train/frames_combined.py first")
    logger.info(f"checkpoint: {checkpoint_path} (split={SPLIT})")

    model = SpeakingDetectorRNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    loader = get_dataloader(SPLIT, shuffle=False, extra_sources=[ava_source(SPLIT)])
    y_true, y_pred, y_prob = collect_predictions(model, loader, device)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    logger.info("\n" + format_report(metrics))

    metrics_path = COMBINED_EVALUATION_DIR / EVAL_METRICS_FILENAME
    with open(metrics_path, "w") as f:
        json.dump({k: v for k, v in metrics.items() if not k.startswith("_")}, f, indent=2)
    logger.info(f"wrote {metrics_path}")

    plot_confusion_matrix(metrics, split=SPLIT, save_path=COMBINED_EVALUATION_DIR / EVAL_CONFUSION_MATRIX_FILENAME)
    plot_roc_curve(metrics, split=SPLIT, save_path=COMBINED_EVALUATION_DIR / EVAL_ROC_CURVE_FILENAME)
    plot_pr_curve(metrics, split=SPLIT, save_path=COMBINED_EVALUATION_DIR / EVAL_PR_CURVE_FILENAME)
    logger.info(f"wrote plots to {COMBINED_EVALUATION_DIR}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="best", choices=sorted(_CHECKPOINT_CHOICES),
        help="which checkpoint to load (default: best, lowest val loss)",
    )
    args = parser.parse_args()
    main(checkpoint=args.checkpoint)  # split is hardcoded to "test" — see module docstring
