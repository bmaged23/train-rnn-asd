"""Trains SpeakingDetectorRNN (src/model.py) on the processed landmark dataset.

Per epoch: one masked-BCE training pass over "train" (gradient updates), then
a no-grad pass over "val" (early stopping / model selection —
config.EARLY_STOPPING_PATIENCE). Training stops when either NUM_EPOCHS is
reached or val loss hasn't improved for EARLY_STOPPING_PATIENCE epochs,
whichever comes first.

The training-loss `pos_weight` is computed once at startup from the "train"
split's actual class ratio (masked, real-detected frames only) and applied
only to the training pass — NOT_SPEAKING outnumbers SPEAKING_AUDIBLE roughly
3:1 in this dataset (see notebooks/03_explore_processed_dataset.ipynb), which
otherwise biases the model toward under-predicting SPEAKING_AUDIBLE (see
evaluation/ from a run of scripts/modeling/evaluate/frames.py: recall 0.53 on
SPEAKING_AUDIBLE vs. 0.94 on NOT_SPEAKING despite ROC-AUC 0.88 — the model
ranks the classes well, the unweighted loss just doesn't push hard enough on
the minority class). Val/test loss stay unweighted so they reflect the
model's actual, uncorrected generalization on the natural class distribution.

Right after that, src/metrics.py charts train-vs-val progress across the
whole run (loss + accuracy, both from logs/train_metrics.csv) and writes a
short report, all to logs/ (gitignored) next to train_metrics.csv/train.log:
    logs/loss_curve.png, logs/accuracy_curve.png  — train vs val, all epochs
    logs/README.md                                — write-up of the above

This script deliberately does NOT touch the held-out "test" split — that
classification-metrics check (accuracy/precision/recall/F1/confusion matrix/
ROC-AUC/PR-AUC) is scripts/modeling/evaluate/frames.py's job, run separately/manually,
writing to its own evaluation/ folder. Keeping the two apart means "test"
stays untouched by anything that runs automatically during/after training.

Checkpoints -> checkpoints/{best,last}_model.pt (gitignored).
Per-epoch metrics -> logs/train_metrics.csv (gitignored).
Human-readable run log (same lines as printed to the console) -> logs/train.log
(gitignored) — so progress/results survive a backgrounded/disconnected run.

Usage:
    python scripts/modeling/train/frames.py
"""
from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from config import (
    ACCURACY_CURVE_FILENAME,
    BALANCED_BATCHES,
    BEST_CHECKPOINT_FILENAME,
    CHECKPOINTS_DIR,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP_NORM,
    LAST_CHECKPOINT_FILENAME,
    LEARNING_RATE,
    LOGS_DIR,
    LOSS_CURVE_FILENAME,
    NUM_EPOCHS,
    TORCH_CPU_THREADS,
    TRAIN_LOG_FILENAME,
    TRAIN_METRICS_FILENAME,
    TRAINING_REPORT_FILENAME,
    USE_MAR,
    USE_MOUTH_LANDMARKS_ONLY,
)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))
from dataset import NUM_INPUT_FEATURES, get_dataloader
from model import SpeakingDetectorRNN
from metrics import build_training_report, plot_accuracy_curve, plot_loss_curve
from train_utils import reset_dir

if not torch.cuda.is_available():
    torch.set_num_threads(TORCH_CPU_THREADS)  # see config.py — avoids CPU thread-contention on this 64-core box

logger = logging.getLogger("train")


def masked_bce_loss(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor | None = None
) -> torch.Tensor:
    per_frame = nn.functional.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction="none", pos_weight=pos_weight
    )
    return (per_frame * mask).sum() / mask.sum().clamp(min=1)


def compute_pos_weight(dataset) -> float:
    """num_negative / num_positive (target_index: 0=NOT_SPEAKING, 1=SPEAKING_AUDIBLE)
    over `dataset`'s masked (real, detected) frames — the value BCEWithLogitsLoss's
    pos_weight needs to make the positive (SPEAKING_AUDIBLE) class's loss
    contribution match the negative class's, offsetting its ~3:1 underrepresentation.
    """
    labels = np.concatenate(dataset.labels)
    mask = np.concatenate(dataset.masks).astype(bool)
    masked_labels = labels[mask]
    num_pos = (masked_labels == 1).sum()
    num_neg = (masked_labels == 0).sum()
    return float(num_neg / num_pos)


@torch.no_grad()
def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    preds = (logits > 0).long()
    correct = ((preds == labels).float() * mask).sum()
    return (correct / mask.sum().clamp(min=1)).item()


def run_epoch(
    model: SpeakingDetectorRNN,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
    pos_weight: torch.Tensor | None = None,
) -> tuple[float, float]:
    """One pass over `loader`. Trains (updates model weights) iff optimizer
    is given; otherwise runs no-grad for eval (val/test). `pos_weight` is
    only applied when training — val/test loss stays unweighted so it
    reflects real generalization on the natural class distribution.
    """
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    for batch in tqdm(loader, desc=desc, leave=False):
        features = batch["features"].to(device)
        lengths = batch["lengths"]
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(features, lengths)
            loss = masked_bce_loss(logits, labels, mask, pos_weight=pos_weight if train_mode else None)

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()

        total_loss += loss.item()
        total_acc += masked_accuracy(logits, labels, mask)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def setup_logging(log_path: Path) -> None:
    """Mirrors every logger.info() line to both the console and log_path, so
    progress/results are still readable after a backgrounded/disconnected run.
    """
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def main() -> None:
    # Every run starts from a clean slate — deletes and recreates both
    # folders, clearing any prior run's checkpoints/logs (see
    # src/train_utils.py; CHECKPOINTS_DIR/LOGS_DIR are shared across every
    # training script, not just this one).
    reset_dir(CHECKPOINTS_DIR)
    reset_dir(LOGS_DIR)
    setup_logging(LOGS_DIR / TRAIN_LOG_FILENAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    logger.info(
        f"features: mouth_landmarks_only={USE_MOUTH_LANDMARKS_ONLY} use_mar={USE_MAR} "
        f"(NUM_INPUT_FEATURES={NUM_INPUT_FEATURES})"
    )
    logger.info(f"balanced_batches: {BALANCED_BATCHES}")

    train_loader = get_dataloader("train")
    val_loader = get_dataloader("val")

    pos_weight_value = compute_pos_weight(train_loader.dataset)
    pos_weight = torch.tensor(pos_weight_value, device=device)
    logger.info(
        f"pos_weight (train split NOT_SPEAKING:SPEAKING_AUDIBLE ratio, masked frames): "
        f"{pos_weight_value:.4f}"
    )

    model = SpeakingDetectorRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_checkpoint_path = CHECKPOINTS_DIR / BEST_CHECKPOINT_FILENAME
    last_checkpoint_path = CHECKPOINTS_DIR / LAST_CHECKPOINT_FILENAME
    metrics_path = LOGS_DIR / TRAIN_METRICS_FILENAME

    best_val_loss = float("inf")
    best_epoch = 1
    epochs_without_improvement = 0

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])

        for epoch in range(1, NUM_EPOCHS + 1):
            t0 = time.time()
            train_loss, train_acc = run_epoch(
                model, train_loader, device, optimizer, desc=f"epoch {epoch} train", pos_weight=pos_weight
            )
            val_loss, val_acc = run_epoch(model, val_loader, device, optimizer=None, desc=f"epoch {epoch} val")
            elapsed = time.time() - t0

            logger.info(
                f"epoch {epoch}/{NUM_EPOCHS}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({elapsed:.1f}s)"
            )
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, round(elapsed, 1)])
            f.flush()

            torch.save(model.state_dict(), last_checkpoint_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), best_checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    logger.info(
                        f"early stopping: val_loss hasn't improved for "
                        f"{EARLY_STOPPING_PATIENCE} epochs (best={best_val_loss:.4f})"
                    )
                    break

    # --- Combined train/val progress charts, across every epoch of this run ---
    # Deliberately no "test" split here — see module docstring. Run
    # scripts/modeling/evaluate/frames.py separately for the held-out classification-metrics check.
    metrics_df = pd.read_csv(metrics_path)
    plot_loss_curve(metrics_df, best_epoch=best_epoch, save_path=LOGS_DIR / LOSS_CURVE_FILENAME)
    plot_accuracy_curve(metrics_df, best_epoch=best_epoch, save_path=LOGS_DIR / ACCURACY_CURVE_FILENAME)
    logger.info(f"wrote {LOGS_DIR / LOSS_CURVE_FILENAME} and {LOGS_DIR / ACCURACY_CURVE_FILENAME}")

    report = build_training_report(
        metrics_df, best_epoch, EARLY_STOPPING_PATIENCE,
        loss_curve_filename=LOSS_CURVE_FILENAME,
        accuracy_curve_filename=ACCURACY_CURVE_FILENAME,
    )
    report_path = LOGS_DIR / TRAINING_REPORT_FILENAME
    report_path.write_text(report)
    logger.info(f"wrote {report_path}")


if __name__ == "__main__":
    main()
