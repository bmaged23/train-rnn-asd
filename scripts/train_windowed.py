"""Trains WindowedSpeakingDetectorRNN (src/windowed_model.py) — the windowed
many-to-one counterpart to scripts/train.py's per-frame many-to-many model.
Structurally mirrors scripts/train.py (device selection, pos_weight, early
stopping, checkpointing, progress charts) but simpler: windows are fixed-length
(config.WINDOW_SIZE) with a single label each, so there's no padding/masking
to account for — plain BCEWithLogitsLoss over the whole batch.

pos_weight is computed once at startup from the "train" split's window label
distribution (config.WINDOW_SIZE-sized, majority-voted labels — see
src/windowed_dataset.py) and applied only to the training pass, same
rationale as scripts/train.py.

Right after training, src/metrics.py charts train-vs-val progress (reusing
the exact same plot_loss_curve/plot_accuracy_curve/build_training_report
functions scripts/train.py uses — the CSV schema is identical) to logs/,
with a "windowed_" prefix so nothing collides with the per-frame run's files.

This script deliberately does NOT touch the held-out "test" split — that's
scripts/evaluate_windowed.py's job, run separately, writing to its own
evaluation_windowed/ folder (kept apart from evaluate.py's evaluation/,
since a windowed "sample" means a window, not a detected frame).

Checkpoints -> checkpoints/windowed_{best,last}_model.pt (gitignored).
Per-epoch metrics -> logs/windowed_train_metrics.csv (gitignored).
Human-readable run log -> logs/windowed_train.log (gitignored).

Usage:
    python scripts/train_windowed.py
"""
from __future__ import annotations

import csv
import logging
import os
import sys
import time
from pathlib import Path

# Must be set before CUDA is initialized (first torch.cuda.* call below) —
# required for torch.use_deterministic_algorithms(True) to cover CUDA >=10.2
# matmul/cuBLAS ops, otherwise those ops raise at runtime instead of just
# running non-deterministically. See main()'s determinism block.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    BALANCED_BATCHES,
    CHECKPOINTS_DIR,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    LOGS_DIR,
    NUM_EPOCHS,
    TORCH_CPU_THREADS,
    USE_MAR,
    USE_MOUTH_LANDMARKS_ONLY,
    WINDOW_SIZE,
    WINDOWED_ACCURACY_CURVE_FILENAME,
    WINDOWED_BEST_CHECKPOINT_FILENAME,
    WINDOWED_LAST_CHECKPOINT_FILENAME,
    WINDOWED_LOSS_CURVE_FILENAME,
    WINDOWED_TRAIN_LOG_FILENAME,
    WINDOWED_TRAIN_METRICS_FILENAME,
    WINDOWED_TRAINING_REPORT_FILENAME,
    WINDOWED_SEED,
)

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from dataset import NUM_INPUT_FEATURES
from windowed_dataset import get_windowed_dataloader
from windowed_model import WindowedSpeakingDetectorRNN
from metrics import build_training_report, plot_accuracy_curve, plot_loss_curve
from train_utils import reset_dir

if not torch.cuda.is_available():
    torch.set_num_threads(TORCH_CPU_THREADS)  # see config.py — avoids CPU thread-contention on this 64-core box

logger = logging.getLogger("train_windowed")


def compute_pos_weight(labels: np.ndarray) -> float:
    """num_negative / num_positive over a flat array of window labels (0/1) —
    same rationale as scripts/train.py's per-frame compute_pos_weight, just
    computed over windows instead of frames.
    """
    num_pos = (labels == 1).sum()
    num_neg = (labels == 0).sum()
    return float(num_neg / num_pos)


@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (logits > 0).long()
    return (preds == labels).float().mean().item()


def run_epoch(
    model: WindowedSpeakingDetectorRNN,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
    pos_weight: torch.Tensor | None = None,
) -> tuple[float, float]:
    """One pass over `loader`. Trains (updates model weights) iff optimizer
    is given; otherwise runs no-grad for eval (val). `pos_weight` is only
    applied when training — val loss stays unweighted so it reflects real
    generalization on the natural class distribution.
    """
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    for batch in tqdm(loader, desc=desc, leave=False):
        features = batch["features"].to(device)
        lengths = batch["lengths"]
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(features, lengths)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels.float(), pos_weight=pos_weight if train_mode else None
            )

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(logits, labels)
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
    # Fixed seed so different config values (e.g. weight_decay sweeps) are
    # actually comparable — without this, weight init/batch order/dropout
    # draws differ run to run and confound any config-effect comparison.
    torch.manual_seed(WINDOWED_SEED)
    np.random.seed(WINDOWED_SEED)

    # A fixed seed alone doesn't guarantee bit-identical runs: cuDNN's LSTM
    # backward pass accumulates gradients via atomicAdd on GPU, whose
    # summation order (and therefore result, in finite precision) isn't
    # reproducible run to run — confirmed 2026-07-17 when a same-seed,
    # same-config retrain (2s windows, no padding) landed at F1=0.6735
    # instead of the original 0.6803. These three calls force deterministic
    # (slower) kernel variants everywhere, closing that gap for future runs.
    # use_deterministic_algorithms(True) raises at runtime if any op in the
    # pipeline lacks a deterministic implementation — that's the point of
    # today's smoke test, to confirm this pipeline has none.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Every run starts from a clean slate — deletes and recreates both
    # folders, clearing any prior run's checkpoints/logs (see
    # src/train_utils.py; CHECKPOINTS_DIR/LOGS_DIR are shared across every
    # training script, not just this one).
    reset_dir(CHECKPOINTS_DIR)
    reset_dir(LOGS_DIR)
    setup_logging(LOGS_DIR / WINDOWED_TRAIN_LOG_FILENAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    logger.info(f"seed: {WINDOWED_SEED}")
    logger.info(
        f"features: mouth_landmarks_only={USE_MOUTH_LANDMARKS_ONLY} use_mar={USE_MAR} "
        f"(NUM_INPUT_FEATURES={NUM_INPUT_FEATURES})  window_size={WINDOW_SIZE}"
    )
    logger.info(f"balanced_batches: {BALANCED_BATCHES}")

    train_loader = get_windowed_dataloader("train")
    val_loader = get_windowed_dataloader("val")

    train_labels = np.array(train_loader.dataset.labels)
    pos_weight_value = compute_pos_weight(train_labels)
    pos_weight = torch.tensor(pos_weight_value, device=device)
    logger.info(
        f"pos_weight (train split NOT_SPEAKING:SPEAKING_AUDIBLE ratio, window labels): "
        f"{pos_weight_value:.4f}"
    )

    model = WindowedSpeakingDetectorRNN().to(device)
    # Weight decay (AdamW) was tried and swept across 4 values — none beat
    # plain Adam with no decay at all, so dropped entirely. Regularization
    # here now comes from reduced model capacity instead — see
    # WindowedSpeakingDetectorRNN's docstring / config.py's
    # WINDOWED_HIDDEN_SIZE/WINDOWED_NUM_RNN_LAYERS comment.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_checkpoint_path = CHECKPOINTS_DIR / WINDOWED_BEST_CHECKPOINT_FILENAME
    last_checkpoint_path = CHECKPOINTS_DIR / WINDOWED_LAST_CHECKPOINT_FILENAME
    metrics_path = LOGS_DIR / WINDOWED_TRAIN_METRICS_FILENAME

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
    # scripts/evaluate_windowed.py separately for the held-out classification-metrics check.
    metrics_df = pd.read_csv(metrics_path)
    plot_loss_curve(metrics_df, best_epoch=best_epoch, save_path=LOGS_DIR / WINDOWED_LOSS_CURVE_FILENAME)
    plot_accuracy_curve(metrics_df, best_epoch=best_epoch, save_path=LOGS_DIR / WINDOWED_ACCURACY_CURVE_FILENAME)
    logger.info(f"wrote {LOGS_DIR / WINDOWED_LOSS_CURVE_FILENAME} and {LOGS_DIR / WINDOWED_ACCURACY_CURVE_FILENAME}")

    report = build_training_report(
        metrics_df, best_epoch, EARLY_STOPPING_PATIENCE,
        loss_curve_filename=WINDOWED_LOSS_CURVE_FILENAME,
        accuracy_curve_filename=WINDOWED_ACCURACY_CURVE_FILENAME,
    )
    report_path = LOGS_DIR / WINDOWED_TRAINING_REPORT_FILENAME
    report_path.write_text(report)
    logger.info(f"wrote {report_path}")


if __name__ == "__main__":
    main()
