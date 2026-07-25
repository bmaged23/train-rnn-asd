"""Trains SpeakingDetectorRNN (src/model.py, the per-frame many-to-many
model) from scratch on UniTalk-ASD + AVA-ActiveSpeaker combined
(src/dataset.py's extra_sources=[ava_source(split)]) — a fresh run, NOT
fine-tuning scripts/modeling/train/frames.py's existing checkpoint. Otherwise structurally
identical to that script (same architecture, optimizer, masked loss, early
stopping) so the two are directly comparable — this is meant to answer
"does adding the AVA subset help the per-frame model the way it helped the
windowed one?" (see [[ava-dataset-integration]] in project memory for that
story — F1 0.6735->0.7151 combined with a capacity increase), not to
introduce confounding changes alongside the new data.

Unlike scripts/modeling/train/frames.py, this uses a fixed seed (config.WINDOWED_SEED,
reused rather than adding a near-duplicate constant — the per-frame
pipeline never had its own seed before this) plus full determinism
(cudnn.deterministic, use_deterministic_algorithms(True),
CUBLAS_WORKSPACE_CONFIG — see scripts/modeling/train/windowed.py's matching comment
for why a seed alone isn't sufficient on GPU), so this specific run is
exactly reproducible, unlike every prior scripts/modeling/train/frames.py run.

pos_weight is computed from the combined train split's own masked-frame
class distribution (not UniTalk's alone) — it naturally reflects whatever
the combined class balance actually is, no special-casing needed since
compute_pos_weight just reads train_loader.dataset.labels/masks.

Own checkpoint/log filenames (config.COMBINED_*), entirely separate from
scripts/modeling/train/frames.py's UniTalk-only files, so this run never overwrites that
one — the two are meant to sit side by side for comparison.

This script deliberately does NOT touch the held-out "test" split — that's
scripts/modeling/evaluate/frames_combined.py's job, writing to its own
evaluation_combined/ folder.

Checkpoints -> checkpoints/combined_{best,last}_model.pt (gitignored).
Per-epoch metrics -> logs/combined_train_metrics.csv (gitignored).
Human-readable run log -> logs/combined_train.log (gitignored).

Usage:
    python scripts/modeling/train/frames_combined.py
"""
from __future__ import annotations

import csv
import logging
import os
import sys
import time
from pathlib import Path

# Must be set before CUDA is initialized (first torch.cuda.* call below) —
# see scripts/modeling/train/windowed.py's matching comment.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from config import (
    BALANCED_BATCHES,
    CHECKPOINTS_DIR,
    COMBINED_ACCURACY_CURVE_FILENAME,
    COMBINED_BEST_CHECKPOINT_FILENAME,
    COMBINED_LAST_CHECKPOINT_FILENAME,
    COMBINED_LOSS_CURVE_FILENAME,
    COMBINED_TRAIN_LOG_FILENAME,
    COMBINED_TRAIN_METRICS_FILENAME,
    COMBINED_TRAINING_REPORT_FILENAME,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    LOGS_DIR,
    NUM_EPOCHS,
    TORCH_CPU_THREADS,
    USE_MAR,
    USE_MOUTH_LANDMARKS_ONLY,
    WINDOWED_SEED,
)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))
from dataset import NUM_INPUT_FEATURES, ava_source, get_dataloader
from model import SpeakingDetectorRNN
from metrics import build_training_report, plot_accuracy_curve, plot_loss_curve
from train_utils import reset_dir

if not torch.cuda.is_available():
    torch.set_num_threads(TORCH_CPU_THREADS)  # see config.py — avoids CPU thread-contention on this 64-core box

logger = logging.getLogger("train_frames_combined")


def masked_bce_loss(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor | None = None
) -> torch.Tensor:
    per_frame = nn.functional.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction="none", pos_weight=pos_weight
    )
    return (per_frame * mask).sum() / mask.sum().clamp(min=1)


def compute_pos_weight(dataset) -> float:
    """num_negative / num_positive over the combined train split's masked
    (real, detected) frames — same rationale as scripts/modeling/train/frames.py's version,
    just over the concatenated UniTalk-ASD + AVA track pool.
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
    # Seed + full determinism — reused from the windowed pipeline (see
    # scripts/modeling/train/windowed.py's matching comment for why a seed alone
    # isn't sufficient on GPU). Per-frame's own train.py has never had this;
    # added here so this combined run is at least exactly reproducible,
    # even though train.py's own UniTalk-only results predate it.
    torch.manual_seed(WINDOWED_SEED)
    np.random.seed(WINDOWED_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Every run starts from a clean slate — deletes and recreates both
    # folders, clearing any prior run's checkpoints/logs (see
    # src/train_utils.py; CHECKPOINTS_DIR/LOGS_DIR are shared across every
    # training script, not just this one).
    reset_dir(CHECKPOINTS_DIR)
    reset_dir(LOGS_DIR)
    setup_logging(LOGS_DIR / COMBINED_TRAIN_LOG_FILENAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    logger.info(f"seed: {WINDOWED_SEED}")
    logger.info(
        f"features: mouth_landmarks_only={USE_MOUTH_LANDMARKS_ONLY} use_mar={USE_MAR} "
        f"(NUM_INPUT_FEATURES={NUM_INPUT_FEATURES})"
    )
    logger.info(f"balanced_batches: {BALANCED_BATCHES}")
    logger.info("data sources: UniTalk-ASD + AVA-ActiveSpeaker (combined, from scratch)")

    train_loader = get_dataloader("train", extra_sources=[ava_source("train")])
    val_loader = get_dataloader("val", extra_sources=[ava_source("val")])

    pos_weight_value = compute_pos_weight(train_loader.dataset)
    pos_weight = torch.tensor(pos_weight_value, device=device)
    logger.info(
        f"pos_weight (combined train split NOT_SPEAKING:SPEAKING_AUDIBLE ratio, masked frames): "
        f"{pos_weight_value:.4f}"
    )

    model = SpeakingDetectorRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_checkpoint_path = CHECKPOINTS_DIR / COMBINED_BEST_CHECKPOINT_FILENAME
    last_checkpoint_path = CHECKPOINTS_DIR / COMBINED_LAST_CHECKPOINT_FILENAME
    metrics_path = LOGS_DIR / COMBINED_TRAIN_METRICS_FILENAME

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
    # scripts/evaluate_combined.py separately for the held-out check.
    metrics_df = pd.read_csv(metrics_path)
    plot_loss_curve(metrics_df, best_epoch=best_epoch, save_path=LOGS_DIR / COMBINED_LOSS_CURVE_FILENAME)
    plot_accuracy_curve(metrics_df, best_epoch=best_epoch, save_path=LOGS_DIR / COMBINED_ACCURACY_CURVE_FILENAME)
    logger.info(
        f"wrote {LOGS_DIR / COMBINED_LOSS_CURVE_FILENAME} and {LOGS_DIR / COMBINED_ACCURACY_CURVE_FILENAME}"
    )

    report = build_training_report(
        metrics_df, best_epoch, EARLY_STOPPING_PATIENCE,
        loss_curve_filename=COMBINED_LOSS_CURVE_FILENAME,
        accuracy_curve_filename=COMBINED_ACCURACY_CURVE_FILENAME,
    )
    report_path = LOGS_DIR / COMBINED_TRAINING_REPORT_FILENAME
    report_path.write_text(report)
    logger.info(f"wrote {report_path}")


if __name__ == "__main__":
    main()
