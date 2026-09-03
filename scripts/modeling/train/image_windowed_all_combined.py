"""Trains ImageWindowedSpeakingDetectorRNN (src/image_model.py) — the raw-pixel
CNN+LSTM candidate, third architecture alongside the per-frame and windowed
landmark-RNN models — from scratch on UniTalk-ASD + AVA + WASD + Columbia +
AVSpeech + SpeakingFaces combined (src/image_dataset.py's
extra_sources=[ava_source(split), ...], same six sources
scripts/modeling/train/windowed_all_combined.py trains on). See
src/image_dataset.py's module docstring for how each source's face-crop
images are located (SpeakingFaces uses the face-cropped mirror,
scripts/dataset/crop_speakingfaces_faces.py's output, not its raw uncropped
frames) and src/image_model.py's for the CNN+LSTM architecture itself.

Structurally identical to scripts/modeling/train/windowed_all_combined.py
(same seed, determinism settings, optimizer, early stopping, pos_weight
computed from the combined train split's own window-label distribution) so
the two architectures' results are directly comparable — only the dataset/
model/config constants differ (IMAGE_* instead of ALL_WINDOWED_*).

Own checkpoint/log filenames (config.IMAGE_ALL_WINDOWED_*) — but CHECKPOINTS_DIR/
LOGS_DIR themselves are shared across every training script in this project
(see src/train_utils.py), so running this wipes whatever checkpoints/logs
the last training run left there, same as every other train/*.py script.
Evaluate the previous run first if you need to keep its results.

This script deliberately does NOT touch the held-out "test" split — that's
scripts/modeling/evaluate/image_windowed_all_combined.py's job.

Checkpoints -> checkpoints/image_all_windowed_{best,last}_model.pt (gitignored).
Per-epoch metrics -> logs/image_all_windowed_train_metrics.csv (gitignored).
Human-readable run log -> logs/image_all_windowed_train.log (gitignored).

Usage:
    python scripts/modeling/train/image_windowed_all_combined.py
"""
from __future__ import annotations

import csv
import logging
import os
import sys
import time
from pathlib import Path

# Must be set before CUDA is initialized — see scripts/modeling/train/windowed.py's
# matching comment.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from config import (
    APPLY_POS_WEIGHT,
    AUGMENT_IMAGES,
    BALANCED_BATCHES,
    CHECKPOINTS_DIR,
    IMAGE_AUGMENT_COPIES_PER_WINDOW,
    IMAGE_AUGMENT_ON_THE_FLY,
    IMAGE_AUGMENT_SHIFT_SECONDS,
    IMAGE_EARLY_STOPPING_PATIENCE,
    IMAGE_VAL_NUM_DATALOADER_WORKERS,
    GRADIENT_CLIP_NORM,
    IMAGE_ALL_WINDOWED_ACCURACY_CURVE_FILENAME,
    IMAGE_ALL_WINDOWED_BEST_CHECKPOINT_FILENAME,
    IMAGE_ALL_WINDOWED_LAST_CHECKPOINT_FILENAME,
    IMAGE_ALL_WINDOWED_LOSS_CURVE_FILENAME,
    IMAGE_ALL_WINDOWED_TRAIN_LOG_FILENAME,
    IMAGE_ALL_WINDOWED_TRAIN_METRICS_FILENAME,
    IMAGE_ALL_WINDOWED_TRAINING_REPORT_FILENAME,
    IMAGE_BATCH_SIZE,
    IMAGE_NUM_DATALOADER_WORKERS,
    IMAGE_RESIZED_CACHE_DIR,
    IMAGE_SIZE,
    IMAGE_WINDOWED_SEED,
    LEARNING_RATE,
    LOGS_DIR,
    NUM_EPOCHS,
    TORCH_CPU_THREADS,
    WINDOW_SIZE,
)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))
from dataset import ava_source, avspeech_source, columbia_source, speakingfaces_source, wasd_source
from image_dataset import build_image_cache, get_image_windowed_dataloader
from image_model import ImageWindowedSpeakingDetectorRNN
from metrics import build_training_report, plot_accuracy_curve, plot_loss_curve
from train_utils import reset_dir

if not torch.cuda.is_available():
    torch.set_num_threads(TORCH_CPU_THREADS)  # see config.py — avoids CPU thread-contention on this 64-core box

logger = logging.getLogger("train_image_windowed_all_combined")


def compute_pos_weight(labels: np.ndarray) -> float:
    """num_negative / num_positive over a flat array of window labels (0/1) —
    same rationale as scripts/modeling/train/windowed.py's compute_pos_weight.
    """
    num_pos = (labels == 1).sum()
    num_neg = (labels == 0).sum()
    return float(num_neg / num_pos)


@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (logits > 0).long()
    return (preds == labels).float().mean().item()


def _strip_compile_prefix(state_dict: dict) -> dict:
    """torch.compile(model.backbone)/torch.compile(model.adaface_backbone)
    (see main()) prefix every wrapped parameter's state_dict key with
    "_orig_mod." — verified directly (2026-08-25) against this exact model.
    Stripped before every checkpoint save so the file stays loadable by the
    evaluate script's plain, uncompiled ImageWindowedSpeakingDetectorRNN()
    + load_state_dict(), same "don't let an internal wrapper's naming leak
    into a saved checkpoint" fix already applied once before for
    nn.DataParallel's "module." prefix.
    """
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def run_epoch(
    model: ImageWindowedSpeakingDetectorRNN,
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

    Mixed precision (added 2026-08-25, explicit user request — the biggest
    real speed lever for this workload: AdaFace is ~92% of this model's
    forward compute, per the branch-timing benchmark that ruled out
    multi-GPU, and it's almost entirely convolutions, exactly what AMP
    helps most). bfloat16 over float16 deliberately — same dynamic range as
    float32 so it needs no GradScaler (float16's narrower range risks
    gradient under/overflow without one, adding real complexity for a GPU
    generation, RTX PRO 6000/Blackwell, that runs bfloat16 at full
    tensor-core speed anyway, unlike older GPUs where float16 was the only
    fast option). autocast wraps forward+loss only, per PyTorch's own
    documented pattern — backward() ops inherit each op's forward dtype
    automatically, no separate handling needed. Applied identically for
    train AND val (correctness unaffected either way; consistent numerics
    make train/val loss comparable).
    """
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    for batch in tqdm(loader, desc=desc, leave=False):
        # non_blocking=True only actually overlaps with compute when the
        # source tensor is pinned (get_image_windowed_dataloader's
        # pin_memory=True) — harmless no-op otherwise.
        features = batch["features"].to(device, non_blocking=True)
        lengths = batch["lengths"]
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train_mode), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(features, lengths)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits.float(), labels.float(), pos_weight=pos_weight if train_mode else None
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
    # Same seed + determinism setup as the landmark windowed_all_combined.py —
    # not what makes the landmark-vs-image comparison meaningful (they're
    # different architectures/inputs regardless of seed), but keeps this run
    # itself reproducible.
    torch.manual_seed(IMAGE_WINDOWED_SEED)
    np.random.seed(IMAGE_WINDOWED_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # TF32 (added 2026-08-25, explicit user request — real, essentially
    # free speed for whatever stays in float32 on top of the bfloat16 AMP
    # autocast in run_epoch(): reduced-mantissa float32 matmuls/convs on
    # Ampere+ GPUs, still a DETERMINISTIC algorithm (same inputs -> same
    # outputs), so this doesn't conflict with use_deterministic_algorithms
    # above — it trades precision, not reproducibility.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Every run starts from a clean slate — deletes and recreates both
    # folders, clearing any prior run's checkpoints/logs (see
    # src/train_utils.py; CHECKPOINTS_DIR/LOGS_DIR are shared across every
    # training script, not just this one).
    reset_dir(CHECKPOINTS_DIR)
    reset_dir(LOGS_DIR)
    setup_logging(LOGS_DIR / IMAGE_ALL_WINDOWED_TRAIN_LOG_FILENAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    logger.info(f"seed: {IMAGE_WINDOWED_SEED}")
    logger.info(f"image_size={IMAGE_SIZE}  window_size={WINDOW_SIZE}  batch_size={IMAGE_BATCH_SIZE}")
    logger.info(f"balanced_batches: {BALANCED_BATCHES}")
    logger.info(
        f"augment_images: {AUGMENT_IMAGES}  on_the_fly: {IMAGE_AUGMENT_ON_THE_FLY}  "
        f"(materialized copies_per_window={IMAGE_AUGMENT_COPIES_PER_WINDOW} shift_seconds={IMAGE_AUGMENT_SHIFT_SECONDS})"
    )
    logger.info(
        "data sources: UniTalk-ASD + AVA-ActiveSpeaker + WASD + Columbia + AVSpeech + SpeakingFaces "
        "(combined, from scratch, raw-pixel CNN+LSTM)"
    )

    # Ensures config.IMAGE_RESIZED_CACHE_DIR is built (train+val+test, all 6
    # combined sources) before touching any dataloader — moves the
    # decode+pad-to-square+resize work that used to happen inline in
    # _load_image, every epoch, out of the per-epoch hot path entirely (see
    # src/image_dataset.py's build_image_cache() docstring). Reuses an
    # existing valid cache when settings (IMAGE_SIZE/IMAGE_MIN_CROP_SIZE)
    # haven't changed since it was last built — see IMAGE_RESIZED_CACHE_DIR's
    # config.py comment for why this was reversed from an initial
    # always-rebuild-fresh choice (a from-scratch build takes ~25 min at
    # this project's full 6-source/3-split scale).
    t_cache0 = time.time()
    build_image_cache()
    logger.info(f"image cache ready in {time.time() - t_cache0:.1f}s -> {IMAGE_RESIZED_CACHE_DIR}")

    # Separate, right-sized worker counts for train vs val — see
    # config.IMAGE_VAL_NUM_DATALOADER_WORKERS' comment: both loaders are
    # built up front and (with persistent_workers=True) BOTH worker pools
    # stay resident for the whole run, so reusing the large train worker
    # count for the much-smaller val split would double memory pressure for
    # no throughput benefit.
    train_loader = get_image_windowed_dataloader(
        "train",
        batch_size=IMAGE_BATCH_SIZE,
        num_workers=IMAGE_NUM_DATALOADER_WORKERS,
        extra_sources=[
            ava_source("train"), wasd_source("train"), columbia_source("train"),
            avspeech_source("train"), speakingfaces_source("train"),
        ],
    )
    val_loader = get_image_windowed_dataloader(
        "val",
        batch_size=IMAGE_BATCH_SIZE,
        num_workers=IMAGE_VAL_NUM_DATALOADER_WORKERS,
        extra_sources=[
            ava_source("val"), wasd_source("val"), columbia_source("val"),
            avspeech_source("val"), speakingfaces_source("val"),
        ],
    )

    train_labels = np.array(train_loader.dataset.labels)
    pos_weight_value = compute_pos_weight(train_labels)
    # config.APPLY_POS_WEIGHT — logged either way so the measured ratio is
    # always visible, even on a run that chooses not to apply it.
    pos_weight = torch.tensor(pos_weight_value, device=device) if APPLY_POS_WEIGHT else None
    logger.info(
        f"pos_weight (combined train split NOT_SPEAKING:SPEAKING_AUDIBLE ratio, window labels): "
        f"{pos_weight_value:.4f}  (apply_pos_weight={APPLY_POS_WEIGHT})"
    )

    model = ImageWindowedSpeakingDetectorRNN().to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"model params: {n_trainable:,} trainable / {n_total:,} total (frozen backbone excluded from trainable)")

    # Single-GPU only (reverted 2026-08-25, explicit user request). Three
    # real multi-GPU strategies were tried and benchmarked on this box —
    # nn.DataParallel (3.5x slower than single-GPU), an optimized version
    # with bulk cross-device transfers (still 2.7x slower), and a
    # branch-split (AdaFace on one GPU, MobileNetV3+LSTM+head on the other,
    # 3.85x slower) — every one measured worse than single-GPU, not better.
    # NCCL itself is also broken on this box (same root cause as the
    # project's known broken-NVML issue: nvmlInit_v2() driver/library
    # mismatch), which additionally rules out nn.DataParallel/DDP outright.
    # Single-GPU at IMAGE_BATCH_SIZE=16 (0.376s/step steady-state,
    # ~3.2h/epoch on the full 6-source combined train set) remains the
    # fastest option actually measured.
    # torch.compile — CNN backbones ONLY, not the whole model (added
    # 2026-08-25, explicit user request, real speedup verified by
    # benchmark: 1.89x on the compiled portion). Deliberately excludes the
    # LSTM/pack_padded_sequence path — dynamic-length sequences are one of
    # torch.compile's known trickier cases, and there was no need to risk
    # it: AdaFace alone is ~92% of this model's forward compute (see the
    # branch-timing benchmark that ruled out branch-split multi-GPU), so
    # compiling just the two CNN backbones already captures nearly all the
    # available win. Real gotcha, verified directly: torch.compile prefixes
    # every wrapped parameter's state_dict key with "_orig_mod." — see
    # _strip_compile_prefix() below, used on every torch.save() call so
    # checkpoints stay loadable by the plain (uncompiled)
    # ImageWindowedSpeakingDetectorRNN() the evaluate script constructs.
    model.backbone = torch.compile(model.backbone)
    model.adaface_backbone = torch.compile(model.adaface_backbone)

    # fused=True (added 2026-08-25, explicit user request): fuses the
    # per-parameter Adam update into fewer, larger CUDA kernels instead of
    # one kernel launch per parameter. Measured no real benefit on its own
    # (optimizer.step() is a tiny fraction of total step time next to the
    # backbone forward/backward) but kept — harmless, and verified to work
    # correctly together with the compiled backbones above (the one
    # documented fused+torch.compile incompatibility is specifically about
    # compiling optimizer.step() itself, which this project doesn't do).
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, fused=True
    )

    best_checkpoint_path = CHECKPOINTS_DIR / IMAGE_ALL_WINDOWED_BEST_CHECKPOINT_FILENAME
    last_checkpoint_path = CHECKPOINTS_DIR / IMAGE_ALL_WINDOWED_LAST_CHECKPOINT_FILENAME
    metrics_path = LOGS_DIR / IMAGE_ALL_WINDOWED_TRAIN_METRICS_FILENAME

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

            torch.save(_strip_compile_prefix(model.state_dict()), last_checkpoint_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(_strip_compile_prefix(model.state_dict()), best_checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= IMAGE_EARLY_STOPPING_PATIENCE:
                    logger.info(
                        f"early stopping: val_loss hasn't improved for "
                        f"{IMAGE_EARLY_STOPPING_PATIENCE} epochs (best={best_val_loss:.4f})"
                    )
                    break

    # --- Combined train/val progress charts, across every epoch of this run ---
    # Deliberately no "test" split here — see module docstring. Run
    # scripts/modeling/evaluate/image_windowed_all_combined.py separately for the held-out check.
    metrics_df = pd.read_csv(metrics_path)
    plot_loss_curve(metrics_df, best_epoch=best_epoch, save_path=LOGS_DIR / IMAGE_ALL_WINDOWED_LOSS_CURVE_FILENAME)
    plot_accuracy_curve(metrics_df, best_epoch=best_epoch, save_path=LOGS_DIR / IMAGE_ALL_WINDOWED_ACCURACY_CURVE_FILENAME)
    logger.info(
        f"wrote {LOGS_DIR / IMAGE_ALL_WINDOWED_LOSS_CURVE_FILENAME} and "
        f"{LOGS_DIR / IMAGE_ALL_WINDOWED_ACCURACY_CURVE_FILENAME}"
    )

    report = build_training_report(
        metrics_df, best_epoch, IMAGE_EARLY_STOPPING_PATIENCE,
        loss_curve_filename=IMAGE_ALL_WINDOWED_LOSS_CURVE_FILENAME,
        accuracy_curve_filename=IMAGE_ALL_WINDOWED_ACCURACY_CURVE_FILENAME,
    )
    report_path = LOGS_DIR / IMAGE_ALL_WINDOWED_TRAINING_REPORT_FILENAME
    report_path.write_text(report)
    logger.info(f"wrote {report_path}")


if __name__ == "__main__":
    main()
