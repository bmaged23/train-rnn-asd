"""Trains VLMASDClassifier (src/vlm_model.py) — the vision-language-model
candidate, third architecture alongside the per-frame/windowed landmark-RNN
models and the raw-pixel CNN+LSTM ("conv", src/image_model.py) — from scratch
on UniTalk-ASD + AVA + WASD + Columbia + AVSpeech + SpeakingFaces combined
(src/image_dataset.py's extra_sources=[ava_source(split), ...], same six
sources scripts/modeling/train/image_windowed_all_combined.py trains on). See
src/image_dataset.py's module docstring for how each source's face-crop
images are located, and src/vlm_model.py's for the VLM architecture itself
(including the list of assumptions about the real backbone that still need
verification before a real run).

SCAFFOLDED 2026-08-23 — has not actually been trained (needs
transformers/peft installed — see requirements.txt — plus real verification
of src/vlm_model.py's assumptions about config.VLM_PRETRAINED_NAME's real
structure first). Kept as its own fully separate file, NOT a switchable
factory shared with the conv approach — matches this project's normal
one-architecture-per-file convention (per explicit user request; an earlier
draft used a shared model_factory.py + switchable script, reverted).

Structurally near-identical to scripts/modeling/train/image_windowed_all_combined.py
(same seed, determinism settings, early-stopping shape, pos_weight computed
from the combined train split's own window-label distribution, same
src/image_dataset.py data loading) — only the model/optimizer/config
constants differ (VLM_* instead of IMAGE_*), plus a config.VLM_FINETUNE_MODE
branch for the optimizer's learning rate.

Own checkpoint/log filenames (config.VLM_ALL_WINDOWED_*). Training
creates the shared output directories when needed and only overwrites this
approach's named files, preserving other experiments' results.

This script deliberately does NOT touch the held-out "test" split — that's
scripts/modeling/evaluate/vlm_all_combined_per_source.py's job.

Checkpoints -> checkpoints/vlm_all_windowed_{best,last}_model.pt (gitignored).
Per-epoch metrics -> logs/vlm_all_windowed_train_metrics.csv (gitignored).
Human-readable run log -> logs/vlm_all_windowed_train.log (gitignored).

Usage:
    python scripts/modeling/train/vlm_all_combined.py
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
    GRADIENT_CLIP_NORM,
    IMAGE_AUGMENT_COPIES_PER_WINDOW,
    IMAGE_AUGMENT_ON_THE_FLY,
    IMAGE_AUGMENT_SHIFT_SECONDS,
    IMAGE_NUM_DATALOADER_WORKERS,
    IMAGE_RESIZED_CACHE_DIR,
    IMAGE_SIZE,
    IMAGE_VAL_NUM_DATALOADER_WORKERS,
    IMAGE_WINDOWED_SEED,
    LOGS_DIR,
    TORCH_CPU_THREADS,
    VLM_ALL_WINDOWED_ACCURACY_CURVE_FILENAME,
    VLM_ALL_WINDOWED_BEST_CHECKPOINT_FILENAME,
    VLM_ALL_WINDOWED_LAST_CHECKPOINT_FILENAME,
    VLM_ALL_WINDOWED_LOSS_CURVE_FILENAME,
    VLM_ALL_WINDOWED_TRAIN_LOG_FILENAME,
    VLM_ALL_WINDOWED_TRAIN_METRICS_FILENAME,
    VLM_ALL_WINDOWED_TRAINING_REPORT_FILENAME,
    VLM_BATCH_SIZE,
    VLM_EARLY_STOPPING_PATIENCE,
    VLM_FINETUNE_MODE,
    VLM_FULL_FINETUNE_LR,
    VLM_GRADIENT_ACCUMULATION_STEPS,
    VLM_HF_CACHE_DIR,
    VLM_LORA_LR,
    VLM_NUM_EPOCHS,
    VLM_PARTIAL_UNFREEZE_LR,
    VLM_TRAIN_SAMPLE_SIZE_PER_SOURCE,
    VLM_USE_GRADIENT_ACCUMULATION,
    VLM_USE_GRADIENT_CHECKPOINTING,
    VLM_USE_PAGED_OPTIMIZER,
    VLM_VAL_TEST_SAMPLE_SIZE_PER_SOURCE,
    WINDOW_SIZE,
)

# Must be set before transformers/huggingface_hub are imported (both read
# HF_HOME once, at import time) — otherwise they default to
# ~/.cache/huggingface, which sits on this box's root filesystem (found
# nearly full, 869MB free, 2026-08-23) rather than /data. See
# config.VLM_HF_CACHE_DIR's comment for the full story.
os.environ.setdefault("HF_HOME", VLM_HF_CACHE_DIR)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))
from dataset import ava_source, avspeech_source, columbia_source, speakingfaces_source, wasd_source
from image_dataset import build_image_cache, get_image_windowed_dataloader
from metrics import build_training_report, plot_accuracy_curve, plot_loss_curve
from vlm_model import VLMASDClassifier

if not torch.cuda.is_available():
    torch.set_num_threads(TORCH_CPU_THREADS)  # see config.py — avoids CPU thread-contention on this 64-core box

logger = logging.getLogger("train_vlm_all_combined")

_LR_BY_FINETUNE_MODE = {
    "lora": VLM_LORA_LR,
    "full": VLM_FULL_FINETUNE_LR,
    "partial_unfreeze": VLM_PARTIAL_UNFREEZE_LR,
}


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


def run_epoch(
    model: VLMASDClassifier,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
    pos_weight: torch.Tensor | None = None,
    gradient_accumulation_steps: int = 1,
) -> tuple[float, float]:
    """One pass over `loader`. Trains (updates model weights) iff optimizer
    is given; otherwise runs no-grad for eval (val). `pos_weight` is only
    applied when training — val loss stays unweighted so it reflects real
    generalization on the natural class distribution.

    `gradient_accumulation_steps` (added 2026-08-23, explicit user request,
    after VLM_BATCH_SIZE=4 OOM'd on the very first real training batch —
    see config.VLM_GRADIENT_ACCUMULATION_STEPS' comment): rather than
    stepping the optimizer every micro-batch (VLM_BATCH_SIZE windows'
    forward+backward), gradients accumulate over this many micro-batches
    before one optimizer.step() — same "per_device_train_batch_size +
    gradient_accumulation_steps" pattern as HF's Trainer, hand-implemented
    here since this project's training scripts never use Trainer. The loss
    is scaled by 1/gradient_accumulation_steps before backward() so
    accumulated .grad values equal the AVERAGE gradient over the effective
    batch (VLM_BATCH_SIZE * gradient_accumulation_steps windows), matching
    what a single larger batch would have produced — but `total_loss`
    below still accumulates the UNSCALED per-micro-batch loss for correct
    reporting. The last (possibly partial) group of micro-batches in an
    epoch always gets a final flush step, so no accumulated gradient is
    ever silently dropped just because the epoch's batch count doesn't
    divide evenly by gradient_accumulation_steps. Val passes never pass
    this (optimizer is None there, so it's a no-op either way).
    """
    train_mode = optimizer is not None
    model.train(train_mode)
    n_micro_batches = len(loader)
    memory_log_interval = 100  # diagnostic only, train-mode only — see below

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    if train_mode:
        optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
        features = batch["features"].to(device, non_blocking=True)
        lengths = batch["lengths"]
        labels = batch["labels"].to(device, non_blocking=True)

        try:
            with torch.set_grad_enabled(train_mode):
                logits = model(features, lengths)
                # logits lands on whichever physical device holds the model's
                # LAST shard under device_map="auto" (src/vlm_model.py,
                # 2026-08-23) — not necessarily `device` (cuda:0) above, so
                # labels/pos_weight must follow logits rather than assume a
                # fixed device, or this errors with a cross-device mismatch.
                labels = labels.to(logits.device)
                batch_pos_weight = pos_weight.to(logits.device) if (train_mode and pos_weight is not None) else None
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels.float(), pos_weight=batch_pos_weight
                )

            if train_mode:
                (loss / gradient_accumulation_steps).backward()
                is_last_micro_batch = (step + 1) == n_micro_batches
                if (step + 1) % gradient_accumulation_steps == 0 or is_last_micro_batch:
                    nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                    optimizer.step()
                    optimizer.zero_grad()
        except RuntimeError:
            # Crash-moment diagnostic (added 2026-08-24, explicit user
            # request: "if the train failed check which got the spike, the
            # training or the vllm or what"). Fires on ANY RuntimeError here
            # (covers both a real torch.cuda.OutOfMemoryError and this box's
            # broken-NVML masked "INTERNAL ASSERT FAILED" — see
            # config.VLM_HF_CACHE_DIR's neighboring comments/project memory
            # for that known issue), logs BEFORE re-raising so the exact
            # moment of failure is captured, not just the periodic every-100
            # snapshot above. `reserved` (OUR process's own allocator
            # footprint) vs `free` (true system-wide free, reflects every
            # OTHER process on this shared box too) lets you tell them
            # apart after the fact: reserved close to its recent periodic
            # values but free suddenly near zero -> something ELSE (e.g.
            # the always-on vllm serve process) grabbed the memory: reserved
            # itself spiked -> it was this training run's own batch (real
            # window lengths vary, so a batch's actual sequence length
            # isn't constant — logged too, to check for an unusually long
            # one).
            mem_report = ", ".join(
                f"cuda:{i} reserved={torch.cuda.memory_reserved(i) / 1e9:.2f}GB "
                f"allocated={torch.cuda.memory_allocated(i) / 1e9:.2f}GB "
                f"free={torch.cuda.mem_get_info(i)[0] / 1e9:.2f}GB"
                for i in range(torch.cuda.device_count())
            )
            logger.error(
                f"  [{desc}] CRASH at step {step}/{n_micro_batches}  "
                f"lengths={lengths.tolist()}  {mem_report}"
            )
            raise

        total_loss += loss.item()
        total_acc += accuracy(logits, labels)
        n_batches += 1

        # Diagnostic logging (added 2026-08-24, explicit user request — a
        # real OOM at step 3797/27085 of a batch_size=2 run, masked by this
        # box's broken NVML into an unhelpful INTERNAL ASSERT FAILED). Two
        # different numbers per device, on purpose: memory_reserved is OUR
        # OWN process's caching-allocator footprint (climbs steadily if
        # WE'RE fragmenting/leaking — plausible here since VLM windows have
        # variable real length, so micro-batch tensor shapes vary run to
        # run); mem_get_info's free is the TRUE system-wide free memory,
        # which also reflects whatever any OTHER process on this shared box
        # (e.g. the always-on vllm serve process) is doing. If our own
        # reserved value stays flat while free suddenly drops, that's an
        # external actor, not us — if reserved climbs over time, that's our
        # own fragmentation. Train-mode only (train_mode gate below) since
        # that's where the crash happened; train-only also matches
        # tqdm(desc=...)'s own per-split labeling so these lines are easy to
        # find in the log.
        if train_mode and (step % memory_log_interval == 0):
            mem_report = ", ".join(
                f"cuda:{i} reserved={torch.cuda.memory_reserved(i) / 1e9:.2f}GB "
                f"free={torch.cuda.mem_get_info(i)[0] / 1e9:.2f}GB"
                for i in range(torch.cuda.device_count())
            )
            logger.info(f"  [{desc}] step {step}/{n_micro_batches}: {mem_report}")

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
    # Same seed + determinism setup as the other *_all_combined.py training
    # scripts — not what makes a vlm-vs-conv comparison meaningful (they're
    # different architectures regardless of seed), but keeps this run itself
    # reproducible.
    torch.manual_seed(IMAGE_WINDOWED_SEED)
    np.random.seed(IMAGE_WINDOWED_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # Create shared output folders without deleting other experiments' results.
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    setup_logging(LOGS_DIR / VLM_ALL_WINDOWED_TRAIN_LOG_FILENAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    logger.info(f"finetune_mode: {VLM_FINETUNE_MODE}")
    logger.info(f"seed: {IMAGE_WINDOWED_SEED}")
    logger.info(f"gradient_checkpointing: {VLM_USE_GRADIENT_CHECKPOINTING}")
    # accumulation_steps actually used this run — VLM_USE_GRADIENT_ACCUMULATION=False
    # behaves exactly like steps=1 (step every micro-batch) regardless of
    # VLM_GRADIENT_ACCUMULATION_STEPS' configured value, see that constant's comment.
    accumulation_steps = VLM_GRADIENT_ACCUMULATION_STEPS if VLM_USE_GRADIENT_ACCUMULATION else 1
    logger.info(
        f"image_size={IMAGE_SIZE}  window_size={WINDOW_SIZE}  batch_size={VLM_BATCH_SIZE}  "
        f"gradient_accumulation: {VLM_USE_GRADIENT_ACCUMULATION}  accumulation_steps={accumulation_steps}  "
        f"(effective_batch_size={VLM_BATCH_SIZE * accumulation_steps})"
    )
    logger.info(f"balanced_batches: {BALANCED_BATCHES}")
    logger.info(
        f"augment_images: {AUGMENT_IMAGES}  on_the_fly: {IMAGE_AUGMENT_ON_THE_FLY}  "
        f"(materialized copies_per_window={IMAGE_AUGMENT_COPIES_PER_WINDOW} shift_seconds={IMAGE_AUGMENT_SHIFT_SECONDS})"
    )
    logger.info(
        "data sources: UniTalk-ASD + AVA-ActiveSpeaker + WASD + Columbia + AVSpeech + SpeakingFaces "
        "(combined, from scratch, vision-language-model)"
    )
    logger.info(
        f"sample_size_per_source: train={VLM_TRAIN_SAMPLE_SIZE_PER_SOURCE} "
        f"val={VLM_VAL_TEST_SAMPLE_SIZE_PER_SOURCE} "
        f"(None = full dataset; a number N = N random windows per source)"
    )

    # Same on-disk resized-frame cache the conv pipeline already built —
    # shared regardless of architecture, since both read identical
    # src/image_dataset.py ImageWindowedDataset windows.
    t_cache0 = time.time()
    build_image_cache()
    logger.info(f"image cache ready in {time.time() - t_cache0:.1f}s -> {IMAGE_RESIZED_CACHE_DIR}")

    train_loader = get_image_windowed_dataloader(
        "train",
        batch_size=VLM_BATCH_SIZE,
        num_workers=IMAGE_NUM_DATALOADER_WORKERS,
        extra_sources=[
            ava_source("train"), wasd_source("train"), columbia_source("train"),
            avspeech_source("train"), speakingfaces_source("train"),
        ],
        sample_size_per_source=VLM_TRAIN_SAMPLE_SIZE_PER_SOURCE,
    )
    val_loader = get_image_windowed_dataloader(
        "val",
        batch_size=VLM_BATCH_SIZE,
        num_workers=IMAGE_VAL_NUM_DATALOADER_WORKERS,
        extra_sources=[
            ava_source("val"), wasd_source("val"), columbia_source("val"),
            avspeech_source("val"), speakingfaces_source("val"),
        ],
        sample_size_per_source=VLM_VAL_TEST_SAMPLE_SIZE_PER_SOURCE,
    )

    train_labels = np.array(train_loader.dataset.labels)
    pos_weight_value = compute_pos_weight(train_labels)
    pos_weight = torch.tensor(pos_weight_value, device=device) if APPLY_POS_WEIGHT else None
    logger.info(
        f"pos_weight (combined train split NOT_SPEAKING:SPEAKING_AUDIBLE ratio, window labels): "
        f"{pos_weight_value:.4f}  (apply_pos_weight={APPLY_POS_WEIGHT})"
    )

    # No .to(device) here — VLMASDClassifier's base model is already placed
    # via accelerate's device_map="auto" inside _load_model_and_processor
    # (src/vlm_model.py, 2026-08-23, explicit user request to test spreading
    # the model across both GPUs against this box's tight shared-GPU memory
    # margin). Calling .to(device) on top would collapse the dispatched
    # shards back onto one device, defeating the point.
    model = VLMASDClassifier()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"model params: {n_trainable:,} trainable / {n_total:,} total")
    lr = _LR_BY_FINETUNE_MODE[VLM_FINETUNE_MODE]
    trainable_params = (p for p in model.parameters() if p.requires_grad)
    if VLM_USE_PAGED_OPTIMIZER:
        from bitsandbytes.optim import PagedAdamW32bit

        # weight_decay left at PagedAdamW32bit's own default (0.01) —
        # deliberate choice, explicit user request 2026-08-23, not the
        # torch.optim.Adam default (0) used in the else branch below. 0.01
        # is a standard AdamW/LoRA convention and bitsandbytes' own default,
        # not an arbitrary number — kept even though it means switching
        # VLM_USE_PAGED_OPTIMIZER changes both WHERE optimizer state lives
        # (paged vs. GPU-resident) AND introduces real weight-decay
        # regularization that wasn't present before, not a paging-only change.
        optimizer = PagedAdamW32bit(trainable_params, lr=lr)
    else:
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
    logger.info(
        f"optimizer: {type(optimizer).__name__}  (paged={VLM_USE_PAGED_OPTIMIZER}, "
        f"weight_decay={optimizer.defaults.get('weight_decay')})"
    )

    best_checkpoint_path = CHECKPOINTS_DIR / VLM_ALL_WINDOWED_BEST_CHECKPOINT_FILENAME
    last_checkpoint_path = CHECKPOINTS_DIR / VLM_ALL_WINDOWED_LAST_CHECKPOINT_FILENAME
    metrics_path = LOGS_DIR / VLM_ALL_WINDOWED_TRAIN_METRICS_FILENAME

    best_val_loss = float("inf")
    best_epoch = 1
    epochs_without_improvement = 0

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])

        for epoch in range(1, VLM_NUM_EPOCHS + 1):
            t0 = time.time()
            train_loss, train_acc = run_epoch(
                model, train_loader, device, optimizer, desc=f"epoch {epoch} train", pos_weight=pos_weight,
                gradient_accumulation_steps=accumulation_steps,
            )
            val_loss, val_acc = run_epoch(model, val_loader, device, optimizer=None, desc=f"epoch {epoch} val")
            elapsed = time.time() - t0

            logger.info(
                f"epoch {epoch}/{VLM_NUM_EPOCHS}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({elapsed:.1f}s)"
            )
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, round(elapsed, 1)])
            f.flush()

            # trainable_state_dict(), NOT state_dict() — see that method's
            # docstring: this model's ~4.56B base params are frozen and
            # unchanged from the pretrained checkpoint, so saving the full
            # state_dict() every epoch would mean ~9GB of pure waste per
            # save for no reason.
            torch.save(model.trainable_state_dict(), last_checkpoint_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(model.trainable_state_dict(), best_checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= VLM_EARLY_STOPPING_PATIENCE:
                    logger.info(
                        f"early stopping: val_loss hasn't improved for "
                        f"{VLM_EARLY_STOPPING_PATIENCE} epochs (best={best_val_loss:.4f})"
                    )
                    break

    # --- Combined train/val progress charts, across every epoch of this run ---
    # Deliberately no "test" split here — see module docstring. Run
    # scripts/modeling/evaluate/vlm_all_combined_per_source.py separately for the held-out check.
    metrics_df = pd.read_csv(metrics_path)
    plot_loss_curve(metrics_df, best_epoch=best_epoch, save_path=LOGS_DIR / VLM_ALL_WINDOWED_LOSS_CURVE_FILENAME)
    plot_accuracy_curve(metrics_df, best_epoch=best_epoch, save_path=LOGS_DIR / VLM_ALL_WINDOWED_ACCURACY_CURVE_FILENAME)
    logger.info(
        f"wrote {LOGS_DIR / VLM_ALL_WINDOWED_LOSS_CURVE_FILENAME} and "
        f"{LOGS_DIR / VLM_ALL_WINDOWED_ACCURACY_CURVE_FILENAME}"
    )

    report = build_training_report(
        metrics_df, best_epoch, VLM_EARLY_STOPPING_PATIENCE,
        loss_curve_filename=VLM_ALL_WINDOWED_LOSS_CURVE_FILENAME,
        accuracy_curve_filename=VLM_ALL_WINDOWED_ACCURACY_CURVE_FILENAME,
    )
    report_path = LOGS_DIR / VLM_ALL_WINDOWED_TRAINING_REPORT_FILENAME
    report_path.write_text(report)
    logger.info(f"wrote {report_path}")


if __name__ == "__main__":
    main()
