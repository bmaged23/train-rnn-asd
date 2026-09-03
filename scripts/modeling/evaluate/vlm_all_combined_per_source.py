"""Evaluates a scripts/modeling/train/vlm_all_combined.py checkpoint on the
held-out "test" split, broken down PER SOURCE DATASET — same shape as
scripts/modeling/evaluate/image_windowed_all_combined_per_source.py's report
(full binary-classification metric suite per source, plus "combined" for
every window pooled together), just for VLMASDClassifier instead. Identical
structure to that script — batches by src/image_dataset.py's native
`batch["source_label"]` field, no extra CSV reload needed.

SCAFFOLDED 2026-08-23 — see scripts/modeling/train/vlm_all_combined.py's own
module docstring for what's untested about this approach before a real run.

Writes to its own config.VLM_ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR folder —
reset (deleted and recreated) at the start of every run:
    evaluation_vlm_windowed_all_combined_per_source/{source}_metrics.json  (one per source, plus "combined")
    evaluation_vlm_windowed_all_combined_per_source/eval.log

Does NOT rebuild config.IMAGE_RESIZED_CACHE_DIR itself (unlike
scripts/modeling/train/vlm_all_combined.py, which always rebuilds it fresh)
— this script expects that training run's cache to still be on disk. Run
training first, or scripts/features/build_image_resized_cache.py standalone,
if evaluating without a preceding training run in the same session;
ImageWindowedDataset raises a clear error if the cache is missing.

Usage:
    python scripts/modeling/evaluate/vlm_all_combined_per_source.py [--checkpoint best|last]
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
    CHECKPOINTS_DIR,
    EVAL_LOG_FILENAME,
    IMAGE_VAL_NUM_DATALOADER_WORKERS,
    TORCH_CPU_THREADS,
    VLM_ALL_WINDOWED_BEST_CHECKPOINT_FILENAME,
    VLM_ALL_WINDOWED_LAST_CHECKPOINT_FILENAME,
    VLM_ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR,
    VLM_BATCH_SIZE,
    VLM_FINETUNE_MODE,
    VLM_HF_CACHE_DIR,
    VLM_VAL_TEST_SAMPLE_SIZE_PER_SOURCE,
)

# Must be set before transformers/huggingface_hub are imported — see
# scripts/modeling/train/vlm_all_combined.py's matching comment /
# config.VLM_HF_CACHE_DIR's own comment for the full story.
os.environ.setdefault("HF_HOME", VLM_HF_CACHE_DIR)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))
from dataset import ava_source, avspeech_source, columbia_source, speakingfaces_source, wasd_source
from image_dataset import get_image_windowed_dataloader
from metrics import compute_metrics, format_report
from train_utils import reset_dir
from vlm_model import VLMASDClassifier

if not torch.cuda.is_available():
    torch.set_num_threads(TORCH_CPU_THREADS)  # see config.py — avoids CPU thread-contention on this 64-core box

logger = logging.getLogger("evaluate_vlm_all_combined_per_source")

SPLIT = "test"
_CHECKPOINT_CHOICES = {
    "best": VLM_ALL_WINDOWED_BEST_CHECKPOINT_FILENAME,
    "last": VLM_ALL_WINDOWED_LAST_CHECKPOINT_FILENAME,
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
    model: VLMASDClassifier, loader, device: torch.device
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Runs `model` over every batch, bucketing (y_true, y_prob) by
    batch["source_label"] (native per-window field, see module docstring) —
    returns {source_label: (y_true, y_pred, y_prob)}, plus "combined" for
    every window pooled together.
    """
    model.eval()
    labels_by_source: dict[str, list] = {}
    probs_by_source: dict[str, list] = {}

    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        lengths = batch["lengths"]
        labels = batch["labels"].to(device, non_blocking=True)
        sources = batch["source_label"]

        logits = model(features, lengths)
        # .float() before .cpu().numpy() — logits are bfloat16 (this
        # model's compute dtype), and numpy has no bfloat16 support at all
        # (real error: "TypeError: Got unsupported ScalarType BFloat16").
        # Training never hit this — run_epoch() only ever calls .item() on
        # the loss, never .numpy() on raw logits.
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
    reset_dir(VLM_ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR)
    setup_logging(VLM_ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR / EVAL_LOG_FILENAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    logger.info(f"finetune_mode: {VLM_FINETUNE_MODE}")

    checkpoint_path = CHECKPOINTS_DIR / _CHECKPOINT_CHOICES[checkpoint]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found — run scripts/modeling/train/vlm_all_combined.py first"
        )
    logger.info(f"checkpoint: {checkpoint_path} (split={SPLIT})")
    logger.info(
        f"sample_size_per_source: {VLM_VAL_TEST_SAMPLE_SIZE_PER_SOURCE} "
        f"(None = full dataset; a number N = N random windows per source, same rule as train's val split)"
    )

    # VLMASDClassifier() already loads the real frozen base + fresh LoRA
    # structure from HF — the checkpoint only ever contains the trainable
    # subset (see src/vlm_model.py's trainable_state_dict()), not a full
    # state_dict(), so it layers on top rather than replacing everything.
    #
    # No .to(device) here (fixed 2026-08-24, real bug: a stray cross-device
    # RuntimeError inside a decoder layer's input_layernorm during eval) —
    # same reasoning as the training script's matching fix: the base model
    # is already placed via accelerate's device_map="auto" inside
    # _load_model_and_processor. Calling .to(device) on top physically moves
    # every parameter onto one device while accelerate's forward hooks keep
    # routing activations per the ORIGINAL device_map, so hidden_states and
    # a layer's own weight end up on different devices mid-forward.
    # map_location=device removed for the same reason — torch.load's
    # checkpoint tensors land on CPU by default, and load_state_dict copies
    # values into each parameter in place, respecting wherever THAT
    # parameter's shard actually lives (unaffected by the checkpoint's own
    # loaded-to device), so no explicit map_location is needed here either.
    model = VLMASDClassifier()
    model.load_trainable_state_dict(torch.load(checkpoint_path))

    loader = get_image_windowed_dataloader(
        SPLIT,
        batch_size=VLM_BATCH_SIZE,
        shuffle=False,
        num_workers=IMAGE_VAL_NUM_DATALOADER_WORKERS,  # test split is similarly small to val, see that config comment
        extra_sources=[fn(SPLIT) for fn in _EXTRA_SOURCES],
        sample_size_per_source=VLM_VAL_TEST_SAMPLE_SIZE_PER_SOURCE,
    )
    by_source = collect_predictions_by_source(model, loader, device)

    for source in _SOURCE_ORDER:
        if source not in by_source:
            continue
        y_true, y_pred, y_prob = by_source[source]
        metrics = compute_metrics(y_true, y_pred, y_prob)
        logger.info(f"\n{'=' * 60}\n{source}\n{'=' * 60}\n" + format_report(metrics))

        metrics_path = VLM_ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR / f"{source}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({k: v for k, v in metrics.items() if not k.startswith("_")}, f, indent=2)
    logger.info(f"wrote per-source metrics to {VLM_ALL_WINDOWED_PER_SOURCE_EVALUATION_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="best", choices=sorted(_CHECKPOINT_CHOICES))
    args = parser.parse_args()
    main(checkpoint=args.checkpoint)
