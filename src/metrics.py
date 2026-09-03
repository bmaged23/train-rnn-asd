"""Classification-metrics computation and plotting helpers.

Two independent families of helpers, kept deliberately separate — see
[[analysis-outputs-in-logs]]-style project convention: training progress
(train/val) and the held-out test-split check are two different concerns
with two different output folders:
  - `compute_metrics`/`format_report` + `plot_confusion_matrix`/`plot_roc_curve`/
    `plot_pr_curve`: consume flat (masked, per-frame) label/prediction/probability
    arrays — see scripts/modeling/evaluate/frames.py for how those are produced from a model's
    per-frame logits — and turn them into the standard binary-classification
    report (accuracy, precision, recall, F1, confusion matrix, ROC-AUC, PR-AUC).
    Used only by scripts/modeling/evaluate/frames.py (hardcoded to the "test" split, writes to
    evaluation/) and notebooks/04_training_analysis.ipynb (renders inline).
  - `plot_loss_curve`/`plot_accuracy_curve`/`build_training_report`: consume
    the per-epoch train/val history in logs/train_metrics.csv and chart
    progress across the whole run — no test-split data involved. Used by
    scripts/modeling/train/frames.py right after training finishes, writing to logs/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import CLASS_NAMES

# Fixed categorical slots (blue, green) from the project's validated palette —
# same two hues used consistently for "train" vs "val"/"positive" series
# across the training-curve and evaluation plots.
COLOR_BLUE = "#2a78d6"
COLOR_GREEN = "#008300"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
# Single-hue sequential ramp (light -> dark blue) for the confusion-matrix heatmap.
SEQUENTIAL_BLUES = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """y_true/y_pred: 0/1 int arrays. y_prob: predicted probability of class 1
    (SPEAKING_AUDIBLE), i.e. sigmoid(logits). Returns a JSON-serializable dict
    (aside from the "_roc_curve"/"_pr_curve" plotting arrays — strip those
    before json.dump, see scripts/modeling/evaluate/frames.py).
    """
    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    accuracy = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    return {
        "n_samples": int(len(y_true)),
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),  # rows=true, cols=pred, order=CLASS_NAMES
        "class_names": CLASS_NAMES,
        "_roc_curve": (fpr, tpr),
        "_pr_curve": (pr_recall, pr_precision),
    }


def format_report(metrics: dict) -> str:
    cm = np.array(metrics["confusion_matrix"])
    class_names = metrics["class_names"]
    return "\n".join([
        f"n_samples={metrics['n_samples']}",
        f"accuracy={metrics['accuracy']:.4f}",
        f"precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  "
        f"f1={metrics['f1']:.4f}",
        f"roc_auc={metrics['roc_auc']:.4f}  pr_auc={metrics['pr_auc']:.4f}",
        "confusion matrix (rows=true, cols=pred):",
        f"                    pred_{class_names[0]:<16} pred_{class_names[1]}",
        f"  true_{class_names[0]:<14} {cm[0, 0]:>10} {cm[0, 1]:>20}",
        f"  true_{class_names[1]:<14} {cm[1, 0]:>10} {cm[1, 1]:>20}",
    ])


def _style_axes(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED)
    ax.grid(color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_confusion_matrix(metrics: dict, split: str = "test", ax=None, save_path=None):
    cm = np.array(metrics["confusion_matrix"])
    cm_pct = cm / cm.sum(axis=1, keepdims=True)  # row-normalized (recall per true class)
    class_names = metrics["class_names"]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUES)
    im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks([0, 1], class_names, rotation=15, ha="right")
    ax.set_yticks([0, 1], class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({split} split)")

    for i in range(2):
        for j in range(2):
            text_color = "#ffffff" if cm_pct[i, j] > 0.5 else "#0b0b0b"
            ax.text(
                j, i, f"{cm[i, j]:,}\n({cm_pct[i, j]:.1%})",
                ha="center", va="center", color=text_color, fontsize=10,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of true class")
    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curve(metrics: dict, split: str = "test", ax=None, save_path=None):
    fpr, tpr = metrics["_roc_curve"]
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    else:
        fig = ax.figure

    ax.plot(fpr, tpr, color=COLOR_BLUE, linewidth=2, label=f"ROC (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=COLOR_MUTED, linewidth=1, linestyle="--", label="chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve ({split} split)")
    ax.legend(frameon=False, loc="lower right")
    _style_axes(ax)

    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pr_curve(metrics: dict, split: str = "test", ax=None, save_path=None):
    recall, precision = metrics["_pr_curve"]
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    else:
        fig = ax.figure

    ax.plot(recall, precision, color=COLOR_GREEN, linewidth=2, label=f"PR (AUC={metrics['pr_auc']:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curve ({split} split)")
    ax.legend(frameon=False, loc="lower left")
    _style_axes(ax)

    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _plot_metric_curve(
    metrics_df, metric: str, ylabel: str, title: str,
    best_epoch: int | None = None, ax=None, save_path=None,
):
    """metric: "loss" or "acc" — reads train_{metric}/val_{metric} columns
    from metrics_df (logs/train_metrics.csv, one row per epoch)."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    ax.plot(metrics_df["epoch"], metrics_df[f"train_{metric}"], color=COLOR_BLUE, linewidth=2, label="train")
    ax.plot(metrics_df["epoch"], metrics_df[f"val_{metric}"], color=COLOR_GREEN, linewidth=2, label="val")
    if best_epoch is not None:
        ax.axvline(best_epoch, color=COLOR_MUTED, linewidth=1, linestyle="--")
        ax.annotate(
            f"best epoch {best_epoch}", (best_epoch, ax.get_ylim()[1]), xytext=(4, -4),
            textcoords="offset points", color=COLOR_MUTED, fontsize=9, va="top",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    _style_axes(ax)

    if own_fig:
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_loss_curve(metrics_df, best_epoch: int | None = None, ax=None, save_path=None):
    """Train vs. val masked-BCE loss across all epochs of a training run."""
    return _plot_metric_curve(
        metrics_df, "loss", "masked BCE loss", "Loss per epoch", best_epoch, ax, save_path
    )


def plot_accuracy_curve(metrics_df, best_epoch: int | None = None, ax=None, save_path=None):
    """Train vs. val masked accuracy across all epochs of a training run."""
    return _plot_metric_curve(
        metrics_df, "acc", "masked accuracy", "Accuracy per epoch", best_epoch, ax, save_path
    )


def build_training_report(
    metrics_df,
    best_epoch: int,
    patience: int,
    *,
    loss_curve_filename: str,
    accuracy_curve_filename: str,
) -> str:
    """Markdown write-up of train-vs-val progress across the run just
    finished — written to logs/README.md by scripts/modeling/train/frames.py. Training-
    progress only: no test-split data touches this (see scripts/modeling/evaluate/frames.py
    for the separate, deliberately-standalone classification-metrics check
    against the held-out "test" split, written to evaluation/ instead).
    Image links are relative (same directory), so this reads correctly when
    logs/ is opened directly.
    """
    best_row = metrics_df.loc[metrics_df["epoch"] == best_epoch].iloc[0]
    last_epoch = int(metrics_df["epoch"].max())
    gap = metrics_df["train_acc"] - metrics_df["val_acc"]
    gap_at_best = float(gap[metrics_df["epoch"] == best_epoch].iloc[0])
    gap_at_last = float(gap.iloc[-1])
    trend = "widened" if gap_at_last > gap_at_best else "did not widen"

    return f"""# Training Progress Report — SpeakingDetectorRNN

Generated automatically by `scripts/modeling/train/frames.py` right after this run finished. Train-vs-val
progress only — for the held-out test-split classification metrics (accuracy/precision/recall/
F1/confusion matrix/ROC-AUC/PR-AUC), run `scripts/modeling/evaluate/frames.py` separately (writes to `evaluation/`).

## Run summary

- Epochs trained: **{last_epoch}** (early stopping patience={patience})
- Best epoch: **{best_epoch}** — val_loss={best_row['val_loss']:.4f}, val_acc={best_row['val_acc']:.4f}
  (this is the checkpoint saved as `checkpoints/best_model.pt`)
- Train/val generalization gap (train_acc − val_acc): {gap_at_best:+.4f} at the best epoch,
  {gap_at_last:+.4f} at the final epoch — gap {trend} after the best epoch.

## Training curves

![Loss curve]({loss_curve_filename})

![Accuracy curve]({accuracy_curve_filename})
"""
