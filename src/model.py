"""RNN classifier over per-frame landmark (+ optional MAR) sequences — which
landmark set and whether MAR is included are config.py toggles, see
src/dataset.py's NUM_INPUT_FEATURES, which this model's default input_size
is derived from.

Consumes the (features, mask, labels, lengths) batches produced by
src/dataset.py's get_dataloader() and predicts one SPEAKING_AUDIBLE /
NOT_SPEAKING logit per frame. Bidirectional LSTM: every pipeline stage here
works over full, already-recorded tracks rather than a live stream, so
there's no reason to give up the future context bidirectionality provides.
If real-time/streaming inference is ever needed, this would have to become
unidirectional (causal) instead — that's a different model, not a flag.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DROPOUT, HIDDEN_SIZE, NUM_RNN_LAYERS, TORCH_CPU_THREADS
from dataset import NUM_INPUT_FEATURES

if not torch.cuda.is_available():
    torch.set_num_threads(TORCH_CPU_THREADS)  # see config.py — avoids CPU thread-contention on this 64-core box


class SpeakingDetectorRNN(nn.Module):
    """Bidirectional LSTM + linear head producing one speaking-logit per frame."""

    def __init__(
        self,
        input_size: int = NUM_INPUT_FEATURES,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_RNN_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size * 2, 1)  # *2: forward+backward concatenated

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """features: (B, T, input_size) zero-padded/masked per src/dataset.py.
        lengths: (B,) true track length per sample (collate_sequences' output —
        includes undetected-but-in-track frames, excludes only batch padding).
        Returns (B, T) per-frame logits.
        """
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=features.size(1))
        return self.head(out).squeeze(-1)


if __name__ == "__main__":
    from dataset import LOGICAL_SPLITS, get_dataloader

    for split in LOGICAL_SPLITS:
        loader = get_dataloader(split)
        batch = next(iter(loader))
        model = SpeakingDetectorRNN()
        logits = model(batch["features"], batch["lengths"])
        assert logits.shape == batch["mask"].shape, (logits.shape, batch["mask"].shape)
        print(f"[{split}] logits shape: {tuple(logits.shape)} (matches mask/labels shape)")
