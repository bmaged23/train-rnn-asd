"""RNN classifier over fixed-length windows of landmark (+ optional MAR)
frames — the windowed many-to-one counterpart to src/model.py's per-frame
many-to-many model. Deliberately SMALLER than the per-frame model
(config.WINDOWED_HIDDEN_SIZE/WINDOWED_NUM_RNN_LAYERS, not the shared
HIDDEN_SIZE/NUM_RNN_LAYERS) — the windowed model overfit hard even with
identical capacity to the per-frame model, and a weight_decay sweep found no
config that fixed it (see config.py's comment + [[windowed-approach]] in
project memory for the sweep table), so reduced capacity is the
regularization mechanism now, not weight decay or dropout.

Otherwise the same shape as the per-frame model: bidirectional LSTM, but
this outputs one logit for the whole window (config.WINDOW_SIZE frames),
pooled from the LSTM's final hidden state — the standard way to get a single
sequence-level representation from a bidirectional LSTM. Windows are
usually fixed-length with no padding, EXCEPT when config.WINDOW_PAD_SHORT_TRACKS
is on (added 2026-07-17) — then some windows are zero-padded past their true
length (src/windowed_dataset.py), and `forward()`'s optional `lengths` arg
uses pack_padded_sequence so the LSTM's final hidden state is always read at
each window's true end, never influenced by padding. When every window in a
batch has `length == window_size` (padding off, or no short tracks in this
split), passing `lengths` is a functional no-op — this only matters when
padding is actually happening.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DROPOUT, TORCH_CPU_THREADS, WINDOWED_HIDDEN_SIZE, WINDOWED_NUM_RNN_LAYERS
from dataset import NUM_INPUT_FEATURES

if not torch.cuda.is_available():
    torch.set_num_threads(TORCH_CPU_THREADS)  # see config.py — avoids CPU thread-contention on this 64-core box


class WindowedSpeakingDetectorRNN(nn.Module):
    """Bidirectional LSTM + linear head producing one speaking-logit for a
    whole fixed-length window."""

    def __init__(
        self,
        input_size: int = NUM_INPUT_FEATURES,
        hidden_size: int = WINDOWED_HIDDEN_SIZE,
        num_layers: int = WINDOWED_NUM_RNN_LAYERS,
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

    def forward(self, features: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """features: (B, window_size, input_size). `lengths` (B,), optional:
        true per-window frame count — only needed when some windows are
        zero-padded (config.WINDOW_PAD_SHORT_TRACKS); pass batch["lengths"]
        from src/windowed_dataset.py's collate_windows either way, it's
        harmless when nothing is actually padded. Returns (B,) one logit
        per window.
        """
        if lengths is not None:
            packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(features)
        # h_n: (num_layers * 2, B, hidden_size), ordered [layer0_fwd, layer0_bwd, ..., lastN_fwd, lastN_bwd].
        # Last layer's forward + backward final hidden states are the standard
        # sequence-level summary for a bidirectional LSTM — pack_padded_sequence
        # (when used) already ensures these reflect each window's true end,
        # not the zero-padded tail.
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, hidden_size * 2)
        return self.head(final_hidden).squeeze(-1)


if __name__ == "__main__":
    from windowed_dataset import get_windowed_dataloader
    from dataset import LOGICAL_SPLITS

    for split in LOGICAL_SPLITS:
        loader = get_windowed_dataloader(split)
        batch = next(iter(loader))
        model = WindowedSpeakingDetectorRNN()
        logits = model(batch["features"], batch["lengths"])
        assert logits.shape == batch["labels"].shape, (logits.shape, batch["labels"].shape)
        print(f"[{split}] logits shape: {tuple(logits.shape)} (matches labels shape)")
