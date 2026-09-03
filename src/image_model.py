"""CNN+LSTM classifier over fixed-length windows of face-crop IMAGES — the
raw-pixel counterpart to src/windowed_model.py's landmark-feature RNN (see
config.py's "image_windowed_all_combined" section and src/image_dataset.py's
module docstring for the full rationale).

DUAL-BACKBONE per-frame encoder (added 2026-08-24, explicit user request —
see config.IMAGE_USE_ADAFACE_BRANCH's comment for the full rationale): two
CNN backbones run on every frame in parallel, and their embeddings are
CONCATENATED, not one replacing the other.
  - Branch A: config.IMAGE_CNN_BACKBONE (MobileNetV3-Small, ImageNet-
    pretrained, fully trainable by default) -> config.IMAGE_CNN_EMBED_DIM.
  - Branch B: AdaFace IResNet100 (WebFace4M-pretrained face-recognition
    backbone, src/adaface_iresnet.py) -> config.IMAGE_ADAFACE_EMBED_DIM.
    PARTIALLY frozen — Stages 1-3 (46 of 49 residual blocks) frozen, Stage 4
    (last 3 blocks) + its own embedding head stay trainable. Needs its own
    preprocessing path (BGR, 112x112, mean/std=0.5 — adaface_preprocess())
    since it was pretrained under different conventions than Branch A.
The concatenated (window_size, IMAGE_COMBINED_EMBED_DIM) sequence is fed to
a bidirectional LSTM exactly the way WindowedSpeakingDetectorRNN consumes a
(window_size, NUM_INPUT_FEATURES) landmark sequence — same pooling
(final forward+backward hidden state concatenated), same optional
pack_padded_sequence handling for zero-padded short-track windows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import models

sys.path.append(str(Path(__file__).resolve().parent.parent))
from adaface_iresnet import adaface_preprocess, load_adaface_backbone
from config import (
    DROPOUT,
    IMAGE_ADAFACE_CHECKPOINT_PATH,
    IMAGE_ADAFACE_NUM_LAYERS,
    IMAGE_ADAFACE_PARTIAL_UNFREEZE_NUM_BLOCKS,
    IMAGE_CNN_BACKBONE,
    IMAGE_CNN_FREEZE_BACKBONE,
    IMAGE_COMBINED_EMBED_DIM,
    IMAGE_HIDDEN_SIZE,
    IMAGE_NUM_RNN_LAYERS,
    IMAGE_USE_ADAFACE_BRANCH,
    TORCH_CPU_THREADS,
)

if not torch.cuda.is_available():
    torch.set_num_threads(TORCH_CPU_THREADS)  # see config.py — avoids CPU thread-contention on this 64-core box


def _build_backbone(name: str) -> nn.Module:
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        net = models.mobilenet_v3_small(weights=weights)
        return net.features  # conv stack only, no classifier head — pooled to embed_dim below
    raise ValueError(f"unsupported IMAGE_CNN_BACKBONE: {name!r}")


class ImageWindowedSpeakingDetectorRNN(nn.Module):
    """CNN frame encoder (frozen ImageNet backbone by default) + bidirectional
    LSTM + linear head producing one speaking-logit for a whole fixed-length
    window of images.
    """

    def __init__(
        self,
        backbone_name: str = IMAGE_CNN_BACKBONE,
        embed_dim: int = IMAGE_COMBINED_EMBED_DIM,
        freeze_backbone: bool = IMAGE_CNN_FREEZE_BACKBONE,
        hidden_size: int = IMAGE_HIDDEN_SIZE,
        num_layers: int = IMAGE_NUM_RNN_LAYERS,
        dropout: float = DROPOUT,
        use_adaface_branch: bool = IMAGE_USE_ADAFACE_BRANCH,
        adaface_checkpoint_path: Path = IMAGE_ADAFACE_CHECKPOINT_PATH,
        adaface_num_layers: int = IMAGE_ADAFACE_NUM_LAYERS,
        adaface_partial_unfreeze_num_blocks: int = IMAGE_ADAFACE_PARTIAL_UNFREEZE_NUM_BLOCKS,
    ) -> None:
        super().__init__()
        self.backbone = _build_backbone(backbone_name)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()  # frozen -> BatchNorm running stats must never update, even during model.train()

        # Branch B: AdaFace (see module docstring). Own list of always-eval
        # submodules (`_adaface_frozen_modules`), separate from
        # `freeze_backbone` above which only ever applied to Branch A —
        # AdaFace's freeze is PARTIAL, not all-or-nothing, so it needs its
        # own bookkeeping rather than reusing that single boolean.
        self.use_adaface_branch = use_adaface_branch
        self._adaface_frozen_modules: list[nn.Module] = []
        if use_adaface_branch:
            self.adaface_backbone = load_adaface_backbone(adaface_checkpoint_path, adaface_num_layers)
            total_blocks = len(self.adaface_backbone.body)
            num_frozen_blocks = total_blocks - adaface_partial_unfreeze_num_blocks
            assert 0 <= num_frozen_blocks <= total_blocks, (
                f"adaface_partial_unfreeze_num_blocks={adaface_partial_unfreeze_num_blocks} "
                f"exceeds total_blocks={total_blocks}"
            )
            self._adaface_frozen_modules = [self.adaface_backbone.input_layer] + [
                self.adaface_backbone.body[i] for i in range(num_frozen_blocks)
            ]
            for module in self._adaface_frozen_modules:
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()  # frozen -> BatchNorm running stats must never update, even during model.train()
            # Stage 4 (last adaface_partial_unfreeze_num_blocks blocks) +
            # output_layer (embedding head) are left as-is: requires_grad
            # already True by default (fresh from load_adaface_backbone,
            # nothing above touched them), normal train()/eval() toggling.

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size * 2, 1)  # *2: forward+backward concatenated

    def train(self, mode: bool = True) -> "ImageWindowedSpeakingDetectorRNN":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()  # override — never let a model.train() call put the frozen backbone back in train mode
        for module in self._adaface_frozen_modules:
            module.eval()  # same override, for AdaFace's frozen Stages 1-3 (Stage 4 + output_layer follow `mode` normally)
        return self

    def encode_frames(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, window_size, 3, H, W) -> (B, window_size, embed_dim).
        Flattens batch and window_size into one axis for a single CNN forward
        pass over all frames at once, rather than looping per-frame. When
        the AdaFace branch is enabled, both backbones run on the SAME flat
        frame batch (each with its own preprocessing) and their per-frame
        embeddings are concatenated — see module docstring.
        """
        b, t, c, h, w = images.shape
        flat = images.reshape(b * t, c, h, w)
        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.pool(self.backbone(flat)).flatten(1)
        else:
            feats = self.pool(self.backbone(flat)).flatten(1)

        if self.use_adaface_branch:
            # No torch.no_grad() wrapper here even though most of
            # adaface_backbone is frozen — Stage 4 + output_layer ARE
            # trainable and sit downstream in the same forward call, so the
            # graph must stay connected through them. Autograd already
            # skips storing .grad for the frozen params (requires_grad=False,
            # set once in __init__) — this is the same partial-fine-tune
            # pattern src/vlm_model.py's finetune_mode="partial_unfreeze"
            # already uses, just for a CNN instead of a transformer.
            adaface_feats = self.adaface_backbone(adaface_preprocess(flat))
            feats = torch.cat([feats, adaface_feats], dim=1)

        return feats.reshape(b, t, -1)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """features: (B, window_size, 3, H, W) raw image tensors (already
        resized/normalized by src/image_dataset.py). `lengths` (B,), optional:
        true per-window frame count — only needed when some windows are
        zero-padded (config.WINDOW_PAD_SHORT_TRACKS); pass batch["lengths"]
        from src/image_dataset.py's collate_images either way, it's harmless
        when nothing is actually padded. Returns (B,) one logit per window.
        """
        embeddings = self.encode_frames(features)  # (B, window_size, embed_dim)
        if lengths is not None:
            packed = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(embeddings)
        # Same final-hidden-state pooling as WindowedSpeakingDetectorRNN — see
        # that model's forward() comment for why h_n[-2]/h_n[-1] are the
        # right slice.
        final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, hidden_size * 2)
        return self.head(final_hidden).squeeze(-1)


if __name__ == "__main__":
    from dataset import LOGICAL_SPLITS
    from image_dataset import build_image_cache, get_image_windowed_dataloader

    build_image_cache()
    for split in LOGICAL_SPLITS:
        loader = get_image_windowed_dataloader(split)
        batch = next(iter(loader))
        model = ImageWindowedSpeakingDetectorRNN()
        logits = model(batch["features"], batch["lengths"])
        assert logits.shape == batch["labels"].shape, (logits.shape, batch["labels"].shape)
        print(f"[{split}] logits shape: {tuple(logits.shape)} (matches labels shape)")
