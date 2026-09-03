"""IResNet backbone architecture, vendored from the official AdaFace
repository (https://github.com/mk-minchul/AdaFace, MIT License, Copyright
(c) 2022 Minchul Kim) so this project can load its own downloaded
adaface_ir101_webface4m.ckpt checkpoint without adding a package dependency
on the whole AdaFace training/eval codebase (which pulls in
face_alignment/MTCNN, dataset tooling, etc. this project doesn't need).
Architecture code (BasicBlockIR/get_blocks/Backbone) is unmodified from
net.py other than trimming to the 'ir' (non-SE) mode this project uses and
adding load_adaface_backbone()/adaface_preprocess() as this project's own
integration glue — see src/image_model.py for how the frozen/trainable
split and BGR/112x112/mean-std=0.5 preprocessing get applied, and this
module's own docstring history for the full layer-by-layer breakdown
(stem -> 49 residual blocks across 4 stages -> embedding head ->
L2-normalize -> 512-d embedding) discussed with the user before this file
was written.

Real per-stage block counts (verified 2026-08-24 against the actual
checkpoint's state_dict, `model.body.0` through `model.body.48` inclusive
= 49 blocks total):
    num_layers=100 -> stage1: 3 blocks (64ch), stage2: 13 blocks (128ch),
                       stage3: 30 blocks (256ch), stage4: 3 blocks (512ch)
"""
from __future__ import annotations

import sys
from collections import namedtuple
from pathlib import Path

import torch
from torch import nn
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Linear,
    MaxPool2d,
    Module,
    PReLU,
    Sequential,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import IMAGE_ADAFACE_INPUT_SIZE, IMAGE_NORMALIZE_MEAN, IMAGE_NORMALIZE_STD


class Flatten(Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class BasicBlockIR(Module):
    """Residual block for IResNet — BN -> Conv3x3 -> BN -> PReLU -> Conv3x3
    -> BN, with a MaxPool or Conv1x1+BN shortcut, matching AdaFace's net.py
    exactly (needed for the checkpoint's state_dict keys to line up).
    """

    def __init__(self, in_channel: int, depth: int, stride: int) -> None:
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth)
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_layer(x) + self.shortcut_layer(x)


class _Block(namedtuple("_Block", ["in_channel", "depth", "stride"])):
    pass


def _get_block(in_channel: int, depth: int, num_units: int, stride: int = 2) -> list[_Block]:
    return [_Block(in_channel, depth, stride)] + [_Block(depth, depth, 1) for _ in range(num_units - 1)]


def get_blocks(num_layers: int) -> list[list[_Block]]:
    if num_layers == 50:
        return [
            _get_block(64, 64, 3), _get_block(64, 128, 4),
            _get_block(128, 256, 14), _get_block(256, 512, 3),
        ]
    if num_layers == 100:
        return [
            _get_block(64, 64, 3), _get_block(64, 128, 13),
            _get_block(128, 256, 30), _get_block(256, 512, 3),
        ]
    raise ValueError(f"unsupported num_layers: {num_layers!r} (this project only vendors 50/100)")


class IResNetBackbone(Module):
    """Trimmed to mode='ir' (BasicBlockIR, no squeeze-excite) and
    input_size=112 — the only combination this project's downloaded
    checkpoint (adaface_ir101_webface4m.ckpt, num_layers=100) needs.
    `self.body` is a flat Sequential of every block across all 4 stages
    (stage boundaries only exist conceptually, via stage_block_counts
    below) — matches the checkpoint's `model.body.0`..`model.body.48` keys.
    """

    def __init__(self, num_layers: int = 100) -> None:
        super().__init__()
        assert IMAGE_ADAFACE_INPUT_SIZE == 112, "only 112x112 input verified against this checkpoint"
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64))
        blocks = get_blocks(num_layers)
        self.stage_block_counts = [len(stage) for stage in blocks]  # e.g. [3, 13, 30, 3] for num_layers=100
        output_channel = 512
        self.output_layer = Sequential(
            BatchNorm2d(output_channel),
            Dropout(0.4),
            Flatten(),
            Linear(output_channel * 7 * 7, 512),
            BatchNorm1d(512, affine=False),
        )
        modules = [
            BasicBlockIR(b.in_channel, b.depth, b.stride) for stage in blocks for b in stage
        ]
        self.body = Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        return torch.div(x, norm)  # L2-normalized embedding — same convention the checkpoint was trained under


def load_adaface_backbone(checkpoint_path: Path, num_layers: int = 100) -> IResNetBackbone:
    """Builds IResNetBackbone(num_layers) and loads real weights from a
    checkpoint downloaded from the official AdaFace repo's pretrained-models
    table (gdrive links in its README) — e.g. models/adaface_ir101_webface4m.ckpt.
    The checkpoint is a PyTorch Lightning save: {"state_dict": {"model.input_layer...":
    ..., "model.body...": ..., "model.output_layer...": ..., "head...": ...}}.
    `head.*` is AdaFace's training-time margin-loss classifier (one weight
    per training identity) — deliberately dropped, not needed for embedding
    extraction and not part of this architecture at all. `model.` prefix is
    stripped so the remaining keys line up with IResNetBackbone's own
    state_dict exactly (verified 2026-08-24: strict=True loads cleanly).
    Loads onto CPU regardless of target device — the caller's own
    `ImageWindowedSpeakingDetectorRNN().to(device)` (existing project
    pattern) moves this backbone to GPU along with everything else, since
    it becomes a real registered submodule, not something needing its own
    separate device handling.
    """
    backbone = IResNetBackbone(num_layers)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {
        k[len("model."):]: v for k, v in raw["state_dict"].items() if k.startswith("model.")
    }
    backbone.load_state_dict(state_dict, strict=True)
    return backbone


def adaface_preprocess(images: torch.Tensor) -> torch.Tensor:
    """images: (N, 3, H, W), already ImageNet-normalized RGB the way the
    rest of this project's pipeline produces them (src/image_dataset.py's
    IMAGE_NORMALIZE_MEAN/STD) — converts to what AdaFace's IResNet actually
    expects: BGR channel order, IMAGE_ADAFACE_INPUT_SIZE x same square size,
    normalized with mean=0.5/std=0.5 (NOT ImageNet stats — see AdaFace's
    own inference.py to_input()). Three real, verified-against-source
    differences from this project's normal preprocessing, all applied here:
    undo ImageNet normalization -> resize -> RGB->BGR channel flip ->
    renormalize to mean/std=0.5. Stays on `images`' own device throughout
    (mean/std tensors built with device=images.device) so this works
    identically on CPU or GPU without any special-casing by the caller.
    """
    mean = torch.tensor(IMAGE_NORMALIZE_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_NORMALIZE_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    raw_0_1 = (images * std + mean).clamp(0, 1)
    resized = nn.functional.interpolate(
        raw_0_1, size=(IMAGE_ADAFACE_INPUT_SIZE, IMAGE_ADAFACE_INPUT_SIZE), mode="bilinear", align_corners=False
    )
    bgr = resized[:, [2, 1, 0], :, :]
    return (bgr - 0.5) / 0.5
