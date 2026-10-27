"""Checkpoint-backed inference on already prepared landmark feature windows.

Never creates a fallback model when weights or preprocessing metadata are absent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from src.windowed_model import WindowedSpeakingDetectorRNN


class WindowPredictor:
    def __init__(self, checkpoint: Path, manifest: Path, device: str = "cpu"):
        self.metadata = json.loads(manifest.read_text())
        required = {"schema_version", "checkpoint_sha256", "feature_names", "window_size",
                    "hidden_size", "num_layers", "dropout", "threshold", "preprocessing"}
        if set(self.metadata) != required or self.metadata["schema_version"] != 1:
            raise ValueError("Unsupported or incomplete checkpoint manifest")
        m = self.metadata
        for key in ("window_size", "hidden_size", "num_layers"):
            if type(m[key]) is not int or m[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        names = m["feature_names"]
        if (not isinstance(names, list) or not names or
                any(not isinstance(name, str) or not name.strip() for name in names) or
                len(names) != len(set(names))):
            raise ValueError("feature_names must contain unique, ordered names")
        for key in ("threshold", "dropout"):
            value = m[key]
            if type(value) not in (int, float) or not 0 <= value < 1:
                raise ValueError(f"{key} must be finite and in [0, 1)")
        if not isinstance(m["preprocessing"], str) or not m["preprocessing"].strip():
            raise ValueError("Document the exact training preprocessing")
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; no implicit CPU fallback")
        with checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != m["checkpoint_sha256"]:
            raise ValueError("Checkpoint SHA256 does not match manifest")
        self.device = torch.device(device)
        self.model = WindowedSpeakingDetectorRNN(
            input_size=len(names), hidden_size=m["hidden_size"],
            num_layers=m["num_layers"], dropout=m["dropout"],
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    def predict(self, request: dict) -> dict:
        if set(request) != {"feature_names", "features", "lengths"}:
            raise ValueError("Expected feature_names, features and lengths")
        if request["feature_names"] != self.metadata["feature_names"]:
            raise ValueError("Feature names/order do not match the training manifest")
        features = torch.as_tensor(request["features"], dtype=torch.float32)
        if features.ndim != 3 or features.shape[0] not in range(1, 65):
            raise ValueError("features must have shape [batch (1..64), time, features]")
        if tuple(features.shape[1:]) != (self.metadata["window_size"], len(self.metadata["feature_names"])):
            raise ValueError("Window shape does not match manifest")
        if not torch.isfinite(features).all():
            raise ValueError("All features must be finite")
        lengths = request["lengths"]
        if (not isinstance(lengths, list) or len(lengths) != features.shape[0] or
                any(type(n) is not int or not 1 <= n <= features.shape[1] for n in lengths)):
            raise ValueError("lengths must contain one valid integer per window")
        with torch.inference_mode():
            probabilities = self.model(features.to(self.device), torch.tensor(lengths)).sigmoid().cpu().tolist()
        return {"probabilities": probabilities,
                "speaking": [p >= self.metadata["threshold"] for p in probabilities],
                "threshold": self.metadata["threshold"],
                "checkpoint_sha256": self.metadata["checkpoint_sha256"]}
