"""Explicit serving configuration; one loaded model per process."""
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    checkpoint: Path | None = None
    manifest: Path | None = None
    device: str = "cpu"
    max_body_bytes: int = 4 * 1024 * 1024
    token: str = field(default="", repr=False)

    def __post_init__(self):
        if self.device not in ("cpu", "cuda"):
            raise ValueError("ASD_DEVICE must be cpu or cuda")
        if self.max_body_bytes < 1:
            raise ValueError("ASD_MAX_BODY_BYTES must be positive")

    @classmethod
    def from_env(cls):
        def path(name):
            return Path(os.environ[name]).expanduser() if os.getenv(name) else None
        return cls(checkpoint=path("ASD_CHECKPOINT"), manifest=path("ASD_MANIFEST"),
                   device=os.getenv("ASD_DEVICE", "cpu"),
                   max_body_bytes=int(os.getenv("ASD_MAX_BODY_BYTES", str(4 * 1024 * 1024))),
                   token=os.getenv("ASD_API_TOKEN", ""))
