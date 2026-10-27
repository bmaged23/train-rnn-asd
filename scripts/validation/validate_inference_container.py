"""Exercise the packaged CLI with synthetic weights; no accuracy evaluation."""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import torch

from src.windowed_model import WindowedSpeakingDetectorRNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="asd-window-toolkit:local")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="asd-container-contract-") as directory:
        root = Path(directory)
        root.chmod(0o755)
        output = root / "output"
        output.mkdir(mode=0o777)
        output.chmod(0o777)
        torch.manual_seed(17)
        model = WindowedSpeakingDetectorRNN(2, 3, 1, 0).eval()
        checkpoint = root / "model.pt"
        torch.save(model.state_dict(), checkpoint)
        metadata = dict(schema_version=1, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        feature_names=["x", "y"], window_size=4, hidden_size=3,
                        num_layers=1, dropout=0, threshold=0.5,
                        preprocessing="Synthetic container contract only; no accuracy claim")
        (root / "manifest.json").write_text(json.dumps(metadata))
        features = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2) / 16
        lengths = torch.tensor([4, 2])
        (root / "input.json").write_text(json.dumps(dict(
            feature_names=["x", "y"], features=features.tolist(), lengths=lengths.tolist())))
        command = ["docker", "run", "--rm", "--network", "none",
                   "--read-only", "--tmpfs", "/tmp", "-v", f"{root}:/fixtures:ro",
                   "-v", f"{output}:/output", args.image, "-m", "scripts.inference.predict_windows",
                   "--checkpoint", "/fixtures/model.pt", "--manifest", "/fixtures/manifest.json",
                   "--input", "/fixtures/input.json", "--output", "/output/predictions.json"]
        subprocess.run(command, check=True, timeout=90)
        actual = json.loads((output / "predictions.json").read_text())
        with torch.inference_mode():
            expected = model(features, lengths).sigmoid()
        torch.testing.assert_close(torch.tensor(actual["probabilities"]), expected)
        if actual["speaking"] != (expected >= .5).tolist():
            raise AssertionError("Threshold labels differ")
        checkpoint.unlink()
        missing = subprocess.run(command, capture_output=True, text=True, timeout=90)
        if missing.returncode == 0 or "FileNotFoundError" not in missing.stderr:
            raise AssertionError("Missing checkpoint must fail explicitly")
    print("PASS: container predictions match native synthetic model; absent weights fail. No accuracy evaluated.")


if __name__ == "__main__":
    main()
