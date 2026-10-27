"""Real HTTP/container checks using synthetic weights, not model-quality evidence."""
import argparse
import hashlib
import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch

from src.windowed_model import WindowedSpeakingDetectorRNN


def check_container(image, fixtures=None):
    command = ["docker", "run", "--rm", "-d", "--read-only", "--tmpfs", "/tmp",
               "-p", "127.0.0.1::8000", "-e", "ASD_API_TOKEN=container-test-token"]
    if fixtures:
        command += ["-v", f"{fixtures}:/fixtures:ro", "-e", "ASD_CHECKPOINT=/fixtures/model.pt",
                    "-e", "ASD_MANIFEST=/fixtures/manifest.json"]
    container = subprocess.check_output(command + [image], text=True, timeout=30).strip()
    try:
        address = subprocess.check_output(["docker", "port", container, "8000/tcp"], text=True, timeout=10).strip()
        base = "http://" + address

        def request(path, body=None, token=True):
            headers = {"Authorization": "Bearer container-test-token"} if token else {}
            if body is not None:
                headers["Content-Type"] = "application/json"
            req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                         headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    return response.status, json.loads(response.read())
            except urllib.error.HTTPError as error:
                return error.code, json.loads(error.read())

        deadline = time.monotonic() + 40
        while True:
            try:
                if request("/health/live")[0] == 200:
                    break
            except (OSError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise AssertionError("API did not become live within 40 seconds")
            time.sleep(.25)
        ready, _ = request("/health/ready")
        if not fixtures:
            if ready != 503:
                raise AssertionError("Container with no checkpoint must not be ready")
            return
        if ready != 200:
            raise AssertionError("Valid fixture checkpoint did not become ready")
        body = json.loads((fixtures / "input.json").read_text())
        status, prediction = request("/v1/predict", body)
        if status != 200:
            raise AssertionError(f"Prediction failed with status {status}")
        expected = json.loads((fixtures / "expected.json").read_text())
        torch.testing.assert_close(torch.tensor(prediction["probabilities"]), torch.tensor(expected))
        if request("/v1/predict", body, token=False)[0] != 401:
            raise AssertionError("Unauthenticated predictions must fail")
        if request("/v1/predict", body | {"lengths": [0]})[0] != 422:
            raise AssertionError("Invalid lengths must fail")
        if request("/metrics")[1]["completed"] != 1:
            raise AssertionError("Prediction counters mismatch")
        if "/v1/predict" not in request("/openapi.json")[1]["paths"]:
            raise AssertionError("OpenAPI prediction schema missing")
    finally:
        subprocess.run(["docker", "stop", "--time", "3", container], check=True, capture_output=True, timeout=15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="asd-window-toolkit:local")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="asd-http-contract-") as directory:
        root = Path(directory)
        root.chmod(0o755)
        torch.manual_seed(23)
        model = WindowedSpeakingDetectorRNN(2, 3, 1, 0).eval()
        checkpoint = root / "model.pt"
        torch.save(model.state_dict(), checkpoint)
        metadata = dict(schema_version=1, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        feature_names=["x", "y"], window_size=4, hidden_size=3, num_layers=1,
                        dropout=0, threshold=.5, preprocessing="Synthetic HTTP contract only")
        (root / "manifest.json").write_text(json.dumps(metadata))
        features, lengths = torch.ones(1, 4, 2) * .25, torch.tensor([3])
        (root / "input.json").write_text(json.dumps(dict(feature_names=["x", "y"],
                                                        features=features.tolist(), lengths=lengths.tolist())))
        with torch.inference_mode():
            (root / "expected.json").write_text(json.dumps(model(features, lengths).sigmoid().tolist()))
        check_container(args.image, root)
        check_container(args.image)
    print("PASS: container HTTP prediction parity, auth, validation, metrics, OpenAPI and missing-model readiness.")


if __name__ == "__main__":
    main()
