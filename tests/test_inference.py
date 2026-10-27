"""Synthetic weights test serving contracts, never speaking-detection accuracy."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.inference import WindowPredictor
from src.windowed_model import WindowedSpeakingDetectorRNN


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.checkpoint, self.manifest = root / "test.pt", root / "manifest.json"
        torch.save(WindowedSpeakingDetectorRNN(2, 3, 1, 0).state_dict(), self.checkpoint)
        metadata = dict(schema_version=1, checkpoint_sha256=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
                        feature_names=["x", "y"], window_size=4, hidden_size=3,
                        num_layers=1, dropout=0, threshold=0.5,
                        preprocessing="Synthetic test features; not a trained model")
        self.manifest.write_text(json.dumps(metadata))
        self.predictor = WindowPredictor(self.checkpoint, self.manifest)
        self.request = dict(feature_names=["x", "y"], features=[[[0., 0.]] * 4], lengths=[3])

    def test_matches_direct_model(self):
        result = self.predictor.predict(self.request)
        with torch.inference_mode():
            expected = self.predictor.model(torch.tensor(self.request["features"]), torch.tensor([3])).sigmoid().tolist()
        self.assertEqual(result["probabilities"], expected)
        json.dumps(result, allow_nan=False)

    def test_rejects_changed_checkpoint(self):
        with self.checkpoint.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            WindowPredictor(self.checkpoint, self.manifest)

    def test_rejects_invalid_requests(self):
        for update in ({"feature_names": ["y", "x"]}, {"lengths": [0]},
                       {"lengths": [2.5]}, {"features": [[[float("nan"), 0.]] * 4]},
                       {"features": [[[0., 0.]] * 3]}, {"features": []}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.predictor.predict(self.request | update)

    def test_missing_weights_never_fall_back(self):
        self.checkpoint.unlink()
        with self.assertRaises(FileNotFoundError):
            WindowPredictor(self.checkpoint, self.manifest)


if __name__ == "__main__":
    unittest.main()
