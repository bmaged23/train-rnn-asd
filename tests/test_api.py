"""API behavior with synthetic checkpoints, not a trained-model evaluation."""
import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.settings import Settings
from src.windowed_model import WindowedSpeakingDetectorRNN


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.checkpoint, self.manifest = root / "model.pt", root / "manifest.json"
        torch.save(WindowedSpeakingDetectorRNN(2, 3, 1, 0).state_dict(), self.checkpoint)
        self.metadata = dict(schema_version=1, checkpoint_sha256=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
                             feature_names=["x", "y"], window_size=4, hidden_size=3, num_layers=1,
                             dropout=0, threshold=.5, preprocessing="Synthetic test only")
        self.manifest.write_text(json.dumps(self.metadata))
        self.settings = Settings(self.checkpoint, self.manifest)
        self.request = dict(feature_names=["x", "y"], features=[[[.1, .2]] * 4], lengths=[3])

    def test_health_metadata_predictions_and_metrics(self):
        app = create_app(self.settings)
        with TestClient(app) as client:
            self.assertEqual(client.get("/health/live").status_code, 200)
            self.assertEqual(client.get("/health/ready").json()["status"], "ready")
            self.assertEqual(client.get("/v1/model").json(), self.metadata)
            result = client.post("/v1/predict", json=self.request)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json(), app.state.predictor.predict(self.request))
            self.assertEqual(client.get("/metrics").json()["completed"], 1)
            self.assertIn("/v1/predict", client.get("/openapi.json").json()["paths"])
        self.assertIsNone(app.state.predictor)

    def test_no_checkpoint_or_invalid_checkpoint_never_ready(self):
        for settings in (Settings(), Settings(self.checkpoint.parent / "absent.pt", self.manifest)):
            with self.subTest(settings=settings), TestClient(create_app(settings)) as client:
                self.assertEqual(client.get("/health/live").status_code, 200)
                self.assertEqual(client.get("/health/ready").status_code, 503)
                self.assertEqual(client.post("/v1/predict", json=self.request).status_code, 503)

    def test_bad_hash_never_ready(self):
        self.manifest.write_text(json.dumps(self.metadata | {"checkpoint_sha256": "wrong"}))
        with TestClient(create_app(self.settings)) as client:
            self.assertEqual(client.get("/health/ready").status_code, 503)

    def test_invalid_inputs_are_422_without_echoing_payloads(self):
        with TestClient(create_app(self.settings)) as client:
            for update in ({"lengths": [1.5]}, {"lengths": [0]}, {"features": []},
                           {"feature_names": ["y", "x"]}, {"unknown": "field"},
                           {"features": [[["not-number", .2]] * 4]}, {"features": [[[.1, .2]] * 3]}):
                with self.subTest(update=update):
                    response = client.post("/v1/predict", json=self.request | update)
                    self.assertEqual(response.status_code, 422)
                    self.assertNotIn('"input":', response.text)

    def test_body_limit_applies_even_without_content_length(self):
        settings = Settings(self.checkpoint, self.manifest, max_body_bytes=32)
        with TestClient(create_app(settings)) as client:
            self.assertEqual(client.post("/v1/predict", content=iter([b"x" * 20, b"x" * 20])).status_code, 413)

    def test_optional_auth_protects_data_endpoints(self):
        with TestClient(create_app(Settings(self.checkpoint, self.manifest, token="test-token"))) as client:
            for endpoint in ("/v1/model", "/metrics"):
                self.assertEqual(client.get(endpoint).status_code, 401)
                self.assertEqual(client.get(endpoint, headers={"Authorization": "Bearer test-token"}).status_code, 200)
            self.assertEqual(client.post("/v1/predict", json=self.request).status_code, 401)
            self.assertEqual(client.get("/health/ready").status_code, 200)

    def test_busy_returns_retryable_503(self):
        app = create_app(self.settings)
        entered, release = threading.Event(), threading.Event()
        with TestClient(app) as client:
            original = app.state.predictor.predict

            def slow(request):
                entered.set()
                if not release.wait(timeout=5):
                    raise RuntimeError("Test synchronization timed out")
                return original(request)

            responses = []
            with patch.object(app.state.predictor, "predict", side_effect=slow):
                thread = threading.Thread(target=lambda: responses.append(client.post("/v1/predict", json=self.request)))
                thread.start()
                try:
                    self.assertTrue(entered.wait(timeout=5))
                    busy = client.post("/v1/predict", json=self.request)
                    self.assertEqual(busy.status_code, 503)
                    self.assertEqual(busy.headers["retry-after"], "1")
                finally:
                    release.set()
                    thread.join(timeout=5)
                self.assertEqual(responses[0].status_code, 200)

    def test_failed_execution_releases_inference_slot(self):
        app = create_app(self.settings)
        with TestClient(app) as client:
            with patch.object(app.state.predictor, "predict", side_effect=RuntimeError("private details")):
                failure = client.post("/v1/predict", json=self.request)
                self.assertEqual(failure.status_code, 500)
                self.assertNotIn("private details", failure.text)
            self.assertEqual(client.post("/v1/predict", json=self.request).status_code, 200)
