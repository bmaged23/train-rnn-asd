"""Real CPU Torch operators and state loading; random weights are not accuracy evidence."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from src.model import SpeakingDetectorRNN
from src.windowed_model import WindowedSpeakingDetectorRNN
from src.dataset import compute_mouth_geometry_features
from config import MOUTH_HEIGHT_PAIRS, MOUTH_WIDTH_PAIRS
from src import dataset


class RNNContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.features = torch.randn(2, 7, 8)
        self.lengths = torch.tensor([3, 7])  # deliberately not length-sorted

    def test_frame_logits_have_correct_shape_and_ignore_padded_tail(self):
        model = SpeakingDetectorRNN(input_size=8, hidden_size=4, num_layers=2, dropout=0).eval()
        changed = self.features.clone()
        changed[0, 3:] = 1000
        with torch.inference_mode():
            original = model(self.features, self.lengths)
            modified = model(changed, self.lengths)
        self.assertEqual(tuple(original.shape), (2, 7))
        self.assertTrue(torch.isfinite(original).all())
        torch.testing.assert_close(original[0, :3], modified[0, :3], rtol=1e-5, atol=1e-6)
        # Invalid/padded frame logits still need the caller's mask when scoring.

    def test_window_order_and_padding_match_individual_true_length_inputs(self):
        model = WindowedSpeakingDetectorRNN(input_size=8, hidden_size=4, num_layers=2, dropout=0).eval()
        changed = self.features.clone()
        changed[0, 3:] = -1000
        with torch.inference_mode():
            batch = model(self.features, self.lengths)
            modified = model(changed, self.lengths)
            individual = torch.cat([model(self.features[i:i+1, :length]) for i, length in enumerate(self.lengths)])
        self.assertEqual(tuple(batch.shape), (2,))
        torch.testing.assert_close(batch, modified, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(batch, individual, rtol=1e-5, atol=1e-6)

    def test_state_dict_roundtrip_preserves_forward_outputs(self):
        model = WindowedSpeakingDetectorRNN(input_size=8, hidden_size=4, num_layers=2, dropout=0).eval()
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'contract.pt'
            torch.save(model.state_dict(), checkpoint)
            restored = WindowedSpeakingDetectorRNN(input_size=8, hidden_size=4, num_layers=2, dropout=0).eval()
            restored.load_state_dict(torch.load(checkpoint, weights_only=True))
            with torch.inference_mode():
                torch.testing.assert_close(model(self.features, self.lengths), restored(self.features, self.lengths), rtol=0, atol=0)

    def test_mouth_pair_distances_match_explicit_landmark_indices(self):
        indices = list(dataset._LANDMARK_INDICES)
        points = np.zeros((2, len(indices), 3), dtype=np.float32)
        points[0, :, 0] = indices
        expected = np.array([abs(a-b) for a, b in MOUTH_HEIGHT_PAIRS + MOUTH_WIDTH_PAIRS])
        actual = compute_mouth_geometry_features(points)
        np.testing.assert_allclose(actual[0], expected)
        np.testing.assert_array_equal(actual[1], np.zeros(len(expected)))


if __name__ == '__main__':
    unittest.main()
