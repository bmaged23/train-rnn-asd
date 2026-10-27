"""Per-source routing uses batch provenance, never globally guessed track IDs."""
import unittest
import torch
from scripts.modeling.evaluate.windowed_all_combined_per_source import collect_predictions_with_source


class FixedLogitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, features, lengths):
        self.calls += 1
        return features[:, 0, 0]


class SourceAttributionTests(unittest.TestCase):
    def batch(self):
        return {'features': torch.tensor([[[-4.]], [[4.]]]), 'lengths': torch.tensor([1, 1]),
                'labels': torch.tensor([0, 1]), 'person_id': ['same-track-id', 'same-track-id'],
                'source_label': ['unitalk', 'ava']}

    def test_same_track_id_in_different_corpora_retains_correct_source(self):
        model = FixedLogitModel()
        result = collect_predictions_with_source(model, [self.batch()], torch.device('cpu'))
        self.assertEqual(result['unitalk'][0].tolist(), [0])
        self.assertEqual(result['ava'][0].tolist(), [1])
        self.assertEqual(result['combined'][0].tolist(), [0, 1])
        self.assertEqual(model.calls, 1)

    def test_missing_or_mismatched_provenance_fails_instead_of_dropping_rows(self):
        for kind in ['missing', 'short', 'unknown']:
            batch = self.batch()
            if kind == 'missing':del batch['source_label']
            elif kind == 'short':batch['source_label'] = ['unitalk']
            else:batch['source_label'] = ['unexpected', 'ava']
            with self.subTest(kind=kind), self.assertRaises((ValueError, KeyError)):
                collect_predictions_with_source(FixedLogitModel(), [batch], torch.device('cpu'))

    def test_empty_evaluation_is_not_silently_successful(self):
        with self.assertRaisesRegex(ValueError, 'No windows'):
            collect_predictions_with_source(FixedLogitModel(), [], torch.device('cpu'))
