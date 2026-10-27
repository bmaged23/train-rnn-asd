"""Metric contracts use synthetic arrays, not trained-model accuracy evidence."""
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.metrics import compute_metrics, format_report, plot_confusion_matrix, plot_pr_curve, plot_roc_curve


class ClassificationMetricTests(unittest.TestCase):
    def test_hand_checked_confusion_and_calibration_metrics(self):
        result = compute_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 0]), np.array([.2, .8, .6, .4]))
        self.assertEqual(result['confusion_matrix'], [[1, 1], [1, 1]])
        for key in ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'balanced_accuracy', 'roc_auc']:
            self.assertAlmostEqual(result[key], .5, msg=key)
        self.assertEqual(result['mcc'], 0.)
        self.assertAlmostEqual(result['brier_score'], .3)
        self.assertAlmostEqual(result['pr_auc'], (1/2 + 2/3) / 2)
        self.assertEqual(result['class_support'], [2, 2])

    def test_perfect_predictions_retain_original_scores(self):
        result = compute_metrics(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]))
        for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'mcc']:
            self.assertEqual(result[key], 1.)

    def test_one_class_reports_are_strict_json_and_plots_are_explicit(self):
        for labels, probabilities in [([0, 0, 0], [.1, .2, .3]), ([1, 1, 1], [.7, .8, .9])]:
            with self.subTest(labels=labels), warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                result = compute_metrics(np.array(labels), np.array(probabilities) > .5, np.array(probabilities))
                self.assertIsNone(result['roc_auc'])
                self.assertIsNone(result['balanced_accuracy'])
                self.assertIn('undefined', format_report(result))
                public = {k: v for k, v in result.items() if not k.startswith('_')}
                json.dumps(public, allow_nan=False)
                with tempfile.TemporaryDirectory() as folder:
                    for index, plot in enumerate([plot_confusion_matrix, plot_roc_curve, plot_pr_curve]):
                        path = Path(folder) / f'{index}.png'
                        figure = plot(result, save_path=path)
                        self.assertGreater(path.stat().st_size, 100)
                        plt.close(figure)

    def test_invalid_shapes_labels_probabilities_and_empty_inputs_rejected(self):
        for truth, prediction, probability in [([], [], []), ([0], [0, 1], [.2]),
                ([[0]], [0], [.2]), ([2], [0], [.2]), ([0], [2], [.2]),
                ([0], [0], [float('nan')]), ([0], [0], [float('inf')]),
                ([0], [0], [-.1]), ([1], [1], [1.1])]:
            with self.subTest(truth=truth, probability=probability), self.assertRaises(ValueError):
                compute_metrics(np.array(truth), np.array(prediction), np.array(probability))


if __name__ == '__main__':
    unittest.main()
