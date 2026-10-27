"""Exercise actual CSV/PyArrow/window construction with small synthetic tracks."""
import csv
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src import dataset as frames
from src import windowed_dataset as windows


class TrainingCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv_path = self.root / 'train.csv'
        self.cache = self.root / 'cache'
        self.enterContext(patch.dict(frames._SPLIT_CSV_PATH, {'train': self.csv_path}))
        self.enterContext(patch.object(windows, 'WINDOWED_CACHE_DIR', self.cache))
        self.enterContext(patch.object(windows, 'AUGMENT_LANDMARKS', False))

    def write_track(self, label):
        with self.csv_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=frames._METADATA_COLUMNS + frames.FEATURE_COLUMNS)
            writer.writeheader()
            for index in range(12):
                row = {name: .1 for name in frames.FEATURE_COLUMNS}
                row.update(person_id='track1', image_name=f'{index / 25:.2f}.jpg', detected=True, target_index=label)
                writer.writerow(row)

    def load(self):
        return windows.WindowedLandmarkSequenceDataset('train', window_size=4, window_stride=4,
                    min_detected_ratio=0, pad_short_tracks=False)

    def test_changed_labels_invalidate_cache_even_if_size_and_timestamp_match(self):
        self.write_track(0)
        original_stat = self.csv_path.stat()
        first = self.load()
        self.assertEqual(first.labels, [0, 0, 0])
        self.write_track(1)
        self.assertEqual(self.csv_path.stat().st_size, original_stat.st_size)
        os.utime(self.csv_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        second = self.load()
        self.assertEqual(second.labels, [1, 1, 1])
        self.assertEqual(len(list(self.cache.glob('*.pkl'))), 2)

    def test_interrupted_cache_write_does_not_publish_partial_or_leave_temp_file(self):
        self.write_track(0)
        with patch.object(windows.pickle, 'dump', side_effect=OSError('controlled disk failure')):
            with self.assertRaisesRegex(OSError, 'controlled disk failure'):
                self.load()
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_unchanged_source_reuses_complete_cache(self):
        self.write_track(0)
        first = self.load()
        with patch.object(windows, 'LandmarkSequenceDataset', side_effect=AssertionError('CSV tensors should not rebuild')):
            second = self.load()
        self.assertEqual(first.labels, second.labels)
        self.assertEqual(len(list(self.cache.glob('*.pkl'))), 1)


if __name__ == '__main__':
    unittest.main()
