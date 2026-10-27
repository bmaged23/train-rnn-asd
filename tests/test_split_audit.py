"""Synthetic metadata exercises split checks; no model/data accuracy claims."""
import csv
from pathlib import Path
import tempfile
import unittest
from scripts.validation.audit_split_isolation import audit_sources


class SplitAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def csv(self, name, video, person=None, include_video=True):
        path = self.root / f'{name}.csv'
        fields = ['person_id', 'target_index', 'detected'] + (['video_id'] if include_video else [])
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for label in [0, 1]:
                row = dict(person_id=person or name, target_index=label, detected=True)
                if include_video:
                    row['video_id'] = video
                writer.writerow(row)
        return path

    def test_disjoint_tracks_and_recordings_pass_with_fingerprints(self):
        paths = {s: self.csv(s, s) for s in ('train', 'val', 'test')}
        report = audit_sources({'source': paths}, hash_inputs=True)
        self.assertEqual(report['within_source_isolation'], 'passed')
        self.assertEqual(report['sources']['source']['splits']['test']['class_counts'], {'0': 1, '1': 1})
        self.assertEqual(len(report['sources']['source']['splits']['test']['sha256']), 64)

    def test_different_tracks_from_same_recording_require_review(self):
        paths = {s: self.csv(s, 'shared-movie') for s in ('train', 'val', 'test')}
        report = audit_sources({'source': paths})
        self.assertEqual(report['within_source_isolation'], 'review_required')
        overlap = report['sources']['source']['overlaps']['val/test']
        self.assertEqual(overlap['shared_tracks'], 0)
        self.assertEqual(overlap['shared_recordings'], 1)

    def test_same_track_in_different_splits_requires_review(self):
        paths = {s: self.csv(s, s, person='repeated-person-track') for s in ('train', 'val', 'test')}
        report = audit_sources({'source': paths})
        self.assertEqual(report['within_source_isolation'], 'review_required')
        self.assertEqual(report['sources']['source']['overlaps']['train/test']['shared_tracks'], 1)

    def test_missing_recording_ids_do_not_silently_pass(self):
        paths = {s: self.csv(s, '', include_video=False) for s in ('train', 'val', 'test')}
        self.assertEqual(audit_sources({'source': paths})['within_source_isolation'], 'review_required')

    def test_csv_quoted_track_ids_are_parsed_correctly(self):
        paths = {s: self.csv(s, s, person=f'{s},person') for s in ('train', 'val', 'test')}
        self.assertEqual(audit_sources({'source': paths})['within_source_isolation'], 'passed')


if __name__ == '__main__':
    unittest.main()
