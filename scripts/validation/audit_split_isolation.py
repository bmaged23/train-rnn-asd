"""Stream processed CSV metadata and audit exact track/recording split overlap.

This never loads landmark arrays or changes data/splits. Source namespaces are
kept separate: equal generic IDs across datasets do not prove equal recordings.
"""
import argparse
from collections import Counter
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path


def scan_split(path, hash_inputs=False):
    path = Path(path)
    tracks, videos = set(), set()
    labels = Counter()
    rows = invalid_rows = missing_video_rows = detected_rows = 0
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'person_id', 'target_index', 'detected'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: requires columns {sorted(required)}')
        has_video_column = 'video_id' in reader.fieldnames
        for row in reader:
            rows += 1
            person = (row.get('person_id') or '').strip()
            label = (row.get('target_index') or '').strip()
            detected = (row.get('detected') or '').strip().lower()
            if None in row or not person or label not in {'0', '1', '0.0', '1.0'} or detected not in {'true', 'false', '0', '1', '0.0', '1.0'}:
                invalid_rows += 1
                continue
            tracks.add(person)
            labels[int(float(label))] += 1
            detected_rows += detected in {'true', '1', '1.0'}
            video = (row.get('video_id') or '').strip()
            if video and video.lower() not in {'nan', 'none', 'null'}:
                videos.add(video)
            else:
                missing_video_rows += 1
    result = {'path': str(path), 'bytes': path.stat().st_size, 'rows': rows,
              'invalid_rows': invalid_rows, 'tracks': len(tracks), 'recordings': len(videos),
              'class_counts': {str(key): labels[key] for key in (0, 1)},
              'detected_rows': detected_rows, 'has_video_column': has_video_column,
              'missing_video_rows': missing_video_rows}
    if hash_inputs:
        with path.open('rb') as stream:
            result['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return result, tracks, videos


def audit_sources(sources, hash_inputs=False):
    report = {'version': 1, 'sources': {}, 'within_source_isolation': 'passed',
              'scope': 'Exact person_id track and video_id recording identifiers within each source. Does not establish subject independence, canonical recording IDs or duplicate-image absence across different datasets.'}
    if not sources:
        raise ValueError('At least one source is required')
    for name, paths in sources.items():
        if set(paths) != {'train', 'val', 'test'}:
            raise ValueError(f'{name}: provide train, val and test CSVs')
        summaries, track_sets, video_sets, issues = {}, {}, {}, []
        for split, path in paths.items():
            summaries[split], track_sets[split], video_sets[split] = scan_split(path, hash_inputs)
            summary = summaries[split]
            if not summary['rows'] or summary['invalid_rows']:
                issues.append(f'{split}: empty data or invalid metadata rows')
            if not summary['has_video_column'] or summary['missing_video_rows']:
                issues.append(f'{split}: recording isolation cannot be established for every row')
        overlaps = {}
        for left, right in combinations(('train', 'val', 'test'), 2):
            tracks = track_sets[left] & track_sets[right]
            videos = video_sets[left] & video_sets[right]
            overlaps[f'{left}/{right}'] = {'shared_tracks': len(tracks), 'shared_recordings': len(videos),
                                         'track_examples': sorted(tracks)[:10], 'recording_examples': sorted(videos)[:10]}
            if tracks or videos:
                issues.append(f'{left}/{right}: shared tracks or recordings')
        report['sources'][name] = {'splits': summaries, 'overlaps': overlaps, 'issues': issues}
        if issues:
            report['within_source_isolation'] = 'review_required'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', nargs=4, action='append', metavar=('NAME', 'TRAIN_CSV', 'VAL_CSV', 'TEST_CSV'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hash-inputs', action='store_true', help='Add full SHA256 fingerprints (extra streaming pass)')
    args = parser.parse_args()
    sources = {}
    for name, train, val, test in args.source:
        if name in sources:
            parser.error('Each source name must be unique')
        sources[name] = dict(train=train, val=val, test=test)
    report = audit_sources(sources, args.hash_inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'within_source_isolation': report['within_source_isolation'], 'output': str(args.output)}))
    if report['within_source_isolation'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
