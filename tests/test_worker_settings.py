"""Server-scale concurrency must be opt-in on the laptop."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
CODE = "import json,config; print(json.dumps([config.WORKER_LIMIT,config.NUM_EXTRACTION_WORKERS,config.NUM_AVA_DOWNLOAD_WORKERS,config.IMAGE_NUM_DATALOADER_WORKERS]))"


class WorkerConfigurationTests(unittest.TestCase):
    def run_config(self, value):
        env = dict(os.environ)
        env.pop('ASD_WORKER_LIMIT', None)
        if value is not None:
            env['ASD_WORKER_LIMIT'] = value
        return subprocess.run([sys.executable, '-c', CODE], cwd=ROOT, env=env, capture_output=True, text=True)

    def test_default_worker_pools_are_small(self):
        result = self.run_config(None)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(1 <= count <= 4 for count in json.loads(result.stdout)))

    def test_explicit_server_limit_restores_original_capacity(self):
        result = self.run_config('64')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [64, 64, 24, 32])

    def test_invalid_worker_limit_fails_before_processing(self):
        for value in ['0', '-1', 'invalid']:
            with self.subTest(value=value):
                result = self.run_config(value)
                self.assertNotEqual(result.returncode, 0)
