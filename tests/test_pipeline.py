"""Synthetic ECG only: no downloads and no clinical/benchmark claims."""
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from check_pipeline import main as pipeline_check
from check_large_files import oversized
from create_issues import tasks
from download_data import verify_archive, verify_files
from leaderboard import generate
from mlee.baseline import eer, read_record
from mlee.experiments import execute


class PipelineTests(unittest.TestCase):
    def test_generated_ecg_end_to_end(self):
        pipeline_check()

    def test_eer_known_cases(self):
        self.assertEqual(eer([0, 0, 1, 1], [0, .1, .9, 1]), 0)
        self.assertAlmostEqual(eer([0, 1], [1, 1]), .5)

    def test_reader_rejects_unsupported_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            header = Path(tmp) / 'bad.hea'
            header.write_text('bad 2 500 10\nbad.dat 212 2000\nbad.dat 212 2000\n')
            with self.assertRaisesRegex(ValueError, 'Unsupported'):
                read_record(header)

    def test_integrity_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'bad.zip'
            archive.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                verify_archive(archive)
            (root / 'SHA256SUMS.txt').write_text('0 ../outside\n')
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                verify_files(root)

    def test_large_file_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'large'
            p.write_bytes(b'x' * 11)
            self.assertEqual(oversized([p], limit=10), [str(p)])

    def test_task_parser(self):
        parsed = list(tasks('## 8 Nov\n### DATA-01: Check\nOwner:\nAcceptance: inspect\n'))
        self.assertEqual(parsed[0][0], 'DATA-01: Check')
        self.assertIn('Milestone: 8 Nov', parsed[0][1])

    def test_failed_run_retains_provenance_and_not_in_leaderboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'config.yaml'
            config.write_text(json.dumps(dict(experiment='missing',seed=42,data='data',
                method='ac_dct_svm_frozen',protocol='last_usable_record')))
            with self.assertRaisesRegex(ValueError, 'No ECG-ID'):
                execute(config, root=root)
            runs = list((root / 'results/missing').iterdir())
            meta = json.loads((runs[0] / 'run.json').read_text())
            self.assertEqual(meta['status'], 'failed')
            self.assertIsNone(meta['git_hash'])
            self.assertEqual(meta['seed'], 42)
            self.assertTrue((runs[0] / 'config.yaml').exists())
            generate(root)
            self.assertNotIn('[missing/', (root/'results/LEADERBOARD.md').read_text())

    def test_notebooks_have_no_outputs(self):
        for path in Path('notebooks').glob('*.ipynb'):
            for cell in json.loads(path.read_text())['cells']:
                if cell['cell_type'] == 'code':
                    self.assertFalse(cell['outputs'])
                    self.assertIsNone(cell['execution_count'])


if __name__ == '__main__':
    unittest.main()
