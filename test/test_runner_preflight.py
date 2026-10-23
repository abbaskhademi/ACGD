"""Python-only runner staging checks; never launch Julia or a benchmark.

Run with: python -m unittest discover -s test -p 'test_*.py' -v
"""

from contextlib import ExitStack, redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_remaining_benchmarks as queue
import run_remaining_suite as suite


class DatasetPreflightTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.source = self.root / "source"
        self.data = self.root / "inputs/libsvm"
        self.depot = self.root / "depot"
        for directory in (self.source, self.data, self.depot):
            directory.mkdir(parents=True)
        self.julia = self.root / "julia"
        self.julia.write_text("placeholder; must never execute\n")
        self.julia.chmod(0o700)
        sources = {"Project.toml", "Manifest.toml", "scripts/run_remaining_benchmarks.py"}
        sources.update(job.script for job in queue.JOBS)
        sources.update("scripts/" + job.analyzer for job in queue.JOBS)
        sources.update("test/" + name for name in suite.TESTS)
        for name in sources:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("source placeholder\n")
        self.datasets = {
            "logistic": self.data / "a9a",
            "sigmoid": self.data / "w8a",
            "collaborative_filtering": self.source / "movielens100k.csv",
            "video_colocalization": self.source / "video_colocalization/aeroplane_data_small.mat",
        }
        self.stack.enter_context(patch.object(queue.shutil, "which", return_value="/mock/taskset"))
        self.stack.enter_context(patch.object(queue.os, "sched_getaffinity", return_value={2}, create=True))
        self.popen = self.stack.enter_context(patch.object(queue.subprocess, "Popen"))
        self.queue_run = self.stack.enter_context(patch.object(queue, "run_command"))
        self.suite_run = self.stack.enter_context(patch.object(suite, "run_command"))

    def arguments(self, module, only=None):
        common = ["--julia", str(self.julia), "--depot", str(self.depot), "--cpu", "2"]
        if module is suite:
            return common + ["--run-dir", str(self.root)]
        args = common + ["--source-dir", str(self.source), "--data-dir", str(self.data),
                         "--output-dir", str(self.root / "output"), "--mode", "pilot"]
        return args if only is None else args + ["--only", only]

    def stage(self, identifiers=None):
        for identifier in self.datasets if identifiers is None else identifiers:
            path = self.datasets[identifier]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"nonempty staging fixture\n")

    def assert_rejected(self, module, path, only=None):
        stderr = io.StringIO()
        with patch.object(module, "tree_checksums") as hashes, redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                module.main(self.arguments(module, only))
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(str(path), stderr.getvalue())
        hashes.assert_not_called()
        self.popen.assert_not_called()
        self.queue_run.assert_not_called()
        self.suite_run.assert_not_called()
        self.assertFalse((self.root / "output").exists())
        self.assertFalse((self.root / "suite_status.json").exists())
        self.assertFalse((self.root / "tests").exists())

    def test_missing_datasets_rejected_by_both_entrypoints(self):
        for module in (queue, suite):
            for identifier, path in self.datasets.items():
                with self.subTest(runner=module.__name__, dataset=identifier):
                    self.stage()
                    path.unlink()
                    self.assert_rejected(module, path)

    def test_empty_datasets_rejected_by_both_entrypoints(self):
        for module in (queue, suite):
            for identifier, path in self.datasets.items():
                with self.subTest(runner=module.__name__, dataset=identifier):
                    self.stage()
                    path.write_bytes(b"")
                    self.assert_rejected(module, path)

    def test_directory_datasets_rejected_by_both_entrypoints(self):
        for module in (queue, suite):
            for identifier, path in self.datasets.items():
                with self.subTest(runner=module.__name__, dataset=identifier):
                    self.stage()
                    path.unlink()
                    path.mkdir()
                    self.assert_rejected(module, path)
                    path.rmdir()

    def test_unreadable_datasets_rejected_by_both_entrypoints(self):
        self.stage()
        original_open = Path.open
        for module in (queue, suite):
            for identifier, path in self.datasets.items():
                with self.subTest(runner=module.__name__, dataset=identifier):
                    def guarded_open(candidate, *args, **kwargs):
                        if candidate == path:
                            raise PermissionError("mock unreadable dataset")
                        return original_open(candidate, *args, **kwargs)
                    with patch.object(Path, "open", guarded_open):
                        self.assert_rejected(module, path)

    def test_selected_jobs_accept_only_their_own_dataset(self):
        for job in queue.JOBS:
            with self.subTest(job=job.identifier):
                for path in self.datasets.values():
                    path.unlink(missing_ok=True)
                if job.identifier in self.datasets:
                    self.stage([job.identifier])
                args = queue.parse_arguments(self.arguments(queue, job.identifier))
                self.assertEqual([chosen.identifier for chosen in args.jobs], [job.identifier])
        self.popen.assert_not_called()

    def test_selected_jobs_reject_their_missing_dataset(self):
        for identifier, path in self.datasets.items():
            with self.subTest(job=identifier):
                self.assert_rejected(queue, path, identifier)

    def test_combined_selection_requires_both_libsvm_inputs(self):
        self.stage(["logistic"])
        self.assert_rejected(queue, self.datasets["sigmoid"], "logistic,sigmoid")
        self.stage(["sigmoid"])
        args = queue.parse_arguments(self.arguments(queue, "logistic,sigmoid"))
        self.assertEqual([job.identifier for job in args.jobs], ["logistic", "sigmoid"])

    def test_all_staged_inputs_pass_both_entrypoints(self):
        self.stage()
        for module, executor in ((queue, "execute_queue"), (suite, "execute_suite")):
            with self.subTest(runner=module.__name__):
                with patch.object(module, executor, return_value=0) as execute, \
                     patch.object(module.signal, "signal"):
                    self.assertEqual(module.main(self.arguments(module)), 0)
                execute.assert_called_once()
        self.popen.assert_not_called()
        self.queue_run.assert_not_called()
        self.suite_run.assert_not_called()

    def test_manifest_required_by_both_entrypoints(self):
        self.stage()
        manifest = self.source / "Manifest.toml"
        manifest.unlink()
        for module in (queue, suite):
            with self.subTest(runner=module.__name__):
                self.assert_rejected(module, manifest)


if __name__ == "__main__":
    unittest.main()
