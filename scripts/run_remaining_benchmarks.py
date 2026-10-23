#!/usr/bin/env python3
"""Run the remaining ACGD experiments serially, preserving every attempt.

The caller should launch this process as a durable service. This script performs
no downloads, installations, or remote operations. A frozen source directory and
already staged LIBSVM datasets are required. Resume is explicit and only skips
validated, checksummed completions from exactly the same configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any


@dataclass(frozen=True)
class Job:
    identifier: str
    script: str
    problem: str
    methods: int
    trace: bool = False

    @property
    def repeats(self) -> int:
        return self.methods if self.methods % 2 == 0 else 2 * self.methods

    @property
    def analyzer(self) -> str:
        return "analyze_curvature_run.jl" if self.trace else "analyze_remaining_run.jl"


JOBS = (
    Job("video_colocalization", "Video_Co_Localization.jl", "video_colocalization", 3, True),
    Job("collaborative_filtering", "Collaborative_Filtering.jl", "collaborative_filtering", 3, True),
    Job("logistic", "Logistic_Regression_with_Regularization.jl", "Logistic_Regression_with_Regularization", 6),
    Job("least_squares", "Least_Squares.jl", "least_squares", 6),
    Job("matrix_balancing", "matrix_balancing.jl", "matrix_balancing", 5),
    Job("stqo", "STQO.jl", "NonconvexQP_Simplex", 6),
    Job("sigmoid", "Sigmoid.jl", "Sigmoid_w8a", 6),
    Job("matrix_factorization", "matrix_factorization.jl", "matrix_factorization", 6),
    Job("traffic_assignment", "traffic_assignment.jl", "traffic_assignment", 5),
)
THREAD_ENV = {
    "JULIA_NUM_THREADS": "1",
    "JULIA_NUM_PRECOMPILE_TASKS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "MPLBACKEND": "Agg",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_checksums(directory: Path) -> dict[str, str]:
    """Hash portable inputs/outputs without traversing unrelated repository data."""
    checksums = {}
    for current, directories, files in os.walk(directory):
        directories[:] = sorted(name for name in directories
                                if name not in {".git", "__pycache__", "corrected_results"})
        for filename in sorted(files):
            path = Path(current) / filename
            if path.is_file():
                checksums[path.relative_to(directory).as_posix()] = sha256_file(path)
    return checksums


def save_state(output: Path, state: dict[str, Any]) -> None:
    state["updated_utc"] = utc_now()
    atomic_text(output / "queue_status.json", json.dumps(state, indent=2, sort_keys=True) + "\n")
    atomic_text(output / "status.txt", state["status"] + "\n")


def write_report(output: Path, state: dict[str, Any]) -> None:
    lines = [f"Queue status: {state['status']}", f"Mode: {state['configuration']['mode']}",
             f"Started (UTC): {state.get('started_utc', 'not started')}",
             f"Updated (UTC): {state['updated_utc']}", ""]
    for job in state["jobs"]:
        lines.append(f"{job['id']}: {job['status']}")
        if job["attempts"]:
            attempt = job["attempts"][-1]
            lines.append(f"  Results: {attempt['result_directory']}")
            lines.append(f"  Driver exit: {attempt.get('driver_exit_code', 'pending')}; "
                         f"analysis exit: {attempt.get('analysis_exit_code', 'pending')}")
            if attempt.get("error"):
                lines.append(f"  Error: {attempt['error']}")
    atomic_text(output / "queue_report.txt", "\n".join(lines) + "\n")


def completed_files_valid(attempt: dict[str, Any]) -> bool:
    manifest = Path(attempt.get("checksums_path", ""))
    if not manifest.is_file():
        return False
    try:
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        actual = tree_checksums(Path(attempt["result_directory"]))
        return bool(recorded) and recorded == actual
    except (OSError, ValueError):
        return False


def run_command(command: list[str], log: Path, source: Path, environment: dict[str, str], on_start) -> int:
    """Wait for one child; if interrupted, stop only the child we just started."""
    with log.open("w", encoding="utf-8") as handle:
        handle.write("Command: " + json.dumps(command) + "\n")
        handle.flush()
        child = subprocess.Popen(command, cwd=source, env=environment,
                                 stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            on_start(child.pid)
            return child.wait()
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise


def job_environment(args, job: Job, results: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    environment.update({
        "JULIA_DEPOT_PATH": str(args.depot),
        "LIBSVMDATA_HOME": str(args.data_dir),
        "ACGD_RUN_BENCHMARKS": "true",
        "ACGD_PROGRESS": "true",
        "ACGD_OUTPUT_ROOT": str(results),
        "ACGD_SEED_COUNT": "1" if args.mode == "pilot" else "100",
        "ACGD_MAX_ITER": str(args.step_cap if args.mode == "pilot" else 100000),
        "ACGD_TRACE_ITER": str(args.step_cap if args.mode == "pilot" else 500),
        "ACGD_WARMUPS": "2",
        "ACGD_REPEATS": str(job.repeats),
    })
    return environment


def run_job(args, job: Job, entry: dict[str, Any], state: dict[str, Any]) -> None:
    attempts_root = args.output_dir / "jobs" / job.identifier
    attempts_root.mkdir(parents=True, exist_ok=True)
    number = len(entry["attempts"]) + 1
    attempt_dir = attempts_root / f"attempt_{number:03d}"
    while attempt_dir.exists():
        number += 1
        attempt_dir = attempts_root / f"attempt_{number:03d}"
    attempt_dir.mkdir()
    results = attempt_dir / "results"
    result_directory = results / job.problem
    environment = job_environment(args, job, results)
    command_prefix = [str(args.taskset), "-c", str(args.cpu), str(args.julia),
                      "--startup-file=no", "--gcthreads=1", f"--project={args.source_dir}"]
    driver_command = command_prefix + [str(args.source_dir / job.script)]
    analyzer_command = command_prefix + [str(args.source_dir / "scripts" / job.analyzer),
                                         str(result_directory)]
    attempt = {
        "number": number, "status": "running", "stage": "driver", "started_utc": utc_now(),
        "directory": str(attempt_dir), "result_directory": str(result_directory),
        "driver_log": str(attempt_dir / "run.log"), "analysis_log": str(attempt_dir / "analysis.log"),
        "driver_command": driver_command, "analysis_command": analyzer_command,
        "environment": {key: environment[key] for key in sorted(environment)
                        if key in THREAD_ENV or key.startswith("ACGD_")
                        or key in {"JULIA_DEPOT_PATH", "LIBSVMDATA_HOME"}},
    }
    entry["attempts"].append(attempt)
    entry["status"] = "running"
    state["active_job"] = job.identifier
    save_state(args.output_dir, state)
    print(f"[{utc_now()}] Starting {job.identifier}, attempt {number}", flush=True)
    started = time.monotonic()

    def on_start(pid):
        attempt["child_pid"] = pid
        save_state(args.output_dir, state)

    try:
        exit_code = run_command(driver_command, Path(attempt["driver_log"]), args.source_dir,
                                environment, on_start)
        attempt["driver_exit_code"] = exit_code
        attempt["driver_finished_utc"] = utc_now()
        attempt.pop("child_pid", None)
        if exit_code != 0:
            entry["status"] = attempt["status"] = "benchmark_failed"
        else:
            attempt["stage"] = "analysis"
            save_state(args.output_dir, state)
            exit_code = run_command(analyzer_command, Path(attempt["analysis_log"]), args.source_dir,
                                    environment, on_start)
            attempt["analysis_exit_code"] = exit_code
            attempt["analysis_finished_utc"] = utc_now()
            attempt.pop("child_pid", None)
            if exit_code != 0:
                entry["status"] = attempt["status"] = "validation_failed"
            else:
                checksums = tree_checksums(result_directory)
                if not checksums:
                    raise RuntimeError("analysis returned success but the result directory is empty")
                checksum_path = attempt_dir / "output_checksums.json"
                atomic_text(checksum_path, json.dumps(checksums, indent=2, sort_keys=True) + "\n")
                attempt["checksums_path"] = str(checksum_path)
                entry["status"] = attempt["status"] = "complete"
    except (KeyboardInterrupt, InterruptedError):
        entry["status"] = attempt["status"] = "interrupted"
        raise
    except Exception as exception:
        entry["status"] = attempt["status"] = "runner_failed"
        attempt["error"] = f"{type(exception).__name__}: {exception}"
    finally:
        attempt.pop("child_pid", None)
        attempt["elapsed_seconds"] = time.monotonic() - started
        attempt["finished_utc"] = utc_now()
        state["active_job"] = None
        save_state(args.output_dir, state)
        write_report(args.output_dir, state)
    print(f"[{utc_now()}] {job.identifier}: {entry['status']}", flush=True)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--julia", type=Path, default=Path("/tmp/julia-1.11.6/bin/julia"))
    parser.add_argument("--depot", type=Path, default=Path("/tmp/acgd-julia-depot"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--only", help="comma-separated job ids: " + ",".join(job.identifier for job in JOBS))
    parser.add_argument("--step-cap", type=int, default=5,
                        help="pilot iteration cap (default 5); full caps are fixed at 100000/500")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for key in ("source_dir", "output_dir", "julia", "depot", "data_dir"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.step_cap < 1:
        parser.error("--step-cap must be positive")
    if args.mode == "full" and args.step_cap != 5:
        parser.error("--step-cap applies only to pilot mode; full caps cannot be overridden")
    if args.source_dir == args.output_dir or args.source_dir in args.output_dir.parents:
        parser.error("--output-dir must not be inside --source-dir")
    requested = None if not args.only else set(args.only.split(","))
    unknown = set() if requested is None else requested - {job.identifier for job in JOBS}
    if unknown:
        parser.error("unknown job ids: " + ", ".join(sorted(unknown)))
    args.jobs = [job for job in JOBS if requested is None or job.identifier in requested]
    for path in (args.source_dir, args.depot, args.data_dir):
        if not path.is_dir():
            parser.error(f"required directory does not exist: {path}")
    if not args.julia.is_file() or not os.access(args.julia, os.X_OK):
        parser.error(f"Julia executable is unavailable: {args.julia}")
    taskset = shutil.which("taskset")
    if taskset is None:
        parser.error("taskset executable is required")
    args.taskset = Path(taskset).resolve()
    if args.cpu not in os.sched_getaffinity(0):
        parser.error(f"CPU {args.cpu} is outside the current allowed affinity")
    required_files = {args.source_dir / "Project.toml"}
    for job in args.jobs:
        required_files.update((args.source_dir / job.script, args.source_dir / "scripts" / job.analyzer))
    for path in sorted(required_files):
        if not path.is_file():
            parser.error(f"required source file is unavailable: {path}")
    return args


def make_configuration(args) -> dict[str, Any]:
    return {
        "mode": args.mode, "source_dir": str(args.source_dir), "julia": str(args.julia),
        "depot": str(args.depot), "data_dir": str(args.data_dir), "cpu": args.cpu,
        "pilot_step_cap": args.step_cap if args.mode == "pilot" else None,
        "seeds": 1 if args.mode == "pilot" else 100, "warmups": 2,
        "comparison_step_cap": args.step_cap if args.mode == "pilot" else 100000,
        "trace_step_cap": args.step_cap if args.mode == "pilot" else 500,
        "thread_environment": THREAD_ENV, "julia_gc_threads": 1,
        "jobs": [asdict(job) | {"repeats": job.repeats} for job in args.jobs],
        "source_sha256": tree_checksums(args.source_dir),
        "staged_data_sha256": tree_checksums(args.data_dir),
    }


def execute_queue(args) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".queue.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exception:
            raise RuntimeError("another runner holds this output directory's queue lock") from exception
        configuration = make_configuration(args)
        status_path = args.output_dir / "queue_status.json"
        if status_path.exists():
            if not args.resume:
                raise RuntimeError("this queue already exists; use --resume with identical configuration")
            state = json.loads(status_path.read_text(encoding="utf-8"))
            if state["configuration"] != configuration:
                raise RuntimeError("resume configuration or source/data checksums changed; use a fresh output directory")
            state.setdefault("resumed_utc", []).append(utc_now())
            for entry in state["jobs"]:
                if entry["status"] == "complete" and not completed_files_valid(entry["attempts"][-1]):
                    raise RuntimeError(f"completed output checksum mismatch for {entry['id']}; preserve this run and use a fresh directory")
                if entry["status"] == "running":
                    entry["status"] = "interrupted"
                    entry["attempts"][-1]["status"] = "interrupted"
                    entry["attempts"][-1]["interruption_detected_utc"] = utc_now()
        else:
            if args.resume:
                raise RuntimeError("--resume requested but queue_status.json does not exist")
            jobs_path = args.output_dir / "jobs"
            if jobs_path.exists() and any(jobs_path.iterdir()):
                raise RuntimeError("existing job output without queue metadata; choose a fresh output directory")
            state = {"schema_version": 1, "configuration": configuration, "status": "pending",
                     "created_utc": utc_now(), "jobs": [
                         {"id": job.identifier, "status": "pending", "attempts": []} for job in args.jobs]}
            save_state(args.output_dir, state)
        state["status"] = "running"
        state.setdefault("started_utc", utc_now())
        state["runner_pid"] = os.getpid()
        save_state(args.output_dir, state)
        try:
            for job, entry in zip(args.jobs, state["jobs"], strict=True):
                if entry["status"] == "complete":
                    print(f"[{utc_now()}] Preserving completed {job.identifier}", flush=True)
                    continue
                run_job(args, job, entry, state)
        except (KeyboardInterrupt, InterruptedError):
            state["status"] = "interrupted"
            state["interrupted_utc"] = utc_now()
            save_state(args.output_dir, state)
            write_report(args.output_dir, state)
            return 130
        state["status"] = ("complete" if all(job["status"] == "complete" for job in state["jobs"])
                           else "complete_with_failures")
        state["finished_utc"] = utc_now()
        save_state(args.output_dir, state)
        write_report(args.output_dir, state)
        print(f"[{utc_now()}] Queue {state['status']}; report: {args.output_dir / 'queue_report.txt'}", flush=True)
        return 0 if state["status"] == "complete" else 1


def main(argv=None) -> int:
    args = parse_arguments(argv)

    def interrupted(signum, frame):
        raise InterruptedError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        return execute_queue(args)
    except (OSError, ValueError, RuntimeError) as exception:
        print(f"Queue error: {exception}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
