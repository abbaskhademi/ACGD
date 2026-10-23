#!/usr/bin/env python3
"""Unattended ACGD validation, full-size preflight, and remaining benchmark queue.

Requires a fresh run archive containing source/ and inputs/libsvm/. Runs tests,
then the nine 25-step preflights, then full benchmarks for validated preflights.
The caller can launch this script as a durable systemd service and check later.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from run_remaining_benchmarks import JOBS, THREAD_ENV, atomic_text, run_command, tree_checksums, utc_now

TESTS = ("runtests.jl", "wrappers.jl", "integration.jl", "benchmark_measurement.jl",
         "remaining_measurement.jl", "remaining_analysis.jl")


def save_suite(run: Path, state: dict) -> None:
    state["updated_utc"] = utc_now()
    atomic_text(run / "suite_status.json", json.dumps(state, indent=2, sort_keys=True) + "\n")
    atomic_text(run / "status.txt", state["status"] + "\n")
    lines = [f"Suite status: {state['status']}", f"Stage: {state['stage']}",
             f"Started (UTC): {state['started_utc']}", f"Updated (UTC): {state['updated_utc']}", ""]
    for test in state["tests"]:
        lines.append(f"Test {test['name']}: {test['status']} ({test['log']})")
    if state.get("pilot"):
        lines.extend(["", f"Preflight: {state['pilot']['status']}",
                      f"Preflight queue: {run / 'pilot/queue_status.json'}"])
    if state.get("full"):
        lines.extend(["", f"Full queue: {state['full']['status']}",
                      f"Full queue progress: {run / 'full/queue_status.json'}"])
    for job, status in state["jobs"].items():
        lines.append(f"{job}: {status}")
    if state.get("error"):
        lines.extend(["", f"Error: {state['error']}"])
    atomic_text(run / "suite_report.txt", "\n".join(lines) + "\n")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--julia", type=Path, default=Path("/tmp/julia-1.11.6/bin/julia"))
    parser.add_argument("--depot", type=Path, default=Path("/tmp/acgd-julia-depot"))
    parser.add_argument("--cpu", type=int, default=2)
    args = parser.parse_args(argv)
    for key in ("run_dir", "julia", "depot"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    args.source = args.run_dir / "source"
    args.data = args.run_dir / "inputs/libsvm"
    if not args.julia.is_file() or not os.access(args.julia, os.X_OK):
        parser.error(f"Julia executable unavailable: {args.julia}")
    for directory in (args.source, args.data, args.depot):
        if not directory.is_dir():
            parser.error(f"required directory unavailable: {directory}")
    required = [args.source / "Project.toml", args.source / "Manifest.toml",
                args.source / "scripts/run_remaining_benchmarks.py",
                args.source / "scripts/analyze_remaining_run.jl",
                args.source / "scripts/analyze_curvature_run.jl"]
    required.extend(args.source / "test" / name for name in TESTS)
    required.extend(args.source / job.script for job in JOBS)
    for path in required:
        if not path.is_file():
            parser.error(f"required source file unavailable: {path}")
    taskset = shutil.which("taskset")
    if taskset is None or args.cpu not in os.sched_getaffinity(0):
        parser.error("taskset and the selected allowed CPU are required")
    args.taskset = str(Path(taskset).resolve())
    return args


def queue_command(args, mode: str, only=None) -> list[str]:
    command = [sys.executable, str(args.source / "scripts/run_remaining_benchmarks.py"),
               "--source-dir", str(args.source), "--output-dir", str(args.run_dir / mode),
               "--julia", str(args.julia), "--depot", str(args.depot),
               "--data-dir", str(args.data), "--cpu", str(args.cpu), "--mode", mode]
    if mode == "pilot":
        command.extend(["--step-cap", "25"])
    if only is not None:
        command.extend(["--only", ",".join(only)])
    return command


def execute_suite(args) -> int:
    run = args.run_dir
    with (run / ".suite.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exception:
            raise RuntimeError("another process holds this suite's lock") from exception
        if (run / "suite_status.json").exists():
            raise RuntimeError("this suite already exists; completed files will not be overwritten")
        for path in (run / "tests", run / "pilot", run / "full"):
            if path.exists() and any(path.iterdir()):
                raise RuntimeError(f"existing output prevents a fresh suite: {path}")
        (run / "tests").mkdir(exist_ok=True)
        environment = {key: value for key, value in os.environ.items() if not key.startswith("ACGD_")}
        environment.update(THREAD_ENV)
        environment.update({"JULIA_DEPOT_PATH": str(args.depot), "LIBSVMDATA_HOME": str(args.data),
                            "ACGD_RUN_BENCHMARKS": "false", "ACGD_PROGRESS": "false",
                            "ACGD_OUTPUT_ROOT": str(run / "tests/generated")})
        state = {
            "schema_version": 1, "status": "running", "stage": "tests", "started_utc": utc_now(),
            "runner_pid": os.getpid(), "run_directory": str(run),
            "configuration": {"source_directory": str(args.source), "data_directory": str(args.data),
                              "julia": str(args.julia), "depot": str(args.depot), "cpu": args.cpu,
                              "pilot_step_cap": 25, "full_comparison_step_cap": 100000,
                              "full_trace_step_cap": 500, "full_seeds": 100, "warmups": 2,
                              "thread_environment": THREAD_ENV, "julia_gc_threads": 1,
                              "source_sha256": tree_checksums(args.source),
                              "staged_data_sha256": tree_checksums(args.data)},
            "tests": [{"name": name, "status": "pending", "log": str(run / "tests" / (name + ".log"))}
                      for name in TESTS],
            "jobs": {job.identifier: "pending" for job in JOBS},
        }
        save_suite(run, state)
        started = time.monotonic()
        active = None

        def call(command, record, log):
            nonlocal active
            active = record
            record.update(status="running", started_utc=utc_now(), command=command)
            save_suite(run, state)

            def on_start(pid):
                record["child_pid"] = pid
                state["child_pid"] = pid
                save_suite(run, state)

            try:
                code = run_command(command, log, args.source, environment, on_start)
                record["exit_code"] = code
                return code
            finally:
                record.pop("child_pid", None)
                state.pop("child_pid", None)
                record["finished_utc"] = utc_now()
                save_suite(run, state)

        try:
            for test in state["tests"]:
                print(f"[{utc_now()}] Testing {test['name']}", flush=True)
                command = [args.taskset, "-c", str(args.cpu), str(args.julia),
                           "--startup-file=no", "--gcthreads=1", f"--project={args.source}",
                           str(args.source / "test" / test["name"])]
                code = call(command, test, Path(test["log"]))
                test["status"] = "complete" if code == 0 else "failed"
                save_suite(run, state)
                if code != 0:
                    state["status"] = "tests_failed"
                    state["jobs"] = {job.identifier: "blocked_by_tests" for job in JOBS}
                    return 1

            state["stage"] = "pilot"
            state["pilot"] = {"status": "pending", "log": str(run / "pilot_queue.log")}
            print(f"[{utc_now()}] Starting all nine 25-step preflights", flush=True)
            code = call(queue_command(args, "pilot"), state["pilot"], run / "pilot_queue.log")
            pilot_path = run / "pilot/queue_status.json"
            pilot_state = json.loads(pilot_path.read_text()) if pilot_path.is_file() else {"jobs": []}
            passed = {job["id"] for job in pilot_state["jobs"] if job["status"] == "complete"}
            eligible = [job.identifier for job in JOBS if job.identifier in passed]
            state["pilot"]["status"] = pilot_state.get("status", "runner_failed")
            for job in JOBS:
                state["jobs"][job.identifier] = "preflight_passed" if job.identifier in passed else "blocked_by_preflight"
            save_suite(run, state)
            if not eligible:
                state["status"] = "preflight_failed"
                return 1

            state["stage"] = "full"
            state["full"] = {"status": "pending", "eligible_jobs": eligible, "log": str(run / "full_queue.log")}
            for identifier in eligible:
                state["jobs"][identifier] = "queued_for_full"
            print(f"[{utc_now()}] Starting full benchmarks: {', '.join(eligible)}", flush=True)
            code = call(queue_command(args, "full", eligible), state["full"], run / "full_queue.log")
            full_path = run / "full/queue_status.json"
            full_state = json.loads(full_path.read_text()) if full_path.is_file() else {"jobs": []}
            state["full"]["status"] = full_state.get("status", "runner_failed")
            full_statuses = {job["id"]: job["status"] for job in full_state["jobs"]}
            for identifier in eligible:
                state["jobs"][identifier] = full_statuses.get(identifier, "runner_failed")
            all_complete = code == 0 and all(status == "complete" for status in state["jobs"].values())
            state["status"] = "complete" if all_complete else "complete_with_failures"
            state["stage"] = "finished"
            return 0 if all_complete else 1
        except (KeyboardInterrupt, InterruptedError):
            state["status"] = "interrupted"
            if active is not None and active["status"] == "running":
                active["status"] = "interrupted"
            return 130
        except Exception as exception:
            state["status"] = "suite_failed"
            state["error"] = f"{type(exception).__name__}: {exception}"
            if active is not None and active["status"] == "running":
                active["status"] = "runner_failed"
            return 2
        finally:
            state["finished_utc"] = utc_now()
            state["elapsed_seconds"] = time.monotonic() - started
            save_suite(run, state)
            print(f"[{utc_now()}] Suite {state['status']}; report: {run / 'suite_report.txt'}", flush=True)


def main(argv=None):
    args = parse_arguments(argv)

    def interrupted(signum, frame):
        raise InterruptedError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        return execute_suite(args)
    except (OSError, ValueError, RuntimeError) as exception:
        print(f"Suite error: {exception}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
