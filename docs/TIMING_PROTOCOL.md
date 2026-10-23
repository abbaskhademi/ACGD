# Corrected experiment protocol

The historical outputs in the repository were distributed at `1a2f5e4`. They must
not be combined with measurements from the corrected solvers. See
[EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md) for the original behavior and manuscript
changes. A successful small validation run is not a replacement for the full
100-instance experiments.

## What is timed

`total_time` is now an alias for **complete solver elapsed time**, measured with
Julia's monotonic `time_ns()`. The clock starts on entry to the shared solver.
It includes argument checks, copies, objective and gradient initialization,
curvature probes, all trial evaluations, stopping tests, and in-memory histories.
Every method uses the same boundary, including AdaBB's first step. No solver
prints, plots, or writes files inside this clock.

`initialization_time` ends after the initial residual, initial stopping test and
any needed curvature probe. `iteration_time` sums subsequent loop attempts,
including a failed final attempt. `times[1]` contains initialization; the remaining
entries contain loop-attempt times. Thus `times` is **not** necessarily the same
length as the iterate histories, and its sum is not the authoritative elapsed
time (small control/return overhead remains). Do not reuse historical plotting
code that assumes `times[1] == 0` or that elapsed time excludes initialization.

The problem instance, reference optimum, and supplied global smoothness bound
are problem metadata prepared before the solver call. Their computation, data
loading, warmup, post-run feasibility checks, figures and file I/O are outside
solver timing. This is a solve-time comparison **given those inputs**; it does
not claim to measure the cost of discovering a global bound for short steps.
If reporting application end-to-end time, separately include preparation and
the method-specific cost of obtaining that bound.

## Stopping, histories and exceptional outcomes

`values[j]`, `gaps[j]`, and `iterate_indices[j]` describe the same iterate. Index
zero is recorded and tested before any step. The gradient and oracle at an
accepted point are computed once, used for its stopping test, and cached for
the next step. Consequently the returned iterate, final objective and final
residual agree, including when an objective criterion triggers convergence.

`termination_iteration`, `stopping_criterion`, `final_residual`, and `status`
describe why and where a run stopped. The audit's stopping-rule table is retained:
least squares and the power problem use objective error only; known-optimum
constrained cases use objective error OR gap; logistic, StQO and sigmoid use
stationarity. Traffic retains its historical OR rule against the lower bound
zero, explicitly distinguished from a known optimum.

All performance comparisons default to 100,000 accepted steps and no wall-time
limit (`Inf`). A finite `max_time` must be passed identically to all methods in a
comparison. Limits are checked at safe boundaries, including within backtracking;
an indivisible objective/oracle evaluation may overrun a deadline, and a target
attained after that deadline counts as a time-limit failure. The two
curvature illustrations retain their explicit 500-step budget.

Every accepted trial's objective is reused. `L_ks[k]` is the curvature used for
accepted step k, before pure backtracking applies eta for the next iteration.
`backtracks[k]` counts rejected trials once. Adjustable scaling updates gamma
after steps 10, 20, … from exactly that block's ten counts; `gamma_iterations`
records these boundaries. The first ten steps use the initial gamma.

Backtracking is bounded and handles nonfinite trial values by increasing
curvature. Invalid initial/probe evaluations and stalled steps return explicit
failure statuses. Zero curvature is floored by `delta`; finite-difference probes
use a local `Xoshiro(23)`, preserving Julia 1.11.6's original seeded probe stream
without resetting the caller's RNG. Both backtracking families share an allowance
of eight floating-point epsilons times `max(1, abs(f_old), abs(f_trial))` in their
decrease tests. This is a numerical safeguard, not an additional stopping target.

AdaBB preserves the published three-case step and numerical initialization.
A zero denominator or nonpositive BB quotient now returns a named failure
(`adabb_zero_curvature` or `adabb_nonpositive_curvature`). No nonconvex fallback
is silently substituted. This changes failure handling for power-quadratic
experiments and must be reported with those results.

## Repeated paired measurements

Each driver prepares one instance per seed and passes the same data, starting
point, constraints and target to every method. The original generator streams
are retained. Solvers copy their initial state. The collaborative-filtering
approximate singular-vector oracle uses an explicit deterministic starting vector;
record this change when comparing with historical runs.

The Lasso driver defaults to one warmup and three measured repetitions per method.
Its run order rotates across repetitions and instances. The reported time is the median
complete elapsed time. All measured repeats must succeed for a method to count
as successful; otherwise its performance-profile time is infinity. Raw repeat
times, statuses, iterations, objectives and residuals are saved. Detailed histories
belong to the repeat nearest the median, identified by `representative_repeat`.
Scalar initialization/iteration times are also medians; they need not sum to the
median complete elapsed time.

The power-shifted quadratic and remaining drivers use two warmups and complete
balanced measurement blocks,
using an external timer
around the complete public solver call. The internal solver time remains saved
separately as `core_elapsed_time`. Each instance uses a Williams design
that balances positions and directed neighboring method pairs; a local seeded RNG
permutes its labels and rows. Every actual execution order is saved alongside the
display order. Full garbage collection runs before each call, outside its timer;
collections triggered during the call remain included. Raw outer/core times,
allocation bytes, GC time, and compilation time are retained for every repetition.
These controls are optional in the shared harness; Lasso retains the defaults
described above. Six-method balanced comparisons use six measured repetitions;
five-method matrix-balancing and traffic comparisons use ten. The three-method
curvature illustrations use six. Even method counts require complete blocks of
n orders; odd counts require 2n orders, combining each order with its reversal.
All methods within a problem have the same repeat count and limits.

Power checkpoints also retain each generated `Q`, starting point, exponent,
shift, smoothness bound, and data hash, plus the representative final iterate.
Objective values and gradient norms are independently reevaluated after timing.
Failures, including AdaBB nonpositive-curvature exits and iteration limits, keep
their raw reasons and receive infinite performance-profile times.

The remaining comparison drivers save numeric instance inputs, representative
final iterates, and a canonical input SHA-256 in atomic per-seed MAT checkpoints.
The checksum is stable across the MAT round trip. Independent postprocessing
recomputes objective values, stopping residuals and feasibility from those inputs.
The traffic validator uses a separate DAG shortest-cost calculation, while its
aggregate-flow feasibility check remains necessary rather than sufficient for
multicommodity feasibility. The original feasible-start/convex-combination
construction is preserved. Dataset file hashes include the exact public a9a/w8a
files and the bundled MovieLens/video inputs. Optional per-call progress messages
are printed after the timer stops.

`scripts/run_remaining_benchmarks.py` runs the remaining experiments sequentially,
with separate output directories and automatic validation. Its pilot mode uses
one instance and a small iteration cap and cannot produce full-benchmark claims.
Full mode uses seeds 1:100 and 100,000 accepted steps for the seven comparisons,
and the original 500-step cap for the two curvature illustrations. The queue
retains failed-run evidence and proceeds to independent experiments; the final
status distinguishes a complete validated queue from one containing failures.
`scripts/run_remaining_suite.py` is the unattended entrypoint: it first runs the
tests and bounded pilots against a frozen source snapshot, then schedules full
runs only for experiments whose pilots passed. A failed pilot is recorded as
blocked and does not prevent independent validated experiments from running.
Inspect `suite_status.json` and each job's analysis report before using results.

The queue launchers require Linux (`taskset`, CPU affinity and file locks) and
Python 3.11 or later. Their current default Julia and depot paths refer to the
development machine's `/tmp` installation. On another machine, explicitly supply
`--julia`, `--depot` and `--cpu`; the selected CPU must be allowed by that machine's
affinity policy. Do not run another timed workload concurrently.

Prepare a fresh archive with this layout before starting the suite:

```text
/path/to/run/
  source/                 frozen repository source, environment and bundled datasets
  inputs/libsvm/a9a        staged public input
  inputs/libsvm/w8a        staged public input
  inputs/libsvm/provenance.json
```

Use `scripts/fetch_benchmark_data.py` to stage the LIBSVM files and record their
hashes. Preserve the source snapshot throughout the run; generated tests, pilots
and full results go beside `source/`, never inside it. For example, after replacing
the paths and choosing an available CPU (2 in this example):

```sh
python3 /path/to/run/source/scripts/run_remaining_suite.py \
  --run-dir /path/to/run \
  --julia /path/to/julia-1.11.6/bin/julia \
  --depot /path/to/julia-depot \
  --cpu 2
```

The depot must already contain the instantiated project packages and configured
plotting environment. The suite performs no package installation or downloads.
The lower-level `run_remaining_benchmarks.py` uses the same explicit Julia, depot
and CPU options, plus `--source-dir`, `--data-dir`, `--output-dir` and
`--mode pilot|full`; its output directory must be outside the frozen source.

Set `ACGD_WARMUPS` and `ACGD_REPEATS` explicitly for a release run, retaining at
least one warmup. Use a resolved project environment and fixed Julia and BLAS
thread counts, e.g.:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  julia --startup-file=no --project=. test/runtests.jl
ACGD_WARMUPS=1 ACGD_REPEATS=3 JULIA_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \
  julia --startup-file=no --project=. lasso_problem.jl
```

Run the other benchmark entrypoints separately in the same environment. The
`ACGD_RUN_BENCHMARKS=false` switch suppresses their sweeps; it does not suppress
all imports or dataset loading. The test loader extracts definitions without
executing those entrypoints.

`ACGD_SEED_COUNT` (default 100) and `ACGD_MAX_ITER` (default 100000) permit
small pipeline checks. `ACGD_TRACE_ITER` controls the two illustration runs
(default 500). For example, set seed count to 1 and the iteration cap to 2 for
a smoke check, and set `ACGD_OUTPUT_ROOT=corrected_results/smoke` to separate
those deliberately incomplete runs. These controls are recorded in the manifest;
they must be reset to manuscript values for full comparisons.

Each run writes a manifest with the actual Julia/package versions, Julia/BLAS
thread counts, BLAS implementation, CPU, source revision, dirty state, source
hashes, and experiment configuration. Corrected outputs go under
`corrected_results/<problem>/`. Archive that directory before rerunning the same
problem; entrypoints use fixed output names within it. The checked-in environment
pins Julia 1.11.6 to match the manuscript; it is a newly resolved package lock,
not a reconstruction of the missing historical lock.

## Profiles, quality and release checklist

Failures retain infinite ratios. Rows where all methods fail remain in the
denominator. Finite times must be positive; zero times are never shifted by an
arbitrary constant. Profile plotting uses actual ratios on a log2-scaled axis.
Finite-cutoff coverage and overall success rate are saved separately. Arithmetic
and harmonic accepted-curvature summaries must use the corrected `L_ks` sequence.

Final feasibility is checked separately from the solver clock. The traffic check
of aggregate conservation is necessary but not by itself a multicommodity
feasibility certificate; its feasible start and convex-combination steps supply
the construction invariant. For StQO, sigmoid, and factorization, report objective
distributions alongside stationarity/target attainment. Do not interpret a faster
stationarity stop as a better objective, or compare objective quality at unequal
work budgets as an equal-work experiment.

Before manuscript release:

1. Run the complete corrected comparisons and two curvature traces with the
   fixed environment and saved configuration.
2. Save and inspect failure reasons, feasibility, objective quality and residuals;
   investigate inconsistent measured repeats.
3. Recreate every profile, percentage, speed ratio and curvature summary from
   those same saved outputs. Old reports are historical evidence only.
4. Test the candidate backtracking parameter configurations in
   `backtracking_candidates.toml` on the same paired instances. Until those
   measurements exist, remove or qualify the claim that alternatives were
   uncompetitive; this repository contains no supporting historical records.
5. Commit/review the exact source and configuration, retain the generated lock
   and run manifests, hash the datasets and outputs, and archive everything used
   for the final manuscript together. A dirty-tree hash records development
   provenance but is not a substitute for an archived release revision.
