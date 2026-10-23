# Software validation of the timing corrections

The checks below were run on 2026-10-21 and 2026-10-22 using Julia 1.11.6,
one Julia thread and one OpenBLAS thread. The checked-in `Manifest.toml` records
the resolved package versions; `requirements-plot.txt` records the Python
plotting environment. This environment is newly resolved and is not claimed
to match the unpublished historical package environment.

This code update includes software validation and the measurement protocol.
Corrected benchmark outputs, rankings and performance claims remain separate
and are not published here.

## Passed checks

- `test/runtests.jl`: **541 assertions**. Covers exact ten-step scaling blocks,
  rejection counting, objective evaluation counts, cached accepted objectives,
  tuple/matrix post-step stopping, accepted curvature before decay, initial
  stationary points, zero directions and curvature, nonfinite evaluations,
  backtracking limits, AdaBB failures, elapsed-time boundaries, deadline overruns,
  RNG isolation, repeated timing, failure-aware profiles, manifest provenance,
  syntax of all eleven entrypoints, and small generated problems with gradient,
  feasibility, pairing and stopping-rule checks.
- `test/wrappers.jl`: **60 public solver interfaces**, with four assertions each,
  including matrix-factorization adapters and all constrained/unconstrained
  variants. An additional development run checked the two-step environment cap
  on all sixty wrappers.
- `test/integration.jl`: **42 assertions** with the resolved packages. Covers the
  actual Zygote paths for balancing, logistic and sigmoid regression; Graphs and
  Zygote for traffic, including objective normalization and flow checks; and a
  MAT export/import round trip for the new metadata.
- `test/benchmark_measurement.jl`: **289 assertions**. Covers compatible default
  timing, balanced method schedules and preservation of complete-call timing,
  compilation, allocation and garbage-collection diagnostics.
- `test/remaining_measurement.jl`: **546 assertions**. Covers balanced schedules
  for even and odd method counts, atomic checkpoints and stable numeric-input
  hashes across MAT export/import.
- `test/remaining_analysis.jl`: the original **27 assertions** passed for
  independent objective, residual, feasibility and stopping-rule calculations.
  The later artifact-selection changes described below have not yet had a
  runtime test in the publication snapshot.
- An independent numerical check matched twelve AdaBB steps to the paper's
  recurrence, including its first theta initialization.
- A bounded Lasso entrypoint check exercised report generation, MAT histories,
  manifests and an all-failed performance profile. This was a pipeline check,
  not evidence of convergence or comparative speed.
- `git diff --check` passes. Historical MAT files, reports and figures are
  unchanged.

The solver and measurement-harness source matches the tested snapshot. The
matrix-factorization analyzer subsequently received two fixes: selecting the
driver's existing prefixed MAT filename and resolving `TOML` correctly in its
error handler. These changes were statically reviewed, and seven artifact-selection
assertions were added to `test/remaining_analysis.jl`. Their runtime verification
is pending; the earlier 27 passing assertions do not certify these new paths.

## Run the checks

Python-only runner preflight regressions (no Julia or benchmark execution):

```sh
python -m unittest discover -s test -p 'test_*.py' -v
```

The queue requires readable, nonempty datasets only for its selected jobs: `a9a`
and `w8a` in `--data-dir` for logistic and sigmoid respectively,
`movielens100k.csv` in the source root for collaborative filtering, and
`video_colocalization/aeroplane_data_small.mat` for video co-localization.
The full suite requires all four before launching any tests. Stage them before
starting a run so driver downloads cannot change the captured input hashes.
These checks validate staging, not dataset format or scientific correctness.

Core solver, wrapper and measurement checks use Julia standard libraries:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --startup-file=no --gcthreads=1 --project=. test/runtests.jl
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --startup-file=no --gcthreads=1 --project=. test/wrappers.jl
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --startup-file=no --gcthreads=1 --project=. test/benchmark_measurement.jl
```

The integration, checkpoint and analysis checks require the instantiated project
packages:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --startup-file=no --gcthreads=1 --project=. test/integration.jl
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --startup-file=no --gcthreads=1 --project=. test/remaining_measurement.jl
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --startup-file=no --gcthreads=1 --project=. test/remaining_analysis.jl
```

Run these checks separately from timed experiments so they do not compete for
the machine's resources.

## What these checks do not establish

Software checks and bounded pilots do not certify the full 100-instance
comparisons or the 500-step curvature illustrations. Inspect each full run's
saved inputs, failure handling and final validation report before using its
figures. Interrupted timing samples must be remeasured, and all derived summaries
must then be regenerated from the corrected saved data.

No alternative-parameter sweep has been performed, and the claim that alternative
settings were uncompetitive remains unsupported. The AD integration tests use
small synthetic injected datasets; public a9a/w8a inputs must be staged separately
with their content hashes. Full-run validation and manuscript release requirements
remain listed in [TIMING_PROTOCOL.md](TIMING_PROTOCOL.md).

The report values summarized in [EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md) describe
historical outcomes only. They must not be presented as measurements of the
corrected implementation.
