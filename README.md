# Adaptive Conditional Gradient Descent

This repository contains the Julia code and data used in the numerical experiments of the paper Adaptive Conditional Gradient Descent by Abbas Khademi and Antonio Silveti-Falls.

The codes and data in this repository are a snapshot of the software and data that were used in the research reported in the paper.

The timing corrections are described in [the experiment audit](docs/EXPERIMENT_AUDIT.md)
and [the reproducibility protocol](docs/TIMING_PROTOCOL.md). Historical reports and
figures are preserved; they have not been regenerated or certified by these fixes.
New runs write to `corrected_results/`.
See [software validation](docs/VALIDATION.md) for completed checks and the
validation still required before updating the manuscript. Corrected experimental
outputs and new performance claims are not included in this code update.

Use Julia **1.11.6** and the checked-in `Project.toml` / `Manifest.toml`:

```sh
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --startup-file=no --project=. test/runtests.jl
```

The core solver, wrapper and measurement tests require only Julia standard
libraries and avoid executing the expensive benchmark entrypoints. Integration,
MAT checkpoint and saved-output analysis tests use the resolved project packages;
their commands are listed in the validation document. Benchmark plots additionally
need a Python environment with matplotlib configured for PyCall; see
`requirements-plot.txt`.
Configure it with `PYTHON=/path/to/python julia --project=. -e 'using Pkg; Pkg.build("PyCall")'`.
For a headless machine set `MPLBACKEND=Agg`. See the protocol before running the
complete experiments or interpreting their performance profiles.


# Contact

We appreciate your interest in our work. For questions, issues with the scripts, or feedback, please contact:

A. Khademi: <abbaskhademi92@gmail.com>

A. Silveti-Falls: <tonys.falls@gmail.com>

