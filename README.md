# Adaptive Conditional Gradient Descent

This repository contains the Julia code and data used in the numerical experiments of the paper *Adaptive Conditional Gradient Descent* by Abbas Khademi and Antonio Silveti-Falls.

The codes and data in this repository are a snapshot of the software and data that were used in the research reported in the paper.

## 5. Numerical Experiments

The following sections identify the Julia source files corresponding to each numerical experiment.

### 5.1. Lasso Problem
- Source code: `lasso_problem.jl`

### 5.2. Matrix Balancing Problem
- Source code: `matrix_balancing.jl`

### 5.3. Matrix Factorization
- Source code: `matrix_factorization.jl`

### 5.4. Simplex Quadratic Optimization Problem
- Source code: `STQO.jl`

### 5.5. Traffic Assignment
- Source code: `traffic_assignment.jl`

### 5.6. Sparse Least Squares Sigmoid Regression Problem
- Source code: `Sigmoid.jl`

### 5.7. Least Squares Problem
- Source code: `Least_Squares.jl`

### 5.8. Logistic Regression with Regularization
- Source code: `Logistic_Regression_with_Regularization.jl`

### 5.9. Power Shifted Quadratic
- Source code: `Power_Shifted_Quadratic.jl`

### 5.10. Collaborative Filtering
- Source code: `Collaborative_Filtering.jl`

### 5.11. Video Co-Localization
- Source code: `Video_Co_Localization.jl`

### 5.12. Ill-Conditioned Quadratic
- Source code: `quadratic_adjustable_lmo.jl`

## How to Run

Clone the repository and navigate to its root directory:

```sh
git clone https://github.com/abbaskhademi/ACGD.git
cd ACGD
```

Use Julia 1.11.6 and the checked-in `Project.toml` and `Manifest.toml`.

Instantiate the project environment:

```sh
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
```

For example, to run the Lasso experiment from the Julia REPL:

```julia
julia> include("lasso_problem.jl")
```

To run another experiment, include its corresponding Julia source file from the repository root. Run each benchmark script in a separate Julia session to avoid conflicts between function definitions.

Run the test suite:

```sh
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --startup-file=no --project=. test/runtests.jl
```

The core solver, wrapper and measurement tests require only Julia standard libraries and avoid executing the expensive benchmark entrypoints. Integration, MAT checkpoint and saved-output analysis tests use the resolved project packages; their commands are listed in the validation document.

Benchmark plots additionally need a Python environment with matplotlib configured for PyCall; see `requirements-plot.txt`.

Configure it with:

```sh
PYTHON=/path/to/python julia --project=. -e 'using Pkg; Pkg.build("PyCall")'
```

For a headless machine, set:

```sh
MPLBACKEND=Agg
```

## Timing Corrections and Reproducibility

The timing corrections are described in [the experiment audit](docs/EXPERIMENT_AUDIT.md) and [the reproducibility protocol](docs/TIMING_PROTOCOL.md).

Historical reports and figures are preserved; they have not been regenerated or certified by these fixes. New runs write to `corrected_results/`.

See [software validation](docs/VALIDATION.md) for completed checks and the validation still required before updating the manuscript.

Corrected experimental outputs and new performance claims are not included in this code update.

See the reproducibility protocol before running the complete experiments or interpreting their performance profiles.

## Contact

We appreciate your interest in our work. For questions, issues with the scripts, or feedback, please contact:

- A. Khademi: <abbaskhademi92@gmail.com>
- A. Silveti-Falls: <tonys.falls@gmail.com>
