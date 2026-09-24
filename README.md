# Adaptive Conditional Gradient Descent

This repository contains the Julia code and data used in the numerical experiments of the paper [*Adaptive Conditional Gradient Descent*](https://arxiv.org/abs/2510.11440) by Abbas Khademi and Antonio Silveti-Falls.

The codes and data in this repository are a snapshot of the software and data that were used in the research reported in the paper.

## Cite

If you find this work useful in your research and/or applications, please cite both the paper and this repository.

- Paper (arXiv): <https://arxiv.org/abs/2510.11440>
- Repository: <https://github.com/abbaskhademi/ACGD>

Below are the BibTeX entries for citing the paper and this snapshot of the repository.

```bibtex
@article{khademi2025adaptiveconditionalgradientdescent,
  title         = {Adaptive Conditional Gradient Descent},
  author        = {Khademi, Abbas and Silveti-Falls, Antonio},
  year          = {2025},
  eprint        = {2510.11440},
  archivePrefix = {arXiv},
  primaryClass  = {math.OC},
  url           = {https://arxiv.org/abs/2510.11440}
}

@misc{khademi2025acgdcode,
  author       = {Khademi, Abbas and Silveti-Falls, Antonio},
  title        = {{ACGD}: Code for Adaptive Conditional Gradient Descent},
  howpublished = {GitHub repository},
  url          = {https://github.com/abbaskhademi/ACGD},
  note         = {Available for download at https://github.com/abbaskhademi/ACGD}
}
```

# Description

The paper proposes an adaptive step-size strategy for first-order methods that rely on a **linear minimization oracle (LMO)**, such as the Conditional Gradient (Frank–Wolfe) method and non-Euclidean Normalized Steepest Descent. Using a simple heuristic that estimates a *local* Lipschitz constant of the gradient, the step-size guarantees sufficient decrease at every iteration without requiring knowledge of the global Lipschitz constant. The resulting **Adaptive Conditional Gradient Descent (ACGD)** algorithm has convergence guarantees for non-convex, quasar-convex, and strongly convex settings.

This repository provides Julia implementations of ACGD and of competing step-size rules, together with scripts that reproduce the experiments in the paper.

## Problem and algorithm

The scripts address problems of the form

$$
\min_{x \in C} f(x),
$$

where $f$ is continuously differentiable with a (locally) Lipschitz gradient and $C$ is a compact convex set accessible through an LMO, $\mathrm{lmo}(g) \in \arg\min_{v \in C} \langle g, v\rangle$.

At iteration $k$, with $v^k = \mathrm{lmo}(\nabla f(x^k))$, $d^k = v^k - x^k$, and Frank–Wolfe gap $G_k = -\langle \nabla f(x^k), d^k \rangle$, the adaptive method

1. estimates the local Lipschitz constant from consecutive gradients,

$$
L_k = \gamma \left( \frac{\|\nabla f(x^k) - \nabla f(x^{k-1})\|}{\|x^k - x^{k-1}\|} + \delta \right),
$$

   where $\gamma > 0$ is a scaling factor and $\delta > 0$ is a small constant for numerical stability;

2. sets the step-size $t_k = \min\left\{ \dfrac{G_k}{L_k \|d^k\|^2},\, 1 \right\}$;

3. backtracks, i.e., multiplies $L_k$ by $\beta > 1$ and recomputes $t_k$, until the sufficient-decrease condition

$$
f(x^k) - f(x^k + t_k d^k) \;\ge\; t_k G_k - \frac{L_k}{2}\, t_k^2 \|d^k\|^2
$$

   holds;

4. updates $x^{k+1} = x^k + t_k d^k$ and stops when the Frank–Wolfe gap falls below a tolerance $\varepsilon$.

Default parameters in the code are $\gamma = 1/4$, $\delta = 10^{-10}$, $\beta = 2$, and $\varepsilon = 10^{-5}$.

## Methods compared

Each experiment script implements the following step-size rules (function names in parentheses):

| Method | Description | Function |
|---|---|---|
| **Adaptive constant** | ACGD with a fixed scaling factor $\gamma$ | `conditional_gradient_adaptive` |
| **Adaptive adjustable** | ACGD where $\gamma$ is adjusted every 10 iterations based on the number of backtracking steps | `conditional_gradient_adjustable_scaling` |
| **Pure backtracking** | Backtracking line search on an estimate $M$ of the Lipschitz constant (Pedregosa et al., 2020) | `conditional_gradient_pure_backtracking` |
| **Auto-conditioned** | Step-size from the maximum of previously realized local curvature estimates | `conditional_gradient_auto_conditioned` |
| **Short-step** | Step-size $\min\{G_k/(L\|d^k\|^2), 1\}$ with a known global constant $L$ | `conditional_gradient_short_step` |
| **Open-loop** | Predefined step-size $t_k = 2/(k+2)$ | `conditional_gradient_open` |

## Repository structure

The repository has one Julia script per experiment in the paper. Each script is self-contained: it defines the problem, the solvers, runs the comparison, saves the results, and produces the figure.

| Script | Experiment |
|---|---|
| `lasso_problem.jl` | Lasso (least squares over an $\ell_1$-ball); 100 random instances, performance profile |
| `Least_Squares.jl` | Least squares |
| `Logistic_Regression_with_Regularization.jl` | Regularized logistic regression |
| `Sigmoid.jl` | Least-squares sigmoid regression (non-convex) |
| `matrix_balancing.jl` | Matrix balancing |
| `matrix_factorization.jl` | Matrix factorization (non-convex) |
| `Collaborative_Filtering.jl` | Collaborative filtering (MovieLens 100k) |
| `STQO.jl` | Standard quadratic optimization over the simplex (non-convex) |
| `Power_Shifted_Quadratic.jl` | Power-shifted quadratic problem |
| `Video_Co_Localization.jl` | Video co-localization (quadratic over a product of 33 simplices) |
| `traffic_assignment.jl` | Traffic assignment |

The subfolders (`Lasso_Problem`, `Logistic_Regression_with_Regularization`, `NonconvexQP_Simplex`, `Sigmoid_w8a`, `collaborative_filtering`, `least_squares`, `matrix_balancing`, `matrix_factorization`, `power_quadratic`, `video_colocalization`) contain the data and saved results associated with the corresponding experiments. The file `movielens100k.csv` is the MovieLens 100k rating data used in the collaborative filtering experiment.

# Dependencies

The scripts are written in [Julia](https://julialang.org/). The following packages are required:

- `LinearAlgebra`, `Statistics`, `Random`, `Downloads` — Julia standard library.
- [`Plots.jl`](https://github.com/JuliaPlots/Plots.jl) and [`Measures.jl`](https://github.com/JuliaGraphics/Measures.jl) — plotting.
- [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl) — Matplotlib backend used by `Plots` (requires a Python installation with Matplotlib).
- [`MAT.jl`](https://github.com/JuliaIO/MAT.jl) — reading and writing `.mat` files.
- [`BenchmarkProfiles.jl`](https://github.com/JuliaSmoothOptimizers/BenchmarkProfiles.jl) — Dolan–Moré performance profiles.

To install all packages, run in the Julia REPL:

```julia
using Pkg
Pkg.add(["Plots", "Measures", "PyPlot", "MAT", "BenchmarkProfiles"])
```

## Replicating Results

To replicate the results of a given experiment, run the corresponding script from the repository root. For example, for the Lasso experiment:

```bash
julia lasso_problem.jl
```

Each script creates an output subfolder next to the script (for example `lasso/` or `video_colocalization/`) and stores there:

- per-iteration histories (objective values, Frank–Wolfe gaps, iteration times, and local Lipschitz estimates $L_k$) as `.mat` files, one per method;
- a plain-text report or summary of the run;
- the figure of the experiment in `.eps` format (for instance, a performance profile or a plot of the local Lipschitz estimates).

Random instances are generated from fixed seeds, so the experiments are reproducible.

## Example Usage

Below is a minimal example showing how to run ACGD on a custom problem over an $\ell_1$-ball. It reuses the solver `conditional_gradient_adaptive` defined in `lasso_problem.jl`.

```julia
using LinearAlgebra, Random

Random.seed!(1)
m, n, tau = 200, 1000, 10

# Data: b = A * x_star with a sparse x_star strictly inside the l1-ball of radius tau
A = randn(m, n)
x_star = zeros(n); x_star[randperm(n)[1:tau]] .= randn(tau)
x_star .*= 0.9 * tau / norm(x_star, 1)
b = A * x_star

# Objective, gradient, and linear minimization oracle over {x : ||x||_1 <= tau}
f(x)      = sum((b - A * x) .^ 2)
grad_f(x) = 2 * A' * (A * x - b)
function lmo(g)
    i = argmax(abs.(g))
    v = zeros(length(g)); v[i] = -tau * sign(g[i])
    return v
end

x0 = zeros(n)

# Run ACGD (adaptive constant) with scaling factor gamma = 1/4
res = conditional_gradient_adaptive(f, grad_f, lmo, x0, 1/4; epsilon = 1e-5)

# Output description:
# - res.x          : final iterate
# - res.values     : objective value at each iteration
# - res.gaps       : Frank-Wolfe gap at each iteration
# - res.L_ks       : local Lipschitz estimate L_k at each iteration
# - res.k          : number of iterations
# - res.solved     : true if the stopping criterion was met
# - res.total_time : total running time (seconds)
```

Any problem for which the gradient and an LMO are available can be plugged in the same way; only `f`, `grad_f`, `lmo`, and the starting point `x0` need to be changed.

## Data

- **Lasso, least squares, logistic regression, matrix factorization, standard quadratic problems.** Instances are generated inside the scripts from fixed random seeds.
- **Video co-localization.** The script downloads the file `aeroplane_data_small.mat` automatically (from the [boostfw](https://github.com/cyrillewcombettes/boostfw) repository) on first use if it is not already present.
- **Collaborative filtering.** Uses the MovieLens 100k ratings provided in `movielens100k.csv`.
- **Logistic regression and sigmoid regression.** Use datasets from the [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/) collection.

Colab notebooks for the same experiments, including those for the Normalized Steepest Descent variant, are available in the companion repository [abbaskhademi/Adaptive-Step-Size](https://github.com/abbaskhademi/Adaptive-Step-Size).

# Contact

We appreciate your interest in our work. For questions, issues with the scripts, or feedback, please contact:

A. Khademi: <abbaskhademi92@gmail.com>

A. Silveti-Falls: <tonys.falls@gmail.com>

If you find this work useful for your research, please cite the paper and repository as indicated in the [Cite](#cite) section.
