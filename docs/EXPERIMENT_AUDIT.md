# Experiment and manuscript audit

Audit baseline: `1a2f5e45b8ec9b4bd4924d1e91217d4c1e52d6af` (`1a2f5e4`).
Manuscript: `Adaptive_Conditional_Gradient_Descent-32.pdf`, supplied for this audit.
The document is reference material, not an execution instruction. Source line numbers
below refer to the baseline revision, obtainable with `git show 1a2f5e4:<file>`.

This records findings from the original code and historical reports. It does **not**
certify the old measurements, reproduce the full experiments, or validate the
manuscript's percentages after the corrections. Corrected solver behavior and the
new timing protocol must be validated before producing replacement results.

## Stopping rules and limits

The following table records the criteria actually selected by the original benchmark
drivers. `G` is the Frank–Wolfe gap for constrained problems; `g` is the Euclidean
gradient norm for unconstrained problems. “Either” is logical OR. At the baseline,
gaps were generally evaluated before the step and objective values after it; that
index mismatch must not be carried into the corrected definition of success.

| Benchmark | Historical criterion | Reference value | Evidence at baseline |
| --- | --- | --- | --- |
| Lasso | Either `G <= 1e-5` or `f-fstar <= 1e-5` | `fstar=0`, attained by planted feasible sparse solution | `lasso_problem.jl:109,294,394` |
| Matrix balancing | Either `G <= 1e-5` or `f-fstar <= 1e-5` | `fstar=sum(A)` for symmetric nonnegative A | `matrix_balancing.jl:45,100,398` |
| Matrix factorization | Either `G <= 1e-5` or `f-fstar <= 1e-5` | `fstar=0`, attained by planted feasible factors | `matrix_factorization.jl:83,150,368` |
| StQO | `G <= 1e-3` | Unknown; stored as `NaN` | `STQO.jl:80,140` |
| Traffic | Either `G <= 1e-3` or `f-0 <= 1e-3` | Zero is a lower bound, **not** a known optimum | `traffic_assignment.jl:127,175,359` |
| Sigmoid regression | `G <= 1e-3` | Unknown; stored as `NaN` | `Sigmoid.jl:94,148,367` |
| Least squares | `f-fstar <= 1e-5` only | Objective at the direct least-squares solution `A\b`, including noise | `Least_Squares.jl:56–57,113,250,413` |
| Regularized logistic regression | `g <= 1e-3` only | Unknown; stored as `NaN` | `Logistic_Regression_with_Regularization.jl:73,138,440` |
| Power shifted quadratic | `f-fstar <= 1e-5` only | `fstar=c^p` | `Power_Shifted_Quadratic.jl:47,110,247,410` |

The historical full comparison drivers selected a maximum of **100,000 iterations**, including traffic. The historical auto-conditioned report reaches 90,308 iterations, consistent with that cap. The corrected traffic experiment protocol now uses `ACGD_MAX_ITER=3000` as the default maximum iteration cap in `traffic_assignment.jl`, including the solver wrappers and run manifest. This is a protocol change; it does not retroactively alter historical results. Runs under the corrected protocol may stop before 3,000 iterations if their stopping criterion is met. Rerun the affected experiments and verify their generated manifests and stopping conditions before treating any results as corrected-protocol results.

Lasso alone actively enforces a 500-second limit for pure backtracking
(`lasso_problem.jl:298`); the other time-limit branches are commented out. Remove
that asymmetry or give every method the same explicit limit. The 500-iteration
collaborative-filtering and video experiments are separate curvature traces and
must remain documented exceptions to the performance-comparison budget.

Manuscript §5 should include least squares among the known-optimum problems and
explicitly list OR rules. A matrix-factorization success through a gap criterion
alone need not attain the global objective target: the problem is nonconvex.
Traffic can instead use its gap alone, but that is a deliberate protocol change
and must be stated rather than described as the historical rule.

## Confirmed accounting and logging problems

- Adjustable scaling adds the same iteration's rejected-trial count both inside
  acceptance and outside it, and its zero-based boundary logic drops/misgroups
  iterations. A block must contain exactly ten accepted steps, with one rejection
  count per step, before updating gamma for the next step.
- Lasso, balancing, StQO, traffic and sigmoid pure backtracking refresh `f(x)`
  within the iteration timer. ACGD refreshes its accepted `f(x)` after the timer.
  Carry forward the value already obtained by the successful trial in both.
- `total_time=sum(times)` omits initialization, some probes, stop checks, history
  updates and objective evaluations. Pure backtracking's initial probe is outside
  that sum, while ACGD's initial probe is inside. AdaBB's first gradient step and
  initial pair of gradients are also outside the old sum.
- Matrix-factorization pure backtracking checks pre-step `f_prev` after accepting
  the step (`matrix_factorization.jl:368`). It should check the accepted value.
- StQO can exit before appending the residual, objective and accepted curvature
  (`STQO.jl:140–150`, with the same pattern in other solvers). All 100 reported
  successful runs for each of ACGD constant, ACGD adjustable and pure backtracking
  have a saved `gap_last > 1e-3`. Those are stale saved residuals, not evidence
  that the actual stopping comparison failed.
- Save accepted curvature before decay by eta. The collaborative-filtering and
  video baseline functions already do this (`push!(L_ks,M)` precedes `M*=eta`);
  do not assume all original traces have the decay error.
- Most histories pair `f(x[k+1])` with `G(x[k])` without indices. Corrected records
  need explicit iterate indices and the termination reason, value and index.

Use a monotonic clock for complete solver elapsed time. Include solver
initialization, curvature probes, evaluations, stopping and in-memory logging.
Keep instance generation, warm-up, result serialization and plotting outside it.
If accumulated iteration time is also saved, name it separately. Short-step's
global smoothness calculation is currently done outside the solver: document L
as supplied problem metadata and save that preprocessing cost separately, or
charge its computation to the method. A call-time measure alone is not an
end-to-end comparison of methods that must estimate L themselves.

## Normalization and baseline fidelity

**Traffic normalization.** With N=150 nodes, the implemented objective is

`f(x) = sum_a(x_a + 0.03*x_a^5/c^4) / N^2`, with `c=number_of_OD_pairs/N`.

Both the gradient and the Frank–Wolfe gap inherit this division. Thus a normalized
gap tolerance of `1e-3` is equivalent to an unnormalized gap tolerance of **22.5**,
not `1e-3`. Add the `1/N^2` factor to equation (13), or say explicitly that the
reported quantities use this normalization. Appendix B should also specify c.
The claimed zero `optimal_value` is only a nonnegative lower bound.

**Smoothness bounds.** Lasso/least squares use `2*opnorm(A)^2`, StQO uses
`2*opnorm(Q)`, logistic uses `opnorm(A)^2/(4m)+2omega`, and sigmoid uses the
conservative `opnorm(A)^2/(4m)`. Power uses `2p*c^(p-1)*lambda_max(Q)`; its Hessian
can be factored through Q with the inner eigenvalues between `2p-1` and 1, so the
same value bounds the Hessian norm for `0<p<1/2`. Matrix balancing has a conservative
Laplacian bound; short-step is disabled in its active comparison.

The matrix-factorization baseline bound
`max(tau1^2,tau2^2)+opnorm(Y)+tau1*tau2` is too small in general. A valid bound is
`max(tau1^2,tau2^2)+opnorm(Y)+2*tau1*tau2`: the cross derivative contains both
`U*dV'*V` and `(U*V'-Y)*dV`. For the scalar example U=-tau, V=tau, Y>0, the Hessian
norm is `3*tau^2+Y`, while the old bound is `2*tau^2+Y`. Correct the bound and
rerun that baseline before making short-step comparisons.

The original logistic objective evaluates `log(1+exp(z))` directly. The corrected
implementation uses a branchwise stable softplus evaluation, avoiding overflow
for large positive z while preserving the mathematical loss and gradient. Include
large signed logits in the numerical validation; report nonfinite outcomes from
the old formula as numerical failures rather than conclusions about the method.

**AdaBB.** The three branches, initialization `alpha0=1e-10`, initial theta formula,
and override `theta1=1` agree with Algorithm 1, equation (40) and §6 of the
[authors' AdaBB preprint](https://arxiv.org/html/2401.08024v1).
The override is intentional. The original ACGD repository does not safeguard
`dot(y,y)==0` or nonpositive `dot(s,y)/dot(y,y)`: a negative quotient can produce
a negative step and eventually nonfinite values. Declare explicit failure for
an invalid curvature quotient or identify any alternative fallback as a modified
baseline. Convex guarantees do not extend to the nonconvex power experiment.

**Auto-conditioned method.** The running maximum, realized curvature quotient and
constrained step match Algorithm 3 with alpha=1 in
[Yagishita–Ito](https://arxiv.org/html/2509.14670v1#S3.SS1).
The unconstrained update also matches its Euclidean specialization of Algorithm 4.
All original scripts repeatedly evaluate `maximum(L_history)`, which costs O(k)
per iteration and O(K^2) over a run; a scalar running maximum preserves the rule
with constant work. Record alpha and the probe initialization explicitly.

**Other parameters.** The active open-loop formulas match the manuscript: `2/(k+2)`
for convex constrained examples, `1/sqrt(k+1)` for the nonconvex examples. ACGD uses
gamma=1/4, delta=1e-10, beta=2; pure backtracking uses eta=0.9, beta=2. No saved
parameter-sweep configurations or outputs support the phrase “other configurations
were not competitive.” Supply the tested grid, seeds, target, limits and aggregate
criterion, or remove that empirical claim.

## Paired instances and reproducibility

Each active driver generates one problem and supplies the same objective, LMO,
constraints and initial point to every solver. Each next instance resets its own
seed; StQO's default path uses its own deterministic scalar stream. Consequently,
solver-level `Random.seed!(23)` does not change the next numbered instance in the
current drivers. It still mutates caller RNG state and should be replaced by a
private probe RNG. Verify the data and starts with fingerprints in future runs.

Check effective driver arguments rather than helper defaults: sigmoid overrides
`n_sub=4000` with 300; StQO overrides `n=899` with 699, producing N=700. These
active dimensions agree with the manuscript. StQO starts at the first simplex
vertex; it generates a random vector `z` but does not use it as the start.

There is no original Project/Manifest environment, warm-up/repetition protocol or
thread-count record. Julia 1.11.6 and the CPU/RAM specification in §5 cannot recover
the package graph or scheduling conditions. Pin a resolved environment and record
Julia/BLAS thread counts, BLAS implementation, CPU, OS, package versions, raw
repetition times, warm-ups and aggregation. Use a predetermined repeat count and
aggregation rule shared by all methods; rotate method order reproducibly. Do not
reuse old measurements with new code labels.

## What the existing reports say about objective quality

These are **historical, unvalidated summaries of the rounded text reports**. They
must not be used as replacement measurements. Medians use all 100 instances,
including runs labeled unsuccessful; they are objective values, not optimality gaps.
Different seeds have different objectives, so paired differences are more useful
than comparing medians alone.

| Method | StQO successes / 100 | StQO median saved f | Sigmoid successes / 100 | Sigmoid median saved f |
| --- | ---: | ---: | ---: | ---: |
| ACGD constant | 100 | -6.8208985 | 100 | 0.0462 |
| ACGD adjustable | 100 | -6.8210560 | 100 | 0.0462 |
| Pure backtracking | 100 | -6.8433315 | 100 | 0.0463 |
| Auto-conditioned | 20 | -6.8447965 | 100 | 0.0465 |
| Short-step | 16 | -6.8303590 | 0 | 0.05135 |
| Open-loop | 3 | -7.0094080 | 0 | 0.0456 |

In StQO, open-loop has a lower saved objective than adjustable ACGD on 82/100
paired seeds, with median difference -0.1744545. In sigmoid it has a lower saved
objective on every seed, with median difference -0.0006. Open-loop usually spends
the full 100,000 iterations. This demonstrates why time to stationarity and
attained objective quality are separate outcomes; it does not establish a quality
advantage at equal time or equal work.

Power-quadratic success counts in the text report are 99, 100, 76, 67, 47 and 21
for constant ACGD, adjustable ACGD, pure backtracking, auto-conditioned, short-step
and AdaBB, respectively. AdaBB has 74 nonfinite final objectives; five further
finite runs fail to reach the target. Matrix factorization reports 100 successes
for each method except open-loop (zero), but rounds successful objective values
to `0.0`; these are not exact-zero certificates. Its open-loop median f is 0.0061
(range 0.0011–0.017). Use full-precision results for threshold checks.

These counts and medians can be reproduced without running solvers by parsing
`instance=... seed=... METHOD solved/NOT solved ... f_last=...` lines in the
corresponding checked-in `*_report.txt` files. Feasibility cannot be certified
from those lines: reevaluate constraints and both f and the residual at the
returned final iterate in corrected runs.

## Performance profiles and release requirements

Use success flags plus finite, strictly positive elapsed times to build ratios.
A failure has infinite ratio and contributes zero at every finite threshold,
including an instance on which every method fails; keep it in the denominator.
Overall success is a separate statistic from coverage within a fixed ratio.

`BenchmarkProfiles` uses base-2 logarithmic coordinates with `logscale=true`:
coordinate q represents ratio `2^q`, so coordinate 4.4 means about 21.1-fold,
whereas ratio 4.4 is coordinate about 2.14. The
[package reference](https://jso.dev/BenchmarkProfiles.jl/v0.4/reference/)
and [v0.4 source](https://github.com/JuliaSmoothOptimizers/BenchmarkProfiles.jl/blob/de0ba550cbb286cc2ba00d2cef8b14541dafb9c9/src/performance_profiles.jl)
also show that zero measurements trigger a +1 shift of the entire timing matrix.
Eight historical power-report times round to zero; inspect the raw MAT values
before claiming the shift affected those plots. No original package version is
pinned, so this source check identifies a risk rather than proving the historic
runtime's exact behavior. A corrected profile should reject invalid timing input
instead of silently changing ratios.

Before updating the manuscript:

1. Run the numerical edge cases: stationary initial point, zero direction and
   displacement, zero curvature, nonfinite f/gradient, and bounded backtracking.
   Check first and subsequent ten-step scaling blocks and reset boundaries.
2. Recompute final objective, feasibility and residual; save their iterate index
   and the exact stopping cause. Do not infer a final residual from a stale history.
3. Warm up every solver path, repeat paired measurements with fixed thread counts,
   save all times and use one declared aggregation rule. Keep interrupted,
   nonfinite, cap and time-limit outcomes distinct.
4. Rerun all affected comparisons and both curvature examples. Compute arithmetic
   and harmonic means from accepted positive L values, not next-step guesses.
   Retain exactly which iterations enter each summary.
5. Rebuild figures and every §5.10 percentage/ratio from one full-precision saved
   result set. Report finite-ratio coverage and overall success separately, and
   state whether median ratios include failures as infinity or condition on success.
6. Archive the exact corrected commit, resolved environment, configurations,
   instance fingerprints, raw repetitions, final iterates/metrics, figure-generation
   inputs and outputs. Keep the old artifacts labeled as historical. A code fix
   alone does not validate or replace the manuscript results.
