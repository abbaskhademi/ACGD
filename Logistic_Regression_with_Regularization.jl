# ==========================================================================
# Logistic_Regression_with_Regularization Frank-Wolfe: all 6 methods x N seeds -> combined performance profile
# ==========================================================================

using LinearAlgebra      # norm, dot, opnorm
using Statistics         # mean, median
using Random             # seed!, randperm, randn
using Plots              # plot!, savefig, display
using Measures           # mm (margin)
using PyPlot             # pyplot() backend + PyCall access for warnings filter
using MAT                # matwrite
using LIBSVMdata         # load_dataset
using Zygote             # gradient

pyplot()
PyPlot.PyCall.pyimport("warnings").filterwarnings("ignore")

# --- output goes into <run dir>/Logistic_Regression_with_Regularization/ -----------------------
problem_name = "Logistic_Regression_with_Regularization"
output_dir = joinpath(get(ENV, "ACGD_OUTPUT_ROOT", joinpath(@__DIR__, "corrected_results")), problem_name)
mkpath(output_dir)
instance_dir = joinpath(output_dir, "instances")
mkpath(instance_dir)

# sanitize method names into valid .mat struct field names
matvarname(s) = replace(s, r"[^A-Za-z0-9_]" => "_")

# --------------------------------------------------------------------
# 1. Problem generator — one seed -> one independent Logistic_Regression_with_Regularization instance
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# 1. Load the FULL a9a dataset ONCE (outside the seed loop)
# --------------------------------------------------------------------

println("Loading a9a dataset using LIBSVMdata...")
A_full, y_full = load_dataset("a9a",
#A_full, y_full = load_dataset("rcv1_train.binary",
    dense   = false,
    replace = false,
    verbose = true,
)


m_total = size(A_full, 1)
n_total = size(A_full, 2)
# --------------------------------------------------------------------
# 2. Problem generator — subsamples rows/cols to build one benchmark instance
# --------------------------------------------------------------------
function generate_problem(seed::Int; m_sub::Int=2000, n_sub::Int=123, tau::Float64=50.0)
    @assert m_total >= m_sub "m_total=$m_total < requested m_sub=$m_sub"
    @assert n_total >= n_sub "n_total=$n_total < requested n_sub=$n_sub"

    Random.seed!(seed)

    row_indices = randperm(m_total)[1:m_sub]
    col_indices = randperm(n_total)[1:n_sub]

    A = A_full[row_indices, col_indices]
    y = y_full[row_indices]

    m, n = size(A)  # This already captures m = size(A, 1)

    # Regularization parameter
    omega = 1/(n*m)  # Common choice; adjust as needed

    # Define regularized logistic loss function:
    # f(x) = (1/m) * sum(log(1 + exp(-y_i * a_i^T x))) + omega * ||x||_2^2
    # Stable softplus avoids overflow in objective and differentiated gradient.
    softplus(z) = z > 0 ? z + log1p(exp(-z)) : log1p(exp(z))
    f(x) = mean(softplus.(-y .* (A * x))) + omega * norm(x, 2)^2

    # Gradient via automatic differentiation
    grad_f(x) = Zygote.gradient(f, x)[1]

    optimal_value = NaN
    rho = 1.0
    function lmo(g)
        ng = norm(g)
        return ng > 0 ? -rho * g / ng : zeros(length(g))
    end

    x0 = zeros(n)

    return (f=f, grad_f=grad_f, lmo=lmo, x0=x0, rho=rho, A=A, y=y,
            seed=seed, optimal_value=optimal_value, m=m, n=n, omega=omega,
            row_indices=row_indices, col_indices=col_indices)
end

# ∇²f = (1/m) Aᵀ diag(σ(1-σ)) A + 2ωI,  σ(1-σ) ≤ 1/4
estimate_L(A, m, omega) = opnorm(Matrix(A))^2 / (4m) + 2*omega
# Shared solver implementation: full solver elapsed time includes initialization,
# curvature probes, stopping checks and in-memory history. Data/I/O stay outside.
if !isdefined(@__MODULE__, :ACGDSolvers)
    include(joinpath(@__DIR__, "src", "ACGDSolvers.jl"))
end
if !isdefined(@__MODULE__, :ACGDBenchmarkSupport)
    include(joinpath(@__DIR__, "scripts", "benchmark_support.jl"))
end
using .ACGDBenchmarkSupport

function conditional_gradient_adaptive(f, grad_f, lmo, x0, rho, gamma;
        epsilon=1e-3, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    isfinite(rho) && rho > 0 || throw(ArgumentError("rho must be finite and positive"))
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:adaptive, constrained=false, stop_rule=:stationarity,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, gamma=gamma, kwargs...)
    return r
end

function conditional_gradient_adjustable_scaling(f, grad_f, lmo, x0, rho, gamma;
        epsilon=1e-3, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    isfinite(rho) && rho > 0 || throw(ArgumentError("rho must be finite and positive"))
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:adjustable, constrained=false, stop_rule=:stationarity,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, gamma=gamma, kwargs...)
    return r
end

function conditional_gradient_pure_backtracking(f, grad_f, lmo, x0, rho;
        epsilon=1e-3, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    isfinite(rho) && rho > 0 || throw(ArgumentError("rho must be finite and positive"))
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:backtracking, constrained=false, stop_rule=:stationarity,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, kwargs...)
    return r
end

function conditional_gradient_short_step(f, grad_f, lmo, x0, rho, L;
        epsilon=1e-3, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    isfinite(rho) && rho > 0 || throw(ArgumentError("rho must be finite and positive"))
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:short, constrained=false, stop_rule=:stationarity,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, L=L, kwargs...)
    return r
end

function conditional_gradient_auto_conditioned(f, grad_f, lmo, x0, rho;
        epsilon=1e-3, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    isfinite(rho) && rho > 0 || throw(ArgumentError("rho must be finite and positive"))
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:auto, constrained=false, stop_rule=:stationarity,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, kwargs...)
    return r
end

function adabb(f, grad_f, x0, alpha0;
        epsilon=1e-3, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, nothing, x0;
        method=:adabb, constrained=false, stop_rule=:stationarity,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, alpha0=alpha0, kwargs...)
    return r
end

if get(ENV, "ACGD_RUN_BENCHMARKS", "true") == "true"

# --------------------------------------------------------------------
# 3. Run all methods on each of the N seeded problems
# --------------------------------------------------------------------
println("\n================ Logistic_Regression_with_Regularization ================")
seeds  = 1:parse(Int, get(ENV, "ACGD_SEED_COUNT", "100"))
isempty(seeds) && throw(ArgumentError("ACGD_SEED_COUNT must be at least one"))

gamma0 = 1/4
alpha0 = 1e-10




method_order = ["Adaptive constant", "Adaptive adjustable", "Pure backtracking",
                "Auto-conditioned",
                 "Short-step",
                 "AdaBB"]

problem_parameters = (m_sub=2000, n_sub=123, tau=50.0)
warmups = parse(Int, get(ENV, "ACGD_WARMUPS", "1"))
repeats = parse(Int, get(ENV, "ACGD_REPEATS", "1"))
order_seed = 20261021

all_times  = Dict(name => fill(Inf, length(seeds)) for name in method_order)
all_solved = Dict(name => falses(length(seeds)) for name in method_order)

# base Julia Dict (NOT Dictionaries.jl's `Dictionary`)
per_iter     = Dict(name => Dict{String,Any}() for name in method_order)
report_lines = String[]                      # text report accumulator

write_run_manifest(joinpath(output_dir, "run_manifest.toml"), Dict(
    "problem" => problem_name, "script" => basename(@__FILE__),
    "stop_rule" => "stationarity", "epsilon" => 1e-3,
    "max_iter_cap" => parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")), "max_time" => Inf,
    "alpha0" => alpha0,
    "gamma" => gamma0, "beta" => 2.0, "eta" => 0.9, "block_size" => 10,
    "probe_seed" => 23, "delta" => 1e-10,
    "generator_rng" => "original Julia default RNG explicitly reseeded per instance; see pinned Julia version",
    "method_order" => method_order,
    "execution_base_order" => ["AdaBB", "Auto-conditioned", "Adaptive constant", "Adaptive adjustable", "Pure backtracking", "Short-step"],
    "dataset" => "a9a", "dataset_rows" => m_total, "dataset_columns" => n_total,
    "problem_parameters" => Dict(string(k) => v for (k, v) in pairs(problem_parameters)),
    "execution_order" => "rotating method order; actual schedules saved per instance",
    "order_mode" => "rotate", "order_seed" => order_seed,
    "timing_boundary" => "external public solver call; core time also recorded",
    "gc_before_call" => true,
    "input_provenance" => "numerical instance snapshot and SHA-256 saved in each instance checkpoint",
    "source_provenance" => "exact source SHA-256 recorded by write_run_manifest",
    "smoothness_preprocessing" => "problem constants computed before solver timing",
    "feasibility_scope" => "representative repeat checked outside solver timing",
    "run_controls" => Dict(name => get(ENV, name, default) for (name, default) in
        [("ACGD_OUTPUT_ROOT", joinpath(@__DIR__, "corrected_results")),
         ("ACGD_SEED_COUNT", "100"), ("ACGD_MAX_ITER", "100000"),
         ("ACGD_TRACE_ITER", "500"), ("ACGD_WARMUPS", "2"), ("ACGD_REPEATS", string(repeats))]),
    "warmups" => warmups,
    "repeats" => repeats,
    "seeds" => collect(seeds),
    "aggregation" => "median complete solver elapsed time; all repeats must solve",
); root=@__DIR__)

for (idx, s) in enumerate(seeds)
    println("\n================ instance = $idx | seed = $s ================")
    prob = generate_problem(s; problem_parameters...)
    L_est = estimate_L(prob.A, prob.m, prob.omega)
    snapshot = problem_snapshot(prob)
    snapshot["L_estimate"] = L_est
    input_hash = snapshot_sha256(snapshot)
    local fstar = prob.optimal_value   # unknown optimum; gradient-norm stopping

    calls = [
        "AdaBB"               => () -> adabb(prob.f, prob.grad_f, prob.x0, alpha0; fstar=fstar),
        "Auto-conditioned"    => () -> conditional_gradient_auto_conditioned(prob.f, prob.grad_f, prob.lmo, prob.x0, prob.rho; fstar=fstar),
        "Adaptive constant"   => () -> conditional_gradient_adaptive(prob.f, prob.grad_f, prob.lmo, prob.x0, prob.rho, gamma0; fstar=fstar),
        "Adaptive adjustable" => () -> conditional_gradient_adjustable_scaling(prob.f, prob.grad_f, prob.lmo, prob.x0, prob.rho, gamma0; fstar=fstar),
        "Pure backtracking"   => () -> conditional_gradient_pure_backtracking(prob.f, prob.grad_f, prob.lmo, prob.x0, prob.rho; fstar=fstar),
        "Short-step"          => () -> conditional_gradient_short_step(prob.f, prob.grad_f, prob.lmo, prob.x0, prob.rho, L_est; fstar=fstar),
    ]
    runs = benchmark_methods(calls; instance_index=idx,
        warmups=warmups, repeats=repeats, external_timing=true,
        gc_before_call=true, order_mode=:rotate, order_seed=order_seed)

    for name in method_order
        r = runs[name]
        all_solved[name][idx] = r.solved
        all_times[name][idx]  = r.solved ? r.total_time : Inf

        # AdaBB records reciprocal accepted step sizes; other methods record curvature.
        Lk = hasproperty(r, :L_ks) ? collect(Float64, r.L_ks) : Float64[]

        f_last   = isempty(r.values) ? NaN : Float64(last(r.values))
        gap_last = isempty(r.gaps)   ? NaN : Float64(last(r.gaps))

        # Independent final checks are outside every solver timer.
        verified_value = prob.f(r.x)
        verified_gradient = prob.grad_f(r.x)
        verified_gap = norm(verified_gradient)

        per_iter[name]["seed_$(s)"] = Dict(
            "initial_data_sha256" => input_hash,
            "final_x" => copy(r.x),
            "independently_verified_final_value" => verified_value,
            "independently_verified_final_gap" => verified_gap,
            "instance"     => idx,
            "seed"         => s,
            "values"       => collect(Float64, r.values),   # f per iteration
            "gaps"         => collect(Float64, r.gaps),     # grad norm per iteration
            "times"        => collect(Float64, r.times),    # time per iteration
            "L_ks"         => Lk,                           # L_k per iteration
            "iterations"   => r.k,
            "solved"       => r.solved ? 1 : 0,
            "total_time"   => r.total_time,
            "fstar"        => fstar,
            "final_value"  => f_last,
            "final_gap"    => gap_last,
            "final_subopt" => f_last - fstar,
        )
        per_iter[name]["seed_$(s)"]["final_feasibility_violation"] = 0.0
        merge!(per_iter[name]["seed_$(s)"], ACGDSolvers.result_metadata(r), benchmark_metadata(r))
        if name != "AdaBB"
            per_iter[name]["seed_$(s)"]["accepted_L_max"] = isempty(r.L_ks) ? NaN : maximum(r.L_ks)
            per_iter[name]["seed_$(s)"]["accepted_L_arithmetic_mean"] = isempty(r.L_ks) ? NaN : mean(r.L_ks)
            per_iter[name]["seed_$(s)"]["accepted_L_harmonic_mean"] = harmonic_mean(r.L_ks)
        end


        status = r.solved ? "solved" : "NOT solved"
        line = string(rpad(name, 22), status,
                      "  iters=", r.k,
                      "  time=", round(r.total_time, sigdigits=6), "s",
                      "  f_last=", round(f_last, digits=8),
                      "  gradnorm_last=", round(gap_last, sigdigits=4))
        feasibility_key = haskey(per_iter[name]["seed_$(s)"], "final_feasibility_violation") ?
            "final_feasibility_violation" : "final_aggregate_flow_violation"
        line *= "  feasibility_check=" * string(per_iter[name]["seed_$(s)"][feasibility_key])
        line *= "  status=" * r.status
        if !r.solved
            line *= "  reasons=" * join(unique(String(sample["status"]) for sample in r.raw_repeats), ",")
        end
        println(line)
        push!(report_lines, "instance=$(idx)  seed=$(s)  " * line)
    end
    save_instance_checkpoint(joinpath(instance_dir, "seed_$(s).mat"), Dict(
        "seed" => s, "method_order" => method_order,
        "problem" => snapshot, "initial_data_sha256" => input_hash,
        "runs" => Dict(matvarname(name) => per_iter[name]["seed_$(s)"] for name in method_order));
        writer=matwrite)
    flush(stdout)
end

# --------------------------------------------------------------------
# 3b. Save per-iteration histories: one .mat per method
# --------------------------------------------------------------------
for name in method_order
    fname = joinpath(output_dir,
                     "$(problem_name)_iterations_$(matvarname(name)).mat")
    matwrite(fname, Dict(
        "problem" => problem_name,
        "method"  => name,
        "seeds"   => collect(seeds),
        "runs"    => per_iter[name],
    ))
    println("Saved per-iteration history: ", basename(fname))
end

# plain-text report (per instance/seed: status, iters, time, last f, last grad norm)
open(joinpath(output_dir, "$(problem_name)_report.txt"), "w") do io
    println(io, "Problem: ", problem_name, "   seeds: ", collect(seeds))
    for l in report_lines
        println(io, l)
    end
end

# --------------------------------------------------------------------
# 4. Summary table + save raw results
# --------------------------------------------------------------------
println("\n# Summary ($(length(seeds))-seed sweep) #")
println(rpad("Method", 22), rpad("Solved", 10), rpad("Mean time (s)", 16), "Median time (s)")
for name in method_order
    t = filter(isfinite, all_times[name])
    n_solved = length(t)
    mean_t = n_solved > 0 ? round(mean(t), sigdigits=6) : NaN
    med_t  = n_solved > 0 ? round(median(t), sigdigits=6) : NaN
    println(rpad(name, 22), rpad("$n_solved/$(length(seeds))", 10), rpad(string(mean_t), 16), string(med_t))

    fv  = [per_iter[name]["seed_$(s)"]["final_value"] for s in seeds]
    gpv = [per_iter[name]["seed_$(s)"]["final_gap"]   for s in seeds]
    println("    last f: mean=", round(mean(fv), digits=6),
            "   last grad norm: mean=", round(mean(gpv), sigdigits=4))
end

matwrite(joinpath(output_dir, "benchmark_all_methods.mat"), Dict(
    "seeds"  => collect(seeds),
    "times"  => Dict(matvarname(name) => all_times[name]  for name in method_order),
    "solved" => Dict(matvarname(name) => all_solved[name] for name in method_order),
))

# --------------------------------------------------------------------
# 5. Dolan-Moré performance profile
# --------------------------------------------------------------------
T = hcat([all_times[name] for name in method_order]...)   # n_problems x n_solvers

style = (
    titlefont = (20, "serif"), guidefont = (20, "serif"),
    tickfont  = (20, "serif"), legendfont = (16, "serif"),
    grid = false, framestyle = :box, margin = 5mm,
    size = (900, 700), linewidth = 3,
)

profile = finite_performance_profile(T; logscale=false)
profile_summary = profile_statistics(T)
speed_comparisons = Dict{String,Any}()
for reference_name in ("Adaptive constant", "Adaptive adjustable")
    reference = findfirst(==(reference_name), method_order)
    paired = paired_speed_ratios(T; reference=reference)
    speed_comparisons[matvarname(reference_name)] = Dict(
        matvarname(name) => Dict(string(key) => value for (key, value) in pairs(paired[j]))
        for (j, name) in enumerate(method_order))
end
matwrite(joinpath(output_dir, "profile_statistics.mat"), Dict(
    "methods" => method_order, "ratios" => profile.ratios,
    "thresholds" => profile.thresholds, "fractions" => profile.fractions,
    "success_rates" => profile.success_rates,
    "finite_ratio_coverage" => profile.finite_ratio_coverage,
    "all_failed_instances" => profile.all_failed_instances,
    "max_ratio" => profile.max_ratio,
    "paired_speed_comparisons" => speed_comparisons,
))
p = Plots.plot(profile.thresholds, profile.fractions;
        labels=permutedims(method_order), seriestype=:steppost,
        xscale=:log2, xlims=(1.0, max(2.0, profile.max_ratio)),
        legend=:bottomright, xlabel="Performance ratio (logarithmic axis)",
        ylabel="Fraction of all instances solved within ratio", style...)

plot!(p,
    xtickfont = Plots.font(20, "serif"),
    ytickfont = Plots.font(20, "serif"),
    guidefont = Plots.font(20, "serif"),
    ylims  = (-0.03, 1.03),
    xrotation = 20,
)

# --------------------------------------------------------------------
# Color mapping keyed by METHOD NAME, not by position in `method_order`.
# A positional color list silently shifts every color whenever a method is
# added/dropped/reordered (e.g. "Pure backtracking" rendering black instead
# of red), breaking visual consistency with the companion matrix_balancing
# figure. This dict is stable for any subset or ordering.
# --------------------------------------------------------------------
method_colors = Dict(
    "Adaptive constant"   => :blue,
    "Adaptive adjustable" => :green,
    "Pure backtracking"   => :red,
    "Auto-conditioned"    => :black,
    "Short-step"          => :orange,
    "Open-loop"           => :purple,
    "AdaBB"               => :olive,
)
for (i, series) in enumerate(p.series_list)
    series[:linecolor] = method_colors[method_order[i]]
end

Plots.savefig(p, joinpath(output_dir,
    "Logistic_Regression_with_Regularization_performance_profile_all_methods.eps"))
Plots.savefig(p, joinpath(output_dir,
    "Logistic_Regression_with_Regularization_performance_profile_all_methods.png"))
println("\n✅ Saved performance profile")
display(p)

end # ACGD_RUN_BENCHMARKS
