# ==========================================================================
# Lasso Frank-Wolfe: all 6 methods x N seeds -> combined performance profile
# ==========================================================================
using LinearAlgebra
using Statistics
using Random
using Plots
using Measures
using PyPlot
using MAT

pyplot()
PyPlot.PyCall.pyimport("warnings").filterwarnings("ignore")

problem_name = "lasso"
output_dir = joinpath(get(ENV, "ACGD_OUTPUT_ROOT", joinpath(@__DIR__, "corrected_results")), problem_name)
mkpath(output_dir)
instance_dir = joinpath(output_dir, "instances")
mkpath(instance_dir)

#output_dir = joinpath(pwd(), problem_name)
#mkpath(output_dir)  # Creates the folder if it doesn't already exist

# sanitize method names into valid .mat struct field names (defined early
# because we now write .mat files inside the seed loop's post-processing)
matvarname(s) = replace(s, r"[^A-Za-z0-9_]" => "_")

# --------------------------------------------------------------------
# 1. Problem generator — one seed -> one independent Lasso instance
# --------------------------------------------------------------------
function generate_problem(seed; m=200, n=1000, tau=10)
    Random.seed!(seed)

    x_star = zeros(n)
    indices = randperm(n)[1:tau]
    #x_star[indices] = randn(tau)
    raw = randn(tau)
    l1_raw = sum(abs.(raw))
    x_star[indices] = raw .* (0.9 * tau / l1_raw)
    A = randn(m, n)
    b = A * x_star

    f(z)      = sum((b - A * z) .^ 2)
    grad_f(z) = 2 * A' * (A * z - b)
    x0 =  zeros(n)

    function lmo(g)
        i = argmax(abs.(g))
        return -tau * sign(g[i]) * [i == j ? 1.0 : 0.0 for j in 1:n]
    end

    # x_star is strictly feasible (0.9*tau < tau) and interpolates b, so the
    # constrained optimum is exactly 0.
    return (f=f, grad_f=grad_f, lmo=lmo, x0=x0, A=A, x_star=x_star,
            seed=seed, tau=tau, optimal_value=0.0)
end


function estimate_L(A)
    L = 2 * maximum(eigvals(A' * A))
    #println("L = ", round(L, digits=4))
    return L
end

# Shared solver implementation: full solver elapsed time includes initialization,
# curvature probes, stopping checks and in-memory history. Data/I/O stay outside.
if !isdefined(@__MODULE__, :ACGDSolvers)
    include(joinpath(@__DIR__, "src", "ACGDSolvers.jl"))
end
if !isdefined(@__MODULE__, :ACGDBenchmarkSupport)
    include(joinpath(@__DIR__, "scripts", "benchmark_support.jl"))
end
using .ACGDBenchmarkSupport

function conditional_gradient_adaptive(f, grad_f, lmo, x0, gamma;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:adaptive, constrained=true, stop_rule=:objective,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, gamma=gamma, kwargs...)
    return r
end

function conditional_gradient_adjustable_scaling(f, grad_f, lmo, x0, gamma;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:adjustable, constrained=true, stop_rule=:objective,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, gamma=gamma, kwargs...)
    return r
end

function conditional_gradient_auto_conditioned(f, grad_f, lmo, x0;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:auto, constrained=true, stop_rule=:objective,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, kwargs...)
    return r
end

function conditional_gradient_pure_backtracking(f, grad_f, lmo, x0;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:backtracking, constrained=true, stop_rule=:objective,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, kwargs...)
    return r
end

function conditional_gradient_open(f, grad_f, lmo, x0;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:open, constrained=true, stop_rule=:objective,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, kwargs...)
    return r
end

function conditional_gradient_short_step(f, grad_f, lmo, x0, L;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:short, constrained=true, stop_rule=:objective,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, L=L, kwargs...)
    return r
end

if get(ENV, "ACGD_RUN_BENCHMARKS", "true") == "true"

# --------------------------------------------------------------------
# 3. Run all 6 methods on each of the N seeded problems
# --------------------------------------------------------------------
seeds  = 1:parse(Int, get(ENV, "ACGD_SEED_COUNT", "100"))
isempty(seeds) && throw(ArgumentError("ACGD_SEED_COUNT must be at least one"))
gamma0 = 1/4

method_order = ["Adaptive constant", "Adaptive adjustable", "Pure backtracking", "Auto-conditioned",
                "Short-step", "Open-loop"]

all_times  = Dict(name => fill(Inf, length(seeds)) for name in method_order)
all_solved = Dict(name => falses(length(seeds)) for name in method_order)

# per-method, per-seed iteration histories (values / gaps / times / L_k)
per_iter     = Dict(name => Dict{String,Any}() for name in method_order)
report_lines = String[]                      # text report accumulator

write_run_manifest(joinpath(output_dir, "run_manifest.toml"), Dict(
    "problem" => problem_name, "script" => basename(@__FILE__),
    "stop_rule" => "objective", "epsilon" => 1e-5,
    "max_iter_cap" => parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")), "max_time" => Inf,
    "gamma" => gamma0, "beta" => 2.0, "eta" => 0.9, "block_size" => 10,
    "probe_seed" => 23, "delta" => 1e-10,
    "generator_rng" => "original Julia default RNG explicitly reseeded per instance; see pinned Julia version",
    "method_order" => method_order,
    "smoothness_preprocessing" => "problem constants computed before solver timing",
    "feasibility_scope" => "representative repeat checked outside solver timing",
    "run_controls" => Dict(name => get(ENV, name, default) for (name, default) in
        [("ACGD_OUTPUT_ROOT", joinpath(@__DIR__, "corrected_results")),
         ("ACGD_SEED_COUNT", "100"), ("ACGD_MAX_ITER", "100000"),
         ("ACGD_TRACE_ITER", "500"), ("ACGD_WARMUPS", "1"), ("ACGD_REPEATS", "3")]),
    "warmups" => parse(Int, get(ENV, "ACGD_WARMUPS", "1")),
    "repeats" => parse(Int, get(ENV, "ACGD_REPEATS", "3")),
    "seeds" => collect(seeds),
    "aggregation" => "median complete solver elapsed time; all repeats must solve",
); root=@__DIR__)

for (idx, s) in enumerate(seeds)
    println("\n================ instance = $idx | seed = $s ================")
    prob  = generate_problem(s)
    L_est = estimate_L(prob.A)
    fstar = prob.optimal_value   # = 0.0 for this generator (attained at x_star)

    calls = [
        "Short-step"          => () -> conditional_gradient_short_step(prob.f, prob.grad_f, prob.lmo, prob.x0, L_est; fstar=fstar),
        "Open-loop"           => () -> conditional_gradient_open(prob.f, prob.grad_f, prob.lmo, prob.x0; fstar=fstar),
        "Adaptive constant"   => () -> conditional_gradient_adaptive(prob.f, prob.grad_f, prob.lmo, prob.x0, gamma0; fstar=fstar),
        "Adaptive adjustable" => () -> conditional_gradient_adjustable_scaling(prob.f, prob.grad_f, prob.lmo, prob.x0, gamma0; fstar=fstar),
        "Auto-conditioned"    => () -> conditional_gradient_auto_conditioned(prob.f, prob.grad_f, prob.lmo, prob.x0; fstar=fstar),
        "Pure backtracking"   => () -> conditional_gradient_pure_backtracking(prob.f, prob.grad_f, prob.lmo, prob.x0; fstar=fstar),

    ]
    runs = benchmark_methods(calls; instance_index=idx,
        warmups = parse(Int, get(ENV, "ACGD_WARMUPS", "1")),
        repeats = parse(Int, get(ENV, "ACGD_REPEATS", "1")))

    for name in method_order
        r = runs[name]
        all_solved[name][idx] = r.solved
        all_times[name][idx]  = r.solved ? r.total_time : Inf

        # Open-loop / Short-step have no L_k sequence -> store empty vector
        Lk = hasproperty(r, :L_ks) ? collect(Float64, r.L_ks) : Float64[]

        f_last   = isempty(r.values) ? NaN : Float64(last(r.values))
        gap_last = isempty(r.gaps)   ? NaN : Float64(last(r.gaps))

        per_iter[name]["seed_$(s)"] = Dict(
            "instance"     => idx,
            "seed"         => s,
            "values"       => collect(Float64, r.values),   # f per iteration
            "gaps"         => collect(Float64, r.gaps),     # FW gap per iteration
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
        per_iter[name]["seed_$(s)"]["final_feasibility_violation"] = max(0.0, norm(r.x, 1) - prob.tau)
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
                      "  f_last=", round(f_last, digits=4),
                      "  gap_last=", round(gap_last, digits=4))
        feasibility_key = haskey(per_iter[name]["seed_$(s)"], "final_feasibility_violation") ?
            "final_feasibility_violation" : "final_aggregate_flow_violation"
        line *= "  feasibility_check=" * string(per_iter[name]["seed_$(s)"][feasibility_key])
        line *= "  status=" * r.status
        println(line)
        push!(report_lines, "instance=$(idx)  seed=$(s)  " * line)
    end
    # Checkpoint one completed instance outside all solver clocks. Write then
    # rename so interruption cannot leave a partially written final checkpoint.
    checkpoint = joinpath(instance_dir, "seed_$(s).mat")
    matwrite(checkpoint * ".tmp", Dict(
        "seed" => s,
        "method_order" => method_order,
        "runs" => Dict(matvarname(name) => per_iter[name]["seed_$(s)"] for name in method_order),
    ))
    mv(checkpoint * ".tmp", checkpoint; force=true)
    println("Checkpoint saved: seed $s ($(idx)/$(length(seeds)))")
    flush(stdout)
end

# --------------------------------------------------------------------
# 3b. Save per-iteration histories: one .mat per method
#     File: <output_dir>/<problem>_iterations_<Method>.mat
#     Struct: runs.seed_<s>.{values, gaps, times, L_ks, final_value, final_gap, ...}
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
    println("✅ Saved per-iteration history: ", basename(fname))
end

# plain-text report (per instance/seed: status, iters, time, last f, last gap)
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
            "   last gap: mean=", round(mean(gpv), sigdigits=4))
end

matwrite(joinpath(output_dir, "benchmark_all_methods.mat"), Dict(
    "seeds"  => collect(seeds),
    "times"  => Dict(matvarname(name) => all_times[name]  for name in method_order),
    "solved" => Dict(matvarname(name) => all_solved[name] for name in method_order),
))

# --------------------------------------------------------------------
# 5. Dolan-Moré performance profile (ratio-based x-axis)
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

# Fixed name -> color mapping (matches the Lasso figure's assignment)
# rather than a positional list, so each method keeps the same color
# across every subsection's figure even when a given problem only plots
# a subset of the six methods.
method_colors = Dict(
    "Adaptive constant"   => :blue,
    "Adaptive adjustable" => :green,
    "Pure backtracking"   => :red,
    "Auto-conditioned"    => :black,
    "Short-step"          => :orange,
    "Open-loop"           => :purple,
)
for (i, series) in enumerate(p.series_list)
    series[:linecolor] = method_colors[method_order[i]]
end

Plots.savefig(p, joinpath(output_dir, "lasso_problem_performance_profile_all_methods.eps"))
println("✅ Saved: lasso_problem_performance_profile_all_methods.eps in ", output_dir)
display(p)

end # ACGD_RUN_BENCHMARKS
