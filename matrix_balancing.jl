# ==========================================================================
# matrix_balancing  Frank-Wolfe: all 6 methods x N seeds -> combined performance profile
# ==========================================================================

using LinearAlgebra
using Statistics
using Random
using Plots
using Measures
using PyPlot
using MAT
using Zygote
using SparseArrays

pyplot()
PyPlot.PyCall.pyimport("warnings").filterwarnings("ignore")

# --- output goes into <run dir>/matrix_balancing/ -------------------
problem_name = "matrix_balancing"
output_dir = joinpath(get(ENV, "ACGD_OUTPUT_ROOT", joinpath(@__DIR__, "corrected_results")), problem_name)
mkpath(output_dir)
instance_dir = joinpath(output_dir, "instances")
mkpath(instance_dir)

# sanitize method names into valid .mat struct field names
matvarname(s) = replace(s, r"[^A-Za-z0-9_]" => "_")

# --------------------------------------------------------------------
# 1. Problem generator — one seed -> one independent matrix_balancing  instance
# --------------------------------------------------------------------
function generate_problem(seed; n=100, a=5, l=1.0, u=10.0)
    Random.seed!(seed)
    A_sparse = sprand(n, n, a/n)
    A = abs.(Matrix(A_sparse) + Matrix(A_sparse)') + 0.05 * I

    f(x) = sum(A .* exp.(x .- x'))          # vectorized, see point 5
    grad_f(x) = Zygote.gradient(f, x)[1]

    function lmo(g)
        return [g[i] > 0 ? l : u for i in 1:n]
    end

    x0 = l .+ (u - l) .* rand(n)

    return (f=f, grad_f=grad_f, lmo=lmo, x0=x0, A=A, l=l, u=u, seed=seed, optimal_value=sum(A), n=n, a=a)
end

function estimate_L(A; τ = 9.0)
    off = A - Diagonal(diag(A))
    Lap = Diagonal(vec(sum(off, dims=2))) - off
    M = (exp(τ) + exp(-τ)) * Lap + Diagonal(2 .* diag(A))
    return eigmax(Symmetric(M))
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
        method=:adaptive, constrained=true, stop_rule=:either,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, gamma=gamma, kwargs...)
    return r
end

function conditional_gradient_adjustable_scaling(f, grad_f, lmo, x0, gamma;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:adjustable, constrained=true, stop_rule=:either,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, gamma=gamma, kwargs...)
    return r
end

function conditional_gradient_auto_conditioned(f, grad_f, lmo, x0;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:auto, constrained=true, stop_rule=:either,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, kwargs...)
    return r
end

function conditional_gradient_pure_backtracking(f, grad_f, lmo, x0;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:backtracking, constrained=true, stop_rule=:either,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, kwargs...)
    return r
end

function conditional_gradient_open(f, grad_f, lmo, x0;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:open, constrained=true, stop_rule=:either,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, kwargs...)
    return r
end

function conditional_gradient_short_step(f, grad_f, lmo, x0, L;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:short, constrained=true, stop_rule=:either,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, L=L, kwargs...)
    return r
end

if get(ENV, "ACGD_RUN_BENCHMARKS", "true") == "true"

# --------------------------------------------------------------------
# 3. Run all methods on each of the N seeded problems
# --------------------------------------------------------------------
seeds  = 1:parse(Int, get(ENV, "ACGD_SEED_COUNT", "100"))
isempty(seeds) && throw(ArgumentError("ACGD_SEED_COUNT must be at least one"))

gamma0 = 1/4

method_order = ["Adaptive constant", "Adaptive adjustable", "Pure backtracking", "Auto-conditioned",
                "Open-loop",
                #"Short-step"
                ]

problem_parameters = (n=100, a=5, l=1.0, u=10.0)
warmups = parse(Int, get(ENV, "ACGD_WARMUPS", "2"))
repeats = benchmark_repeats(length(method_order))
order_seed = 20261021

all_times  = Dict(name => fill(Inf, length(seeds)) for name in method_order)
all_solved = Dict(name => falses(length(seeds)) for name in method_order)

# FIX #1: use Dict{String,Any} (base Julia), NOT Dictionaries.jl's
# `Dictionary`, which was never `using`-imported and would throw
# UndefVarError at runtime.
per_iter     = Dict(name => Dict{String,Any}() for name in method_order)
report_lines = String[]                      # text report accumulator

write_run_manifest(joinpath(output_dir, "run_manifest.toml"), Dict(
    "problem" => problem_name, "script" => basename(@__FILE__),
    "stop_rule" => "either", "epsilon" => 1e-5,
    "max_iter_cap" => parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")), "max_time" => Inf,
    "open_schedule" => "2/(k+2)",
    "gamma" => gamma0, "beta" => 2.0, "eta" => 0.9, "block_size" => 10,
    "probe_seed" => 23, "delta" => 1e-10,
    "generator_rng" => "original Julia default RNG explicitly reseeded per instance; see pinned Julia version",
    "method_order" => method_order,
    "execution_base_order" => ["Open-loop", "Adaptive constant", "Adaptive adjustable", "Auto-conditioned", "Pure backtracking"],
    "problem_parameters" => Dict(string(k) => v for (k, v) in pairs(problem_parameters)),
    "execution_order" => "balanced Williams design; actual schedules saved per instance",
    "order_mode" => "balanced", "order_seed" => order_seed,
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
    snapshot = problem_snapshot(prob)
    input_hash = snapshot_sha256(snapshot)
    # L_est = estimate_L(prob.A)   # only needed if Short-step is re-enabled
    fstar = prob.optimal_value   # true optimal objective value (sum(A)), NOT 0

    calls = [
        #"Short-step"          => () -> conditional_gradient_short_step(prob.f, prob.grad_f, prob.lmo, prob.x0, L_est; fstar=fstar),
        "Open-loop"           => () -> conditional_gradient_open(prob.f, prob.grad_f, prob.lmo, prob.x0; fstar=fstar),
        "Adaptive constant"   => () -> conditional_gradient_adaptive(prob.f, prob.grad_f, prob.lmo, prob.x0, gamma0; fstar=fstar),
        "Adaptive adjustable" => () -> conditional_gradient_adjustable_scaling(prob.f, prob.grad_f, prob.lmo, prob.x0, gamma0; fstar=fstar),
        "Auto-conditioned"    => () -> conditional_gradient_auto_conditioned(prob.f, prob.grad_f, prob.lmo, prob.x0; fstar=fstar),
        "Pure backtracking"   => () -> conditional_gradient_pure_backtracking(prob.f, prob.grad_f, prob.lmo, prob.x0; fstar=fstar),
    ]
    runs = benchmark_methods(calls; instance_index=idx,
        warmups=warmups, repeats=repeats, external_timing=true,
        gc_before_call=true, order_mode=:balanced, order_seed=order_seed)

    for name in method_order
        r = runs[name]
        all_solved[name][idx] = r.solved
        all_times[name][idx]  = r.solved ? r.total_time : Inf

        # Open-loop / Short-step have no L_k sequence -> store empty vector
        Lk = hasproperty(r, :L_ks) ? collect(Float64, r.L_ks) : Float64[]

        f_last   = isempty(r.values) ? NaN : Float64(last(r.values))
        gap_last = isempty(r.gaps)   ? NaN : Float64(last(r.gaps))

        # Independent final checks are outside every solver timer.
        verified_value = prob.f(r.x)
        verified_gradient = prob.grad_f(r.x)
        verified_gap = dot(verified_gradient, r.x - prob.lmo(verified_gradient))

        per_iter[name]["seed_$(s)"] = Dict(
            "initial_data_sha256" => input_hash,
            "final_x" => copy(r.x),
            "independently_verified_final_value" => verified_value,
            "independently_verified_final_gap" => verified_gap,
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
        per_iter[name]["seed_$(s)"]["final_feasibility_violation"] = max(0.0, prob.l - minimum(r.x), maximum(r.x) - prob.u)
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
                      "  gap_last=", round(gap_last, sigdigits=4))
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
# FIX #3: color mapping is now keyed by METHOD NAME, not by position in
# `method_order`. Since "Short-step" is dropped from this benchmark, a
# purely positional color list would silently shift every color after
# "Adaptive adjustable" — e.g. "Pure backtracking" would render black
# instead of red, and "Auto-conditioned" red instead of black — breaking
# visual consistency with companion figures (e.g. the Lasso benchmark)
# that use the full 6-method roster. This dict is stable regardless of
# which subset of methods is included or their order.
# --------------------------------------------------------------------
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

Plots.savefig(p, joinpath(output_dir,
    "Matrix_Balancing_performance_profile_all_methods.eps"))
Plots.savefig(p, joinpath(output_dir,
    "Matrix_Balancing_performance_profile_all_methods.png"))
println("\n✅ Saved performance profile")
display(p)

end # ACGD_RUN_BENCHMARKS
