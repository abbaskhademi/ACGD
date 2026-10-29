# ==========================================================================
# Non-Convex StQP over the Simplex: Frank-Wolfe (6 methods) x N seeds
# ==========================================================================

using LinearAlgebra, Statistics, Random
using Plots, Measures, PyPlot, MAT

pyplot()
PyPlot.PyCall.pyimport("warnings").filterwarnings("ignore")

problem_name = "NonconvexQP_Simplex"
output_dir = joinpath(get(ENV, "ACGD_OUTPUT_ROOT", joinpath(@__DIR__, "corrected_results")), problem_name)
mkpath(output_dir)
instance_dir = joinpath(output_dir, "instances")
mkpath(instance_dir)

matvarname(s) = replace(s, r"[^A-Za-z0-9_]" => "_")

# --------------------------------------------------------------------
# 1. Problem generator — StQP instance k (deterministic myrandom stream)
# --------------------------------------------------------------------
function myrandom(r, a, b)
    r = mod(r * 41475557, 1.0)
    num = a + r * (b - a)
    return r, num
end

function generate_problem(igen; n=899, dens=0.75, dvert=2.0, x0_mode=:myrandom)
    N = n + 1
    myseed = Float64(igen)
    r = (4 * myseed + 1) / 16384 / 16384

    Fc = zeros(Float64, N, N)
    Fl = zeros(Float64, N, N)

    for i in 1:n
        for j in (i+1):N
            r, num = myrandom(r, 0.0, 1.0)
            if num < dens
                r, num = myrandom(r, 0.0, 10.0)
            else
                r, num = myrandom(r, -10.0, 0.0)
            end
            Fc[i, j] = num
            Fc[j, i] = num
        end
    end

    for i in 1:N
        r, num = myrandom(r, 0.0, dvert)
        Fl[i, i] = num
    end
    for i in 1:n
        for j in (i+1):N
            Fl[i, j] = 0.5 * (Fl[i, i] + Fl[j, j])
            Fl[j, i] = Fl[i, j]
        end
    end

    Q = Fl - Fc

    f(x)      = dot(x, Q * x)
    grad_f(x) = 2.0 * (Q * x)
    function lmo(g)
        _, i = findmin(g)
        v = zeros(N); v[i] = 1.0
        return v
    end

    z = zeros(N)
    if x0_mode == :myrandom
        for i in 1:N
            r, num = myrandom(r, 0.0, 1.0)
            z[i] = num
        end
    else
        Random.seed!(igen)
        z .= rand(N)
    end
    #x0 = z / sum(z)
    x0 = zeros(N); x0[1] = 1.0

    # StQP global optimum is unknown in general (NP-hard) -> NaN, same as Sigmoid template
    optimal_value = NaN

    return (f=f, grad_f=grad_f, lmo=lmo, x0=x0, Q=Q, n=n, N=N,
            dens=dens, dvert=dvert, optimal_value=optimal_value, seed=igen)
end

function estimate_L(Q)
    evals = eigvals(Symmetric(Q))
    return 2.0 * maximum(abs.(evals))
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
        epsilon=1e-3, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=NaN, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:adaptive, constrained=true, stop_rule=:stationarity,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, gamma=gamma, kwargs...)
    return r
end

function conditional_gradient_adjustable_scaling(f, grad_f, lmo, x0, gamma;
        epsilon=1e-3, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=NaN, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:adjustable, constrained=true, stop_rule=:stationarity,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, gamma=gamma, kwargs...)
    return r
end

function conditional_gradient_auto_conditioned(f, grad_f, lmo, x0;
        epsilon=1e-3, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=NaN, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:auto, constrained=true, stop_rule=:stationarity,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, kwargs...)
    return r
end

function conditional_gradient_pure_backtracking(f, grad_f, lmo, x0;
        epsilon=1e-3, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=NaN, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:backtracking, constrained=true, stop_rule=:stationarity,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, kwargs...)
    return r
end

function conditional_gradient_open(f, grad_f, lmo, x0;
        epsilon=1e-3, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=NaN, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:open, open_schedule=:nonconvex, constrained=true, stop_rule=:stationarity,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, kwargs...)
    return r
end

function conditional_gradient_short_step(f, grad_f, lmo, x0, L;
        epsilon=1e-3, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=NaN, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:short, constrained=true, stop_rule=:stationarity,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, L=L, kwargs...)
    return r
end

if get(ENV, "ACGD_RUN_BENCHMARKS", "true") == "true"

# --------------------------------------------------------------------
# 3. Run all 6 methods on each of the N seeded StQP instances
#    (SAME output structure as the Sigmoid/Gisette template)
# --------------------------------------------------------------------
println("\n================ Simplex Quadratic Optimization Problem ================")
seeds  = 1:parse(Int, get(ENV, "ACGD_SEED_COUNT", "100"))
isempty(seeds) && throw(ArgumentError("ACGD_SEED_COUNT must be at least one"))
gamma0 = 1/4
n_dim  = 699          # -> N = 700
dens_val  = 0.75
dvert_val = 2.0

method_order = ["Adaptive constant", "Adaptive adjustable", "Pure backtracking",
                 "Auto-conditioned", "Short-step", "Open-loop"]

problem_parameters = (n=n_dim, dens=dens_val, dvert=dvert_val)
warmups = parse(Int, get(ENV, "ACGD_WARMUPS", "1"))
repeats = parse(Int, get(ENV, "ACGD_REPEATS", "1"))
order_seed = 20261021

all_times  = Dict(name => fill(Inf, length(seeds)) for name in method_order)
all_solved = Dict(name => falses(length(seeds)) for name in method_order)

per_iter     = Dict(name => Dict{String,Any}() for name in method_order)
report_lines = String[]

write_run_manifest(joinpath(output_dir, "run_manifest.toml"), Dict(
    "problem" => problem_name, "script" => basename(@__FILE__),
    "stop_rule" => "stationarity", "epsilon" => 1e-3,
    "max_iter_cap" => parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")), "max_time" => Inf,
    "open_schedule" => "1/sqrt(k+1)",
    "gamma" => gamma0, "beta" => 2.0, "eta" => 0.9, "block_size" => 10,
    "probe_seed" => 23, "delta" => 1e-10,
    "generator_rng" => "deterministic myrandom stream initialized by instance seed; x0 is the first simplex vertex",
    "method_order" => method_order,
    "execution_base_order" => ["Open-loop", "Short-step", "Adaptive constant", "Adaptive adjustable", "Auto-conditioned", "Pure backtracking"],
    "dimension" => n_dim + 1, "initial_point" => "first simplex vertex",
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
    L_est = estimate_L(prob.Q)
    snapshot = problem_snapshot(prob)
    snapshot["L_estimate"] = L_est
    input_hash = snapshot_sha256(snapshot)
    fstar = prob.optimal_value   # NaN — global StQP optimum unknown

    calls = [
        "Open-loop"           => () -> conditional_gradient_open(prob.f, prob.grad_f, prob.lmo, prob.x0; fstar=fstar),
        "Short-step"          => () -> conditional_gradient_short_step(prob.f, prob.grad_f, prob.lmo, prob.x0, L_est; fstar=fstar),
        "Adaptive constant"   => () -> conditional_gradient_adaptive(prob.f, prob.grad_f, prob.lmo, prob.x0, gamma0; fstar=fstar),
        "Adaptive adjustable" => () -> conditional_gradient_adjustable_scaling(prob.f, prob.grad_f, prob.lmo, prob.x0, gamma0; fstar=fstar),
        "Auto-conditioned"    => () -> conditional_gradient_auto_conditioned(prob.f, prob.grad_f, prob.lmo, prob.x0; fstar=fstar),
        "Pure backtracking"   => () -> conditional_gradient_pure_backtracking(prob.f, prob.grad_f, prob.lmo, prob.x0; fstar=fstar),
    ]
    runs = benchmark_methods(calls; instance_index=idx,
        warmups=warmups, repeats=repeats, external_timing=true,
        gc_before_call=true, order_mode=:rotate, order_seed=order_seed)

    for name in method_order
        r = runs[name]
        all_solved[name][idx] = r.solved
        all_times[name][idx]  = r.solved ? r.total_time : Inf

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
            "values"       => collect(Float64, r.values),
            "gaps"         => collect(Float64, r.gaps),
            "times"        => collect(Float64, r.times),
            "L_ks"         => Lk,
            "iterations"   => r.k,
            "solved"       => r.solved ? 1 : 0,
            "total_time"   => r.total_time,
            "fstar"        => fstar,
            "final_value"  => f_last,
            "final_gap"    => gap_last,
            "final_subopt" => f_last - fstar,
        )
        per_iter[name]["seed_$(s)"]["final_feasibility_violation"] = max(0.0, -minimum(r.x), abs(sum(r.x) - 1.0))
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
                      "  f_last=", round(f_last, digits=6),
                      "  gap_last=", round(gap_last, digits=6))
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
    fname = joinpath(output_dir, "$(problem_name)_iterations_$(matvarname(name)).mat")
    matwrite(fname, Dict(
        "problem" => problem_name,
        "method"  => name,
        "seeds"   => collect(seeds),
        "runs"    => per_iter[name],
    ))
    println("✅ Saved per-iteration history: ", basename(fname))
end

# plain-text report
open(joinpath(output_dir, "$(problem_name)_report.txt"), "w") do io
    println(io, "Problem: ", problem_name, "   seeds: ", collect(seeds))
    println(io, "n_dim: $n_dim (N=$(n_dim+1)), dens: $dens_val, dvert: $dvert_val")
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
    "seeds"    => collect(seeds),
    "n_dim"    => n_dim,
    "dens"     => dens_val,
    "dvert"    => dvert_val,
    "times"    => Dict(matvarname(name) => all_times[name]  for name in method_order),
    "solved"   => Dict(matvarname(name) => all_solved[name] for name in method_order),
))

# --------------------------------------------------------------------
# 5. Dolan-Moré performance profile (ratio-based x-axis)
# --------------------------------------------------------------------
T = hcat([all_times[name] for name in method_order]...)

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

# Fixed name -> color mapping (SAME palette as Sigmoid template, keyed by name
# so it's robust to any reordering of method_order)
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

Plots.savefig(p, joinpath(output_dir, "$(problem_name)_performance_profile_all_methods.eps"))
Plots.savefig(p, joinpath(output_dir, "$(problem_name)_performance_profile_all_methods.png"))
println("✅ Saved: $(problem_name)_performance_profile_all_methods.eps in ", output_dir)
display(p)

end # ACGD_RUN_BENCHMARKS
