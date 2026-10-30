# ==========================================================================
# Matrix Factorization
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

problem_name = "matrix_factorization"
output_dir = joinpath(get(ENV, "ACGD_OUTPUT_ROOT", joinpath(@__DIR__, "corrected_results")), problem_name)
mkpath(output_dir)
instance_dir = joinpath(output_dir, "instances")
mkpath(instance_dir)


# MAT.jl struct field names must be valid identifiers -- sanitize just for
# the .mat export. method_order, console output, and the profile plot
# below still use the original names. (Defined early: now also needed by
# the per-iteration .mat writer in section 4b.)
matvarname(s) = replace(s, r"[^A-Za-z0-9_]" => "_")

# --------------------------------------------------------------------
# 1. Shared helper functions (do not depend on seed)
# --------------------------------------------------------------------
fro_inner(A, B) = dot(A, B)
fro_norm2_concat(A, B) = norm(A)^2 + norm(B)^2
fro_norm_concat(A, B) = sqrt(fro_norm2_concat(A, B))

function lmo_spectral(G, tau)
    F = svd(G)
    return -tau * F.U * F.Vt
end

function project_spectral(X, tau)
    F = svd(X)
    return F.U * Diagonal(min.(F.S, tau)) * F.Vt
end

# --------------------------------------------------------------------
# 2. Problem generator -- one seed -> one independent Matrix
#    Factorization instance
# --------------------------------------------------------------------
function generate_problem(seed; m=400, n=300, r=20, sigma_max=10.0, sigma_min=0.01)
    Random.seed!(seed)

    sigmas = exp.(range(log(sigma_max), log(sigma_min), length=r)) |> collect
    Sigma_half = Diagonal(sqrt.(sigmas))

    U_bar, _ = qr(randn(m, r)); U_bar = Matrix(U_bar)[:, 1:r]
    V_bar, _ = qr(randn(n, r)); V_bar = Matrix(V_bar)[:, 1:r]

    U_star = U_bar * Sigma_half
    V_star = V_bar * Sigma_half
    Y = U_star * V_star'

    # ceil() => tau_i >= opnorm of the planted factor, so (U_star, V_star)
    # is feasible and the global optimum is exactly 0 (see header note).
    tau1 = ceil(Int, opnorm(U_star))
    tau2 = ceil(Int, opnorm(V_star))

    f(U, V) = 0.5 * norm(U * V' - Y)^2
    function grad_f(U, V)
        R = U * V' - Y
        return R * V, R' * U
    end
    lmo(grad_U, grad_V) = (lmo_spectral(grad_U, tau1), lmo_spectral(grad_V, tau2))

    # Initial point: at constraint boundary, far from optimum.
    # Varies with `seed` (see note above the script about the double
    # Random.seed!(23) in the original single-run version).
    U0 = project_spectral(10.0 * randn(m, r), tau1)
    V0 = project_spectral(10.0 * randn(n, r), tau2)

    # The cross Hessian contains two factor-product terms.
    L_global = max(tau1^2, tau2^2) + opnorm(Y) + 2 * tau1 * tau2

    return (f=f, grad_f=grad_f, lmo=lmo, U0=U0, V0=V0,
            tau1=tau1, tau2=tau2, Y=Y, L_global=L_global,
            U_star=U_star, V_star=V_star, optimal_value=0.0, seed=seed,
            m=m, n=n, rank=r, sigma_max=sigma_max, sigma_min=sigma_min)
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

function conditional_gradient_adaptive_matfac(f, grad_f, lmo, U0, V0, gamma;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    r = ACGDSolvers.solve(x -> f(x...), x -> grad_f(x...), g -> lmo(g...), (U0, V0);
        method=:adaptive, constrained=true, stop_rule=:either,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, gamma=gamma, kwargs...)
    return merge(r, (U=r.x[1], V=r.x[2]))
end

function conditional_gradient_adjustable_scaling_matfac(f, grad_f, lmo, U0, V0, gamma;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    r = ACGDSolvers.solve(x -> f(x...), x -> grad_f(x...), g -> lmo(g...), (U0, V0);
        method=:adjustable, constrained=true, stop_rule=:either,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, gamma=gamma, kwargs...)
    return merge(r, (U=r.x[1], V=r.x[2]))
end

function conditional_gradient_auto_conditioned_matfac(f, grad_f, lmo, U0, V0;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    r = ACGDSolvers.solve(x -> f(x...), x -> grad_f(x...), g -> lmo(g...), (U0, V0);
        method=:auto, constrained=true, stop_rule=:either,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, kwargs...)
    return merge(r, (U=r.x[1], V=r.x[2]))
end

function conditional_gradient_pure_backtracking_matfac(f, grad_f, lmo, U0, V0;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    r = ACGDSolvers.solve(x -> f(x...), x -> grad_f(x...), g -> lmo(g...), (U0, V0);
        method=:backtracking, constrained=true, stop_rule=:either,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, kwargs...)
    return merge(r, (U=r.x[1], V=r.x[2]))
end

function conditional_gradient_open_matfac(f, grad_f, lmo, U0, V0;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    r = ACGDSolvers.solve(x -> f(x...), x -> grad_f(x...), g -> lmo(g...), (U0, V0);
        method=:open, open_schedule=:nonconvex, constrained=true, stop_rule=:either,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, kwargs...)
    return merge(r, (U=r.x[1], V=r.x[2]))
end

function conditional_gradient_short_step_matfac(f, grad_f, lmo, U0, V0, L;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    r = ACGDSolvers.solve(x -> f(x...), x -> grad_f(x...), g -> lmo(g...), (U0, V0);
        method=:short, constrained=true, stop_rule=:either,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, L=L, kwargs...)
    return merge(r, (U=r.x[1], V=r.x[2]))
end

if get(ENV, "ACGD_RUN_BENCHMARKS", "true") == "true"

# --------------------------------------------------------------------
# 4. Run all 6 methods on each of the N seeded problems
# --------------------------------------------------------------------
println("\n================ Matrix Factorization Problem ================")
seeds  = 1:parse(Int, get(ENV, "ACGD_SEED_COUNT", "100"))   # consider trying seeds = 1:5 first -- SVDs each iteration
isempty(seeds) && throw(ArgumentError("ACGD_SEED_COUNT must be at least one"))
                # (400x20, 300x20) make this considerably slower than Lasso
gamma0 = 1/4

method_order = ["Adaptive constant", "Adaptive adjustable", "Pure backtracking", "Auto-conditioned",
                 "Short-step", "Open-loop"]

problem_parameters = (m=400, n=300, r=20, sigma_max=10.0, sigma_min=0.01)
warmups = parse(Int, get(ENV, "ACGD_WARMUPS", "2"))
repeats = benchmark_repeats(length(method_order))
order_seed = 20261021

all_times  = Dict(name => fill(Inf, length(seeds)) for name in method_order)
all_solved = Dict(name => falses(length(seeds)) for name in method_order)

# per-method, per-seed iteration histories (values / gaps / times / L_k)
per_iter     = Dict(name => Dict{String,Any}() for name in method_order)
report_lines = String[]                      # text report accumulator

write_run_manifest(joinpath(output_dir, "run_manifest.toml"), Dict(
    "problem" => problem_name, "script" => basename(@__FILE__),
    "stop_rule" => "either", "epsilon" => 1e-5,
    "max_iter_cap" => parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")), "max_time" => Inf,
    "open_schedule" => "1/sqrt(k+1)",
    "gamma" => gamma0, "beta" => 2.0, "eta" => 0.9, "block_size" => 10,
    "probe_seed" => 23, "delta" => 1e-10,
    "generator_rng" => "original Julia default RNG explicitly reseeded per instance; see pinned Julia version",
    "method_order" => method_order,
    "execution_base_order" => ["Short-step", "Open-loop", "Adaptive constant", "Adaptive adjustable", "Auto-conditioned", "Pure backtracking"],
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
    fstar = prob.optimal_value   # = 0.0 (attained at the feasible planted factors)

    calls = [
        "Short-step"          => () -> conditional_gradient_short_step_matfac(prob.f, prob.grad_f, prob.lmo, prob.U0, prob.V0, prob.L_global; fstar=fstar),
        "Open-loop"           => () -> conditional_gradient_open_matfac(prob.f, prob.grad_f, prob.lmo, prob.U0, prob.V0; fstar=fstar),
        "Adaptive constant"   => () -> conditional_gradient_adaptive_matfac(prob.f, prob.grad_f, prob.lmo, prob.U0, prob.V0, gamma0; fstar=fstar),
        "Adaptive adjustable" => () -> conditional_gradient_adjustable_scaling_matfac(prob.f, prob.grad_f, prob.lmo, prob.U0, prob.V0, gamma0; fstar=fstar),
        "Auto-conditioned"    => () -> conditional_gradient_auto_conditioned_matfac(prob.f, prob.grad_f, prob.lmo, prob.U0, prob.V0; fstar=fstar),
        "Pure backtracking"   => () -> conditional_gradient_pure_backtracking_matfac(prob.f, prob.grad_f, prob.lmo, prob.U0, prob.V0; fstar=fstar),
    ]
    runs = benchmark_methods(calls; instance_index=idx,
        warmups=warmups, repeats=repeats, external_timing=true,
        gc_before_call=true, order_mode=:balanced, order_seed=order_seed)

    for name in method_order
        r = runs[name]
        all_solved[name][idx] = r.solved
        all_times[name][idx]  = r.solved ? r.total_time : Inf

        # Curvature histories contain accepted estimates; open-loop has no estimate.
        Lk = hasproperty(r, :L_ks) ? collect(Float64, r.L_ks) : Float64[]

        f_last   = isempty(r.values) ? NaN : Float64(last(r.values))
        gap_last = isempty(r.gaps)   ? NaN : Float64(last(r.gaps))

        # Independent final checks are outside every solver timer.
        verified_value = prob.f(r.U, r.V)
        verified_grad_U, verified_grad_V = prob.grad_f(r.U, r.V)
        verified_lmo_U, verified_lmo_V = prob.lmo(verified_grad_U, verified_grad_V)
        verified_gap = dot(verified_grad_U, r.U - verified_lmo_U) +
                       dot(verified_grad_V, r.V - verified_lmo_V)

        per_iter[name]["seed_$(s)"] = Dict(
            "initial_data_sha256" => input_hash,
            "final_U" => copy(r.U), "final_V" => copy(r.V),
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
            "tau1"         => prob.tau1,                    # seed-dependent radii
            "tau2"         => prob.tau2,
        )
        per_iter[name]["seed_$(s)"]["final_feasibility_violation"] = max(0.0, opnorm(r.U) - prob.tau1, opnorm(r.V) - prob.tau2)
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
# 4b. Save per-iteration histories: one .mat per method
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
# 5. Summary table + save raw results
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
    println("    last f: mean=", round(mean(fv), sigdigits=6),
            "   last gap: mean=", round(mean(gpv), sigdigits=4))
end

matwrite(joinpath(output_dir, "matrixfactorization_benchmark_all_methods.mat"), Dict(
    "seeds"  => collect(seeds),
    "times"  => Dict(matvarname(name) => all_times[name]  for name in method_order),
    "solved" => Dict(matvarname(name) => all_solved[name] for name in method_order),
))

# --------------------------------------------------------------------
# 6. Dolan-Moré performance profile (ratio-based x-axis)
# See: https://tmigot.github.io/posts/2024/06/teaching/
#
# T[p, s] = metric (total_time) for solver s on problem p; Inf means
# solver s did NOT solve problem p (hit max_iter_cap / max_time) -- the
# Failures remain Inf. At a finite cutoff, profile coverage can be lower
# than overall success; both quantities are saved separately.
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

Plots.savefig(p, joinpath(output_dir, "matrix_factorization_performance_profile_all_methods.eps"))
Plots.savefig(p, joinpath(output_dir, "matrix_factorization_performance_profile_all_methods.png"))
println("✅ Saved: matrix_factorization_performance_profile_all_methods.eps in ", output_dir)
display(p)

end # ACGD_RUN_BENCHMARKS
