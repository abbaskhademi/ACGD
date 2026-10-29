# ==========================================================================
# ill_conditioned_quadratic: conditional_gradient_adjustable_scaling
#   x 3 LMOs (L1 / L2 / Linf) x N seeds -> combined performance profile
#   f(x) = 1/2 x' H x ,  H = Diagonal(log-spaced eigenvalues, kappa=1e4)
#   x* = 0 ,  f* = 0   (exact, no noise)
# ==========================================================================
using LinearAlgebra
using Statistics
using Random
using Plots
using Measures
using PyPlot
using MAT
using SparseArrays

pyplot()
PyPlot.PyCall.pyimport("warnings").filterwarnings("ignore")

# --- output goes into <run dir>/quadratic_adjustable_lmo/ -
problem_name = "quadratic_adjustable_lmo"
output_dir = joinpath(get(ENV, "ACGD_OUTPUT_ROOT", joinpath(@__DIR__, "corrected_results")), problem_name)
mkpath(output_dir)
instance_dir = joinpath(output_dir, "instances")
mkpath(instance_dir)

matvarname(s) = replace(s, r"[^A-Za-z0-9_]" => "_")

# --------------------------------------------------------------------
# 1. Problem generator — one seed -> one independent ill-conditioned
#    diagonal quadratic instance.
#    H is fixed in SPECTRUM (log-spaced, kappa) but the seed randomizes
#    which coordinate each eigenvalue sits on (random permutation) and
#    the starting point x0, so 100 seeds give 100 distinct instances
#    while keeping H diagonal (axis-aligned) and x* = 0 exactly.
# --------------------------------------------------------------------
function generate_problem(seed; d=50, kappa=1e4)
    Random.seed!(seed)

    eigs_sorted = exp.(range(log(1.0), log(kappa), length=d))  # log-spaced
    perm = randperm(d)
    eigs = eigs_sorted[perm]                 # randomize axis assignment per seed
    H = Diagonal(eigs)

    x_star = zeros(d)
    fstar  = 0.0

    f(x)      = 0.5 * dot(x, H * x)
    grad_f(x) = H * x

    rho = 1.0
    x0  = randn(d)                           # random start per seed

    return (f=f, grad_f=grad_f, x0=x0, rho=rho, H=H, eigs=eigs,
            seed=seed, optimal_value=fstar, x_star=x_star, d=d, kappa=kappa)
end

# "Lipschitz constant" proxy kept for snapshot/metadata only (Euclidean smoothness)
estimate_L(H) = maximum(diag(H))


# --------------------------------------------------------------------
# LMO factory (same steepest-descent LMOs as before)
# --------------------------------------------------------------------
function lmo_l2(g; rho=1.0)
    ng = norm(g)
    return ng > 0 ? (-rho / ng) .* g : zeros(length(g))
end

function lmo_l1(g; rho=1.0)
    i = argmax(abs.(g))
    d = zeros(length(g))
    d[i] = g[i] == 0 ? 0.0 : -rho * sign(g[i])
    return d
end

function lmo_linf(g; rho=1.0)
    d = -rho .* sign.(g)
    d[d .== 0] .= -rho
    return d
end

make_lmo(geometry::Symbol, rho::Real) = g -> begin
    geometry === :l1   ? lmo_l1(g; rho=rho)   :
    geometry === :l2   ? lmo_l2(g; rho=rho)   :
    geometry === :linf ? lmo_linf(g; rho=rho) :
    error("unknown geometry $geometry")
end

if !isdefined(@__MODULE__, :ACGDSolvers)
    include(joinpath(@__DIR__, "src", "ACGDSolvers.jl"))
end
if !isdefined(@__MODULE__, :ACGDBenchmarkSupport)
    include(joinpath(@__DIR__, "scripts", "benchmark_support.jl"))
end
using .ACGDBenchmarkSupport

function conditional_gradient_adjustable_scaling(f, grad_f, lmo, x0, rho, gamma;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=0.0, kwargs...)
    isfinite(rho) && rho > 0 || throw(ArgumentError("rho must be finite and positive"))
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:adjustable, constrained=false, stop_rule=:objective,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, gamma=gamma, kwargs...)
    return r
end

if get(ENV, "ACGD_RUN_BENCHMARKS", "true") == "true"

# --------------------------------------------------------------------
# 3. Run adjustable-scaling CG with each of the 3 LMOs on each seeded
#    ill-conditioned quadratic
# --------------------------------------------------------------------
println("\n================ Adaptive adjustable: L1 vs L2 vs Linf LMO (ill-conditioned quadratic) ================")
seeds  = 1:parse(Int, get(ENV, "ACGD_SEED_COUNT", "100"))
isempty(seeds) && throw(ArgumentError("ACGD_SEED_COUNT must be at least one"))

gamma0 = 1/4

method_order = ["Adaptive adjustable + norm one",
                "Adaptive adjustable + norm two",
                "Adaptive adjustable + norm infinity"]

geometry_of = Dict(
    "Adaptive adjustable + norm one"   => :l1,
    "Adaptive adjustable + norm two"   => :l2,
    "Adaptive adjustable + norm infinity" => :linf,
)

problem_parameters = (d=50, kappa=1e4)
warmups = parse(Int, get(ENV, "ACGD_WARMUPS", "1"))
repeats = parse(Int, get(ENV, "ACGD_REPEATS", "1"))
order_seed = 20261021

all_times  = Dict(name => fill(Inf, length(seeds)) for name in method_order)
all_solved = Dict(name => falses(length(seeds)) for name in method_order)

per_iter     = Dict(name => Dict{String,Any}() for name in method_order)
report_lines = String[]

write_run_manifest(joinpath(output_dir, "run_manifest.toml"), Dict(
    "problem" => problem_name, "script" => basename(@__FILE__),
    "stop_rule" => "objective", "epsilon" => 1e-5,
    "max_iter_cap" => parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")), "max_time" => Inf,
    "gamma" => gamma0, "beta" => 2.0, "delta" => 1e-10,
    "solver_method" => "conditional_gradient_adjustable_scaling",
    "lmo_geometries" => method_order,
    "objective" => "f(x) = 1/2 x'Hx, H = Diagonal(log-spaced eigenvalues), x* = 0, f* = 0",
    "generator_rng" => "original Julia default RNG explicitly reseeded per instance; see pinned Julia version",
    "method_order" => method_order,
    "execution_base_order" => method_order,
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
    L_est = estimate_L(prob.H)
    snapshot = problem_snapshot(prob)
    snapshot["L_estimate"] = L_est
    input_hash = snapshot_sha256(snapshot)
    local fstar = prob.optimal_value   # exactly 0.0 for this problem

    calls = [
        name => () -> conditional_gradient_adjustable_scaling(
            prob.f, prob.grad_f, make_lmo(geometry_of[name], prob.rho),
            prob.x0, prob.rho, gamma0; fstar=fstar)
        for name in method_order
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

        verified_value = prob.f(r.x)
        verified_gradient = prob.grad_f(r.x)
        verified_gap = norm(verified_gradient)
        verified_distance_to_xstar = norm(r.x .- prob.x_star)

        per_iter[name]["seed_$(s)"] = Dict(
            "initial_data_sha256" => input_hash,
            "final_x" => copy(r.x),
            "independently_verified_final_value" => verified_value,
            "independently_verified_final_gap" => verified_gap,
            "independently_verified_distance_to_xstar" => verified_distance_to_xstar,
            "instance"     => idx,
            "seed"         => s,
            "geometry"     => string(geometry_of[name]),
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
        per_iter[name]["seed_$(s)"]["final_feasibility_violation"] = 0.0
        merge!(per_iter[name]["seed_$(s)"], ACGDSolvers.result_metadata(r), benchmark_metadata(r))
        per_iter[name]["seed_$(s)"]["accepted_L_max"] = isempty(r.L_ks) ? NaN : maximum(r.L_ks)
        per_iter[name]["seed_$(s)"]["accepted_L_arithmetic_mean"] = isempty(r.L_ks) ? NaN : mean(r.L_ks)
        per_iter[name]["seed_$(s)"]["accepted_L_harmonic_mean"] = harmonic_mean(r.L_ks)

        status = r.solved ? "solved" : "NOT solved"
        line = string(rpad(name, 32), status,
                      "  iters=", r.k,
                      "  time=", round(r.total_time, sigdigits=6), "s",
                      "  f_last=", round(f_last, sigdigits=6),
                      "  gradnorm_last=", round(gap_last, sigdigits=4),
                      "  ||x-x*||=", round(verified_distance_to_xstar, sigdigits=4))
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
# 3b. Save per-iteration histories: one .mat per LMO variant
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

open(joinpath(output_dir, "$(problem_name)_report.txt"), "w") do io
    println(io, "Problem: ", problem_name, "   seeds: ", collect(seeds))
    for l in report_lines
        println(io, l)
    end
end

# --------------------------------------------------------------------
# 4. Summary table
# --------------------------------------------------------------------
println("\n# Summary ($(length(seeds))-seed sweep) #")
println(rpad("Variant", 32), rpad("Solved", 10), rpad("Mean time (s)", 16), "Median time (s)")
for name in method_order
    t = filter(isfinite, all_times[name])
    n_solved = length(t)
    mean_t = n_solved > 0 ? round(mean(t), sigdigits=6) : NaN
    med_t  = n_solved > 0 ? round(median(t), sigdigits=6) : NaN
    println(rpad(name, 32), rpad("$n_solved/$(length(seeds))", 10), rpad(string(mean_t), 16), string(med_t))

    fv  = [per_iter[name]["seed_$(s)"]["final_value"] for s in seeds]
    gpv = [per_iter[name]["seed_$(s)"]["final_gap"]   for s in seeds]
    dxv = [per_iter[name]["seed_$(s)"]["independently_verified_distance_to_xstar"] for s in seeds]
    println("    last f: mean=", round(mean(fv), sigdigits=6),
            "   last grad norm: mean=", round(mean(gpv), sigdigits=4),
            "   ||x-x*||: mean=", round(mean(dxv), sigdigits=4))
end

matwrite(joinpath(output_dir, "benchmark_adjustable_lmo.mat"), Dict(
    "seeds"  => collect(seeds),
    "times"  => Dict(matvarname(name) => all_times[name]  for name in method_order),
    "solved" => Dict(matvarname(name) => all_solved[name] for name in method_order),
))

# --------------------------------------------------------------------
# 5. Dolan-Moré performance profile (3 curves only)
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
for reference_name in method_order
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

method_colors = Dict(
    "Adaptive adjustable + norm one"   => :mediumpurple,
    "Adaptive adjustable + norm two"   => :green,
    "Adaptive adjustable + norm infinity" => :darkslategray,
)
for (i, series) in enumerate(p.series_list)
    series[:linecolor] = method_colors[method_order[i]]
end

Plots.savefig(p, joinpath(output_dir,
    "Adaptive_adjustable_LMO_comparison_performance_profile.eps"))
Plots.savefig(p, joinpath(output_dir,
    "Adaptive_adjustable_LMO_comparison_performance_profile.png"))
println("\n✅ Saved performance profile")
display(p)

end # ACGD_RUN_BENCHMARKS