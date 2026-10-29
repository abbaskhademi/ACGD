# ==========================================================================
# Collaborative filtering (MovieLens 100k, Huber loss, nuclear-norm ball)
# L_k-history plot for the 3 backtracking methods only:
#   Adaptive constant, Adaptive adjustable, Pure backtracking
# ==========================================================================

using DataFrames, CSV, Downloads
using LinearAlgebra, Statistics, Random, SparseArrays, Arpack
using MAT, Plots, Measures, PyPlot, SHA

pyplot()
PyPlot.PyCall.pyimport("warnings").filterwarnings("ignore")

problem_name = "collaborative_filtering"
output_dir = joinpath(get(ENV, "ACGD_OUTPUT_ROOT", joinpath(@__DIR__, "corrected_results")), problem_name)
mkpath(output_dir)
matvarname(s) = replace(s, r"[^A-Za-z0-9_]" => "_")

# --------------------------------------------------------------------
# 1. Problem: MovieLens 100k
# --------------------------------------------------------------------
csv_path = joinpath(@__DIR__, "movielens100k.csv")
if !isfile(csv_path)
    Downloads.download("https://raw.githubusercontent.com/cyrillewcombettes/boostfw/master/movielens100k.csv", csv_path)
    println("File downloaded successfully.")
end

data = CSV.read(csv_path, DataFrame, header=false)
rename!(data, [:user_id, :item_id, :rating, :timestamp])

Ytab = unstack(data, :user_id, :item_id, :rating)
Y = convert(Matrix{Union{Float64,Missing}}, Matrix(select(Ytab, Not(:user_id))))

m, n = size(Y)
mn   = m * n
y    = reshape(Y, mn)
obs  = Float64.(.!ismissing.(y))      # 1 on observed entries, 0 otherwise
yv   = coalesce.(y, 0.0)
N    = nrow(data)                     # number of ratings
rho  = 1.0
tau  = 5000.0
L_global = 1 / N                      # theoretical Lipschitz constant
println("m=$m, n=$n, N=$N, L = 1/N = $L_global")

huber(r, t)     = abs(t) <= r ? t^2 / 2 : r * (abs(t) - r / 2)
der_huber(t, r) = clamp(t, -r, r)

f(x)      = sum(huber.(rho, (yv .- x) .* obs)) / N
grad_f(x) = -der_huber.((yv .- x) .* obs, rho) ./ N

lmo_v0 = randn(MersenneTwister(23), min(m, n))
# Arpack v0 is explicit so each method uses the same Krylov initialization.
function lmo(g)
    G = reshape(-g, (m, n))
    u, s, v = svds(G, nsv=1, v0=copy(lmo_v0))[1]
    return tau * reshape(u * v', mn)
end

# reproducible random rank-1 starting point on the boundary (||X0||_* = tau)
Random.seed!(23)
u0 = randn(m); v0 = randn(n)
x0 = reshape(tau * (u0 * v0') / (norm(u0) * norm(v0)), mn)

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
        max_time=Inf, fstar=NaN, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:adaptive, constrained=true, stop_rule=:stationarity,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, gamma=gamma, kwargs...)
    return r
end

function conditional_gradient_adjustable_scaling(f, grad_f, lmo, x0, gamma;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=NaN, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:adjustable, constrained=true, stop_rule=:stationarity,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, gamma=gamma, kwargs...)
    return r
end

function conditional_gradient_pure_backtracking(f, grad_f, lmo, x0;
        epsilon=1e-5, delta=1e-10, beta=2.0, max_iter_cap=parse(Int, get(ENV, "ACGD_MAX_ITER", "100000")),
        max_time=Inf, fstar=NaN, kwargs...)
    r = ACGDSolvers.solve(f, grad_f, lmo, x0;
        method=:backtracking, constrained=true, stop_rule=:stationarity,
        epsilon=epsilon, fstar=fstar, delta=delta, beta=beta, max_iter_cap=max_iter_cap, max_time=max_time, kwargs...)
    return r
end

if get(ENV, "ACGD_RUN_BENCHMARKS", "true") == "true"

# --------------------------------------------------------------------
# 3. Run the 3 methods
# --------------------------------------------------------------------
gamma0   = 1/4
max_iter = parse(Int, get(ENV, "ACGD_TRACE_ITER", "500"))
warmups = parse(Int, get(ENV, "ACGD_WARMUPS", "1"))
repeats = benchmark_repeats(3)
order_seed = 20261022
source_file = csv_path
source_file_sha256 = bytes2hex(sha256(read(source_file)))

method_order = ["Adaptive constant", "Adaptive adjustable", "Pure backtracking"]

observed_indices = findall(!iszero, obs)
instance_data = problem_snapshot((x0=x0, m=m, n=n, N=N, rho=rho, tau=tau,
    L_global=L_global, lmo_v0=lmo_v0, observed_indices=observed_indices,
    observed_ratings=yv[observed_indices], source_file_sha256=source_file_sha256))
initial_data_sha256 = snapshot_sha256(instance_data)
instance_data["initial_data_sha256"] = initial_data_sha256

write_run_manifest(joinpath(output_dir, "run_manifest.toml"), Dict(
    "problem" => problem_name, "script" => basename(@__FILE__),
    "stop_rule" => "stationarity", "epsilon" => 1e-5,
    "max_iter_cap" => max_iter, "max_time" => Inf,
    "gamma" => gamma0, "beta" => 2.0, "eta" => 0.9, "block_size" => 10,
    "probe_seed" => 23, "delta" => 1e-10,
    "generator_rng" => "original Julia default RNG explicitly reseeded per instance; see pinned Julia version",
    "method_order" => method_order,
    "experiment_kind" => "accepted-curvature trace; stationarity stopping or fixed iteration cap",
    "source_file" => "movielens100k.csv", "source_file_sha256" => source_file_sha256,
    "source_url" => "https://raw.githubusercontent.com/cyrillewcombettes/boostfw/master/movielens100k.csv",
    "initial_data_sha256" => initial_data_sha256,
    "external_timing" => true, "gc_before_call" => true,
    "order_mode" => "rotate", "order_seed" => order_seed,
    "accepted_curvature" => "accepted L_k before eta scaling; one value per accepted step",
    "smoothness_preprocessing" => "problem constants computed before solver timing",
    "feasibility_scope" => "representative repeat checked outside solver timing",
    "run_controls" => Dict(name => get(ENV, name, default) for (name, default) in
        [("ACGD_OUTPUT_ROOT", joinpath(@__DIR__, "corrected_results")),
         ("ACGD_SEED_COUNT", "100"), ("ACGD_MAX_ITER", "100000"),
         ("ACGD_TRACE_ITER", "500"), ("ACGD_WARMUPS", "2"), ("ACGD_REPEATS", "6")]),
    "warmups" => warmups,
    "repeats" => repeats,
    "seeds" => [23],
    "aggregation" => "median external complete-call time; trace may reach the iteration cap without stationarity",
); root=@__DIR__)

calls = [
    "Adaptive constant" => () -> conditional_gradient_adaptive(f, grad_f, lmo, x0, gamma0; max_iter_cap=max_iter),
    "Adaptive adjustable" => () -> conditional_gradient_adjustable_scaling(f, grad_f, lmo, x0, gamma0; max_iter_cap=max_iter),
    "Pure backtracking" => () -> conditional_gradient_pure_backtracking(f, grad_f, lmo, x0; max_iter_cap=max_iter),
]
runs = benchmark_methods(calls; warmups=warmups, repeats=repeats,
    instance_index=23, external_timing=true, gc_before_call=true,
    order_mode=:rotate, order_seed=order_seed)

report_lines = String[]
saved_runs = Dict{String,Any}()
for name in method_order
    r = runs[name]
    raw_statuses = join(unique(String(raw["status"]) for raw in r.raw_repeats), ", ")
    final_gradient = grad_f(r.x)
    final_gap = iszero(norm(final_gradient)) ? 0.0 : dot(final_gradient, r.x - lmo(final_gradient))
    line = string(rpad(name, 22), raw_statuses,
            "  iters=", r.k, "  time=", round(r.total_time, sigdigits=6), "s",
            "  f_last=", round(last(r.values), digits=6),
            "  gap_last=", round(last(r.gaps), sigdigits=4),
            "  max L_k=", round((isempty(r.L_ks) ? NaN : maximum(r.L_ks)), sigdigits=4),
            "  mean L_k=", round((isempty(r.L_ks) ? NaN : mean(r.L_ks)), sigdigits=4),
            "  harmonic mean L_k=", round(harmonic_mean(r.L_ks), sigdigits=4))

    println(line); push!(report_lines, line)

    saved = Dict{String,Any}(
        "problem" => problem_name, "method" => name,
        "seed" => 23, "initial_data_sha256" => initial_data_sha256,
        "source_file_sha256" => source_file_sha256,
        "final_x" => r.x,
        "independently_verified_final_value" => f(r.x),
        "independently_verified_final_gap" => final_gap,
        "independently_verified_final_gradnorm" => norm(final_gradient),
        "values" => collect(Float64, r.values), "gaps" => collect(Float64, r.gaps),
        "times" => collect(Float64, r.times), "L_ks" => collect(Float64, r.L_ks),
        "iterations" => r.k, "solved" => r.solved ? 1 : 0,
        "total_time" => r.total_time, "L_global" => L_global,
        "final_feasibility_violation" => max(0.0, sum(svdvals(reshape(r.x, m, n))) - tau),
        "accepted_L_max" => isempty(r.L_ks) ? NaN : maximum(r.L_ks),
        "accepted_L_arithmetic_mean" => isempty(r.L_ks) ? NaN : mean(r.L_ks),
        "accepted_L_harmonic_mean" => harmonic_mean(r.L_ks),
        "feasibility_scope" => "representative repeat; all raw repeat objective/residual outcomes retained",
        ACGDSolvers.result_metadata(r)..., benchmark_metadata(r)...,
    )
    saved_runs[matvarname(name)] = saved
    save_instance_checkpoint(joinpath(output_dir, "$(problem_name)_iterations_$(matvarname(name)).mat"), saved; writer=matwrite)
end
save_instance_checkpoint(joinpath(output_dir, "instances", "seed_23.mat"),
    Dict("problem" => instance_data, "runs" => saved_runs); writer=matwrite)
flush(stdout)

open(joinpath(output_dir, "$(problem_name)_summary.txt"), "w") do io
    println(io, "Problem: ", problem_name, "   L = ", L_global)
    foreach(l -> println(io, l), report_lines)
end

# --------------------------------------------------------------------
# 4. L_k plot (EPS) — layered scatter style
# --------------------------------------------------------------------
ped_Lk  = runs["Pure backtracking"].L_ks
our_Lk  = runs["Adaptive constant"].L_ks
our5_Lk = runs["Adaptive adjustable"].L_ks

style = Dict(
    :titlefont  => (20, "serif"),
    :guidefont  => (20, "serif"),
    :tickfont   => (20, "serif"),
    :legendfont => (14, "serif"),
    :grid       => false,
    :framestyle => :box,
    :margin     => 5mm,
    :size       => (900, 700),
    :linewidth  => 2
)

p1 = Plots.plot()  # Start with empty plot

# 1. RED first — hidden from legend (drawn underneath)
Plots.plot!(p1, ped_Lk,
            label=false,
            color=:red,
            seriestype=:scatter,
            markersize=8,
            marker=:star5,
            markerstrokewidth=0)

# 2. BLUE — legend entry
Plots.plot!(p1, our_Lk,
            label="Adaptive constant",
            color=:blue,
            seriestype=:scatter,
            markersize=0.1,
            marker=:rtriangle,
            markerstrokewidth=0)

# 3. GREEN — legend entry
Plots.plot!(p1, our5_Lk,
            label="Adaptive adjustable",
            color=:green,
            seriestype=:scatter,
            markersize=0.1,
            marker=:circle,
            markerstrokewidth=0)

# 4. RED again — legend entry (appears LAST in legend)
Plots.plot!(p1, ped_Lk,
            label="Pure backtracking",
            color=:red,
            seriestype=:scatter,
            markersize=5,
            marker=:star5,
            markerstrokewidth=0)

# 5. BLUE on top — not in legend
Plots.plot!(p1, our_Lk,
            label=false,
            color=:blue,
            seriestype=:scatter,
            markersize=7,
            marker=:rtriangle,
            markerstrokewidth=0)

# 6. GREEN on top — not in legend
Plots.plot!(p1, our5_Lk,
            label=false,
            color=:green,
            seriestype=:scatter,
            markersize=5,
            marker=:circle,
            markerstrokewidth=0)

# Axis labels and style
Plots.plot!(p1,
        xlabel="Iteration",
        ylabel="Local Lipschitz Estimate";
        style...)

fname = joinpath(output_dir, "Collaborative_L.eps")
Plots.savefig(p1, fname)
Plots.savefig(p1, joinpath(output_dir, "Collaborative_L.png"))
println("✅ Saved: ", fname)
display(p1)

end # ACGD_RUN_BENCHMARKS
