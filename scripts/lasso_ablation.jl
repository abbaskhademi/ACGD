#!/usr/bin/env julia
# Controlled diagnostic: original versus narrowly patched Lasso solver bodies.
# Usage: julia --project=. scripts/lasso_ablation.jl OUTPUT_DIR [SEED_COUNT]
# Loads function definitions only: no original sweep, plots, or file writes.
using LinearAlgebra, Random, Statistics, TOML, SHA, Dates, Printf

const ROOT = dirname(@__DIR__)
const OUT = abspath(ARGS[1])
const N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
mkpath(OUT)
const SNAPSHOT = joinpath(OUT, "source")
mkpath(SNAPSHOT)
const REV = "1a2f5e45b8ec9b4bd4924d1e91217d4c1e52d6af"
const METHODS = ["constant", "adjustable", "backtracking"]
const VARIANTS = ["original", "cached", "cached_counter_fixed", "current"]
const FNAMES = [:conditional_gradient_adaptive,
    :conditional_gradient_adjustable_scaling, :conditional_gradient_pure_backtracking]
BLAS.set_num_threads(1)

original = replace(read(`git -C $ROOT show $REV:lasso_problem.jl`, String), "\r\n" => "\n")
write(joinpath(SNAPSHOT, "original.jl"), original)
function replace_checked(s, old, new, expected)
    @assert length(findall(old, s)) == expected "source patch anchor changed"
    return replace(s, old => new)
end
cached = replace_checked(original,
    "current_f = f(x_curr); prev_grad = current_grad",
    "current_f = new_f; prev_grad = current_grad", 2)
# Julia's inner while introduces a local scope: declare the accepted value in
# the enclosing loop so the post-loop cache update can access it.
cached = replace_checked(cached,
    "        Lknormd2 = L_k * normd2\n        t_k = min(gap / Lknormd2, 1.0)",
    "        new_f = current_f\n        Lknormd2 = L_k * normd2\n        t_k = min(gap / Lknormd2, 1.0)", 2)
cached = replace_checked(cached,
    "f0 = f(x0)\n    values = [f(x_prev)]",
    "f0 = f(x0)\n    current_f = f0\n    values = [f(x_prev)]", 1)
cached = replace_checked(cached, "f_prev = f(x_prev)", "f_prev = current_f", 1)
write(joinpath(SNAPSHOT, "cached.jl"), cached)

counter = replace_checked(cached, "                push!(recent_backtracks, i)\n", "", 1)
block_start = findfirst("        if k % 10 == 0 && k > 0", counter)
block_end = findnext("        k += 1", counter, last(block_start))
counter = counter[1:first(block_start)-1] * """
        push!(recent_backtracks, i)
        if (k + 1) % 10 == 0
            total_backtracks = sum(recent_backtracks)
            if total_backtracks == 0
                gamma *= 0.9
            elseif total_backtracks > 10
                gamma = min(1.0, gamma * 1.1)
            end
            empty!(recent_backtracks)
        end
""" * counter[first(block_end):end]
write(joinpath(SNAPSHOT, "cached_counter_fixed.jl"), counter)
cp(joinpath(ROOT, "src", "ACGDSolvers.jl"), joinpath(SNAPSHOT, "ACGDSolvers.jl"); force=true)
cp(joinpath(ROOT, "lasso_problem.jl"), joinpath(SNAPSHOT, "current.jl"); force=true)
cp(@__FILE__, joinpath(SNAPSHOT, "lasso_ablation.jl"); force=true)
include(joinpath(SNAPSHOT, "ACGDSolvers.jl"))
include(joinpath(ROOT, "test", "load_definitions.jl"))
mods = [load_problem_definitions(joinpath(SNAPSHOT, v * ".jl")) for v in VARIANTS]
solvers = [(variant=v, method=METHODS[m], fn=getfield(mods[v], FNAMES[m]))
    for v in eachindex(VARIANTS) for m in eachindex(METHODS)]

function invoke_solver(fn, method, f, g, lmo, x0)
    if method == "backtracking"
        return fn(f, g, lmo, x0; max_time=Inf)
    end
    return fn(f, g, lmo, x0, 0.25; max_time=Inf)
end
function measure(fn, method, p)
    started = time_ns()
    r = invoke_solver(fn, method, p.f, p.grad_f, p.lmo, p.x0)
    elapsed = (time_ns() - started) / 1e9
    return r, elapsed
end
function audit(fn, method, p)
    nf = Ref(0); ng = Ref(0); nl = Ref(0)
    f(x) = (nf[] += 1; p.f(x))
    g(x) = (ng[] += 1; p.grad_f(x))
    lmo(x) = (nl[] += 1; p.lmo(x))
    r = invoke_solver(fn, method, f, g, lmo, p.x0)
    gfinal = p.grad_f(r.x)
    return Dict("objective_evaluations" => nf[], "gradient_evaluations" => ng[],
        "lmo_evaluations" => nl[], "iterations" => r.k, "solved" => r.solved,
        "final_objective" => p.f(r.x), "post_step_gap" => -dot(gfinal, p.lmo(gfinal) - r.x),
        "feasibility_violation" => max(0.0, norm(r.x, 1) - 10),
        "iterate_sha256" => bytes2hex(sha256(reinterpret(UInt8, r.x))),
        "history_sha256" => bytes2hex(sha256(reinterpret(UInt8, r.values))),
        "rejected_trials" => hasproperty(r, :backtracks) ? sum(r.backtracks) : -1)
end
function save_toml(path, x)
    open(path, "w") do io
        TOML.print(io, x; sorted=true)
    end
end
manifest = Dict("started_utc" => string(now(UTC)), "base_revision" => REV,
    "julia_version" => string(VERSION), "julia_threads" => Threads.nthreads(),
    "blas_threads" => BLAS.get_num_threads(), "blas_config" => string(BLAS.get_config()),
    "cpu" => Sys.CPU_NAME, "seeds" => collect(1:N), "warmups" => 1, "repeats" => 3,
    "variants" => VARIANTS, "methods" => METHODS, "m" => 200, "n" => 1000,
    "epsilon" => 1e-5, "max_iter_cap" => 100000, "max_time" => "Inf",
    "aggregation" => "per-seed median of three repeats; rotating variant/method order",
    "timing" => "both original reported iteration sum and complete external elapsed; call counters in separate untimed audit",
    "counter_patch" => "one rejection count per accepted step; update after steps10,20,...",
    "cache_patch" => "accepted-value reuse only; original initialization left intact",
    "source_sha256" => Dict(f => bytes2hex(sha256(read(joinpath(SNAPSHOT, f)))) for f in readdir(SNAPSHOT)))
save_toml(joinpath(OUT, "manifest.toml"), manifest)
all_results = Dict{String,Any}()
for seed in 1:N
    p = mods[1].generate_problem(seed)
    data = Dict{String,Any}()
    # Warm the exact uninstrumented specialization used for timing.
    for s in solvers
        r, _ = measure(s.fn, s.method, p)
        @assert r.solved
        data[VARIANTS[s.variant] * "/" * s.method] = Dict{String,Any}(
            "external_seconds" => Float64[], "reported_seconds" => Float64[],
            "iterations" => Int[])
    end
    for rep in 1:3
        for j in 0:length(solvers)-1
            s = solvers[mod1(j + seed + rep - 1, length(solvers))]
            r, elapsed = measure(s.fn, s.method, p)
            @assert r.solved
            d = data[VARIANTS[s.variant] * "/" * s.method]
            push!(d["external_seconds"], elapsed)
            push!(d["reported_seconds"], r.total_time)
            push!(d["iterations"], r.k)
        end
    end
    # Count calls separately; instrumentation never enters measured timings.
    for s in solvers
        key = VARIANTS[s.variant] * "/" * s.method
        data[key]["audit"] = audit(s.fn, s.method, p)
        @assert all(==(data[key]["audit"]["iterations"]), data[key]["iterations"])
    end
    # Caching must not change a single accepted objective or final iterate.
    for m in METHODS
        @assert data["original/" * m]["audit"]["history_sha256"] == data["cached/" * m]["audit"]["history_sha256"]
        @assert data["original/" * m]["audit"]["iterate_sha256"] == data["cached/" * m]["audit"]["iterate_sha256"]
    end
    data["input_sha256"] = bytes2hex(sha256(vcat(reinterpret(UInt8, vec(p.A)), reinterpret(UInt8, p.x_star))))
    all_results["seed_$seed"] = data
    save_toml(joinpath(OUT, "seed_$seed.toml"), data)
    println("Completed seed $seed/$N"); flush(stdout)
end
save_toml(joinpath(OUT, "measurements.toml"), all_results)
open(joinpath(OUT, "summary.txt"), "w") do io
    println(io, "Controlled Lasso attribution, seeds1:$N; one warmup/three measured repetitions.")
    println(io, "All variants use identical original problem closures, one Julia/BLAS thread.")
    println(io, "Every solve converged. Caching preserved bit-identical objective histories and final iterates.")
    println(io, "Times are medians across per-seed medians. Ratios >1 favor ACGD.")
    for variant in VARIANTS, metric in ("reported_seconds", "external_seconds")
        t = [[median(all_results["seed_$seed"][variant * "/" * m][metric]) for seed in 1:N] for m in METHODS]
        println(io, "\n$variant, $metric:")
        for (j, m) in enumerate(METHODS)
            records = [all_results["seed_$seed"][variant * "/" * m]["audit"] for seed in 1:N]
            @printf(io, "%-14s time %.8f, iterations %.1f, f calls %.1f, gradient calls %.1f, fastest %d/%d\n",
                m, median(t[j]), median([r["iterations"] for r in records]),
                median([r["objective_evaluations"] for r in records]),
                median([r["gradient_evaluations"] for r in records]),
                count(s -> t[j][s] == minimum(v[s] for v in t), 1:N), N)
        end
        @printf(io, "Median paired BT/constant %.6f, BT/adjustable %.6f\n", median(t[3] ./ t[1]), median(t[3] ./ t[2]))
    end
end
manifest["finished_utc"] = string(now(UTC))
save_toml(joinpath(OUT, "manifest.toml"), manifest)
println(read(joinpath(OUT, "summary.txt"), String))
