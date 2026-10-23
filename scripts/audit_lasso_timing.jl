#!/usr/bin/env julia
# Independent stopwatch / compilation / allocation / GC sensitivity audit.
# Does not change the solver or overwrite the archived full benchmark.
using LinearAlgebra, Statistics, Random, TOML, SHA, Dates, Printf

const ROOT = dirname(@__DIR__)
const OUT = abspath(ARGS[1])
const N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
const ARCHIVE = joinpath(ROOT, "corrected_results", "lasso_20261021T163316Z", "source")
const METHODS = ["constant", "adjustable", "backtracking"]
const ORDERS = [(1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1)]
const POLICIES = ["natural_gc", "collect_before_call"]
mkpath(OUT)
mkpath(joinpath(OUT, "source"))
for (src, name) in [(joinpath(ARCHIVE, "src", "ACGDSolvers.jl"), "ACGDSolvers.jl"),
                    (joinpath(ARCHIVE, "lasso_problem.jl"), "lasso_problem.jl"),
                    (joinpath(ROOT, "test", "load_definitions.jl"), "load_definitions.jl"),
                    (joinpath(ROOT, "Project.toml"), "Project.toml"),
                    (joinpath(ROOT, "Manifest.toml"), "Manifest.toml"),
                    (@__FILE__, "audit_lasso_timing.jl")]
    cp(src, joinpath(OUT, "source", name); force=true)
end
include(joinpath(OUT, "source", "ACGDSolvers.jl"))
include(joinpath(OUT, "source", "load_definitions.jl"))
const DEFS = load_problem_definitions(joinpath(OUT, "source", "lasso_problem.jl"))
const SOLVERS = [DEFS.conditional_gradient_adaptive,
    DEFS.conditional_gradient_adjustable_scaling, DEFS.conditional_gradient_pure_backtracking]
BLAS.set_num_threads(1)

function invoke(fn::F, method, p) where F
    if method == 3
        return fn(p.f, p.grad_f, p.lmo, p.x0; max_iter_cap=100000, max_time=Inf)
    end
    return fn(p.f, p.grad_f, p.lmo, p.x0, 0.25; max_iter_cap=100000, max_time=Inf)
end
function measure(fn::F, method, p, collect_before) where F
    # Full collection is outside both stopwatches; GC remains enabled throughout
    # the solve and any collections triggered by that solve stay inside its time.
    collect_before && GC.gc(true)
    sample = @timed invoke(fn, method, p)
    r = sample.value
    @assert r.solved && r.stopping_criterion == "objective"
    return Dict{String,Any}(
        "external_seconds" => sample.time, "internal_seconds" => r.total_time,
        "gc_seconds" => sample.gctime, "allocated_bytes" => sample.bytes,
        "compile_seconds" => sample.compile_time, "recompile_seconds" => sample.recompile_time,
        "gc_pauses" => sample.gcstats.pause, "full_collections" => sample.gcstats.full_sweep,
        "iterations" => r.k, "objective_evaluations" => r.objective_evaluations,
        "final_value" => r.final_value, "final_gap" => r.final_gap,
        "iterate_sha256" => bytes2hex(sha256(reinterpret(UInt8, r.x))),
        "solved" => r.solved, "stopping_criterion" => r.stopping_criterion)
end
function save(path, x)
    open(path, "w") do io
        TOML.print(io, x; sorted=true)
    end
end
manifest = Dict{String,Any}(
    "started_utc" => string(now(UTC)), "julia_version" => string(VERSION),
    "julia_threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(),
    "cpu" => Sys.CPU_NAME, "blas_config" => string(BLAS.get_config()),
    "process_status" => read("/proc/self/status", String),
    "seeds" => collect(1:N), "methods" => METHODS, "policies" => POLICIES,
    "warmups" => 2, "repeats_per_policy" => 6, "order_rng_seed" => 20261021,
    "order_rule" => "all six permutations once per seed/policy, shuffled by local RNG; policy order alternates by seed",
    "memory_rule" => "only scalar summaries retained between solves; independent isolated-call sensitivity check",
    "source_archive" => ARCHIVE,
    "source_sha256" => Dict(f => bytes2hex(sha256(read(joinpath(OUT, "source", f)))) for f in readdir(joinpath(OUT, "source"))))
save(joinpath(OUT, "fresh_manifest.toml"), manifest)
rng = Xoshiro(20261021)
results = Dict{String,Any}()
for seed in 1:N
    p = DEFS.generate_problem(seed)
    # Warm precisely the measured call path, including @timed bookkeeping.
    for _ in 1:2, m in 1:3
        measure(SOLVERS[m], m, p, false)
    end
    data = Dict{String,Any}()
    for policy_index in (isodd(seed) ? (1,2) : (2,1))
        policy = POLICIES[policy_index]
        records = Dict{String,Any}(m => Any[] for m in METHODS)
        order_indices = randperm(rng, 6)
        GC.gc(true) # start each policy block without pending prior-block garbage
        schedule = Any[]
        for (rep, order_index) in enumerate(order_indices)
            order = ORDERS[order_index]
            push!(schedule, collect(order))
            for (position, m) in enumerate(order)
                record = measure(SOLVERS[m], m, p, policy_index == 2)
                record["repeat"] = rep
                record["position"] = position
                push!(records[METHODS[m]], record)
            end
        end
        data[policy] = records
        data[policy * "_schedule"] = schedule
    end
    # Timing policy must not alter the optimization path or work counts.
    for m in METHODS
        samples = vcat(data[POLICIES[1]][m], data[POLICIES[2]][m])
        for field in ("iterations", "objective_evaluations", "iterate_sha256")
            @assert all(s[field] == samples[1][field] for s in samples)
        end
    end
    results["seed_$seed"] = data
    save(joinpath(OUT, "fresh_seed_$seed.toml"), data)
    println("Completed independent timing check $seed/$N"); flush(stdout)
end
save(joinpath(OUT, "fresh_measurements.toml"), results)
open(joinpath(OUT, "fresh_summary.txt"), "w") do io
    println(io, "Independent Lasso timing check: $N seeds; six balanced-order repeats per GC policy.")
    println(io, "External @timed includes the complete public solver call. GC remains enabled inside.")
    println(io, "All solves attained objective tolerance; all paths/work counts identical across policies.")
    for policy in POLICIES
        println(io, "\nPolicy: $policy")
        medians = Dict{String,Vector{Float64}}()
        for m in METHODS
            per_seed = [results["seed_$s"][policy][m] for s in 1:N]
            medians[m] = [median([r["external_seconds"] for r in samples]) for samples in per_seed]
            samples = vcat(per_seed...)
            @printf(io, "%-14s median %.6fs; median GC %.3f%%; median allocations %.2fMB; max compile %.6fs; max external/internal difference %.5f%%\n",
                m, median(medians[m]), 100median([r["gc_seconds"] / r["external_seconds"] for r in samples]),
                median([r["allocated_bytes"] for r in samples]) / 1e6,
                maximum(r["compile_seconds"] for r in samples),
                100maximum((r["external_seconds"] - r["internal_seconds"]) / r["external_seconds"] for r in samples))
        end
        for m in METHODS[1:2]
            ratios = medians["backtracking"] ./ medians[m]
            @printf(io, "BT/%s: median paired ratio %.6f; BT faster %d/%d seeds\n", m, median(ratios), count(<(1), ratios), N)
        end
    end
end
manifest["finished_utc"] = string(now(UTC))
save(joinpath(OUT, "fresh_manifest.toml"), manifest)
println(read(joinpath(OUT, "fresh_summary.txt"), String))
