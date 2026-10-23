#!/usr/bin/env julia
# Postprocessing only: validate saved MovieLens/video accepted-curvature traces.
using Arpack, Dates, LinearAlgebra, MAT, SHA, Statistics, TOML
include(joinpath(@__DIR__, "benchmark_support.jl"))
using .ACGDBenchmarkSupport

scalar(x) = x isa AbstractArray ? only(x) : x
number(x) = Float64(scalar(x))
integer(x) = Int(scalar(x))
truth(x) = Bool(scalar(x))
flat(x) = x isa AbstractArray ? vec(x) : [x]
numbers(x) = Float64.(flat(x))
integers(x) = Int.(flat(x))
strings(x) = strip.(String.(flat(x)))
matname(x) = replace(x, r"[^A-Za-z0-9_]" => "_")
same(a, b; rtol=1e-9, atol=1e-11) = isequal(a, b) || (isfinite(a) && isfinite(b) && isapprox(a, b; rtol=rtol, atol=atol))
file_sha256(path) = open(io -> bytes2hex(sha256(io)), path)

function analyze_curvature_run(output_directory)
    directory = abspath(output_directory)
    manifest_path = joinpath(directory, "run_manifest.toml")
    manifest = TOML.parsefile(manifest_path)
    config = manifest["config"]
    problem = String(config["problem"])
    problem in ("collaborative_filtering", "video_colocalization") || error("Not a supported curvature trace: $problem")
    methods = String.(config["method_order"])
    cap, epsilon = Int(config["max_iter_cap"]), Float64(config["epsilon"])
    repeats, warmups = Int(config["repeats"]), Int(config["warmups"])
    problems, notes = String[], String[]
    check(condition, message) = condition || push!(problems, message)
    check(methods == ["Adaptive constant", "Adaptive adjustable", "Pure backtracking"], "Unexpected trace method set/order")
    check(repeats >= 6 && repeats % 6 == 0, "Three-method balanced design requires multiples of six repeats")
    check(warmups >= 2, "At least two warmups expected")
    check(config["stop_rule"] == "stationarity" && epsilon == 1e-5, "Trace stopping rule differs from stationarity tolerance 1e-5")
    check(manifest["julia_threads"] == 1 && manifest["blas_threads"] == 1, "Expected one Julia/BLAS thread")
    check(Float64(config["max_time"]) == Inf, "Expected unlimited wall-time limit")
    for (key, expected) in (("gamma", .25), ("beta", 2.0), ("eta", .9), ("block_size", 10), ("probe_seed", 23), ("delta", 1e-10))
        check(config[key] == expected, "Unexpected parameter: $key")
    end
    input_path = joinpath(directory, "instances", "seed_23.mat")
    saved = matread(input_path)
    instance = saved["problem"]
    archived_hash = String(pop!(instance, "initial_data_sha256"))
    check(snapshot_sha256(instance) == archived_hash, "Saved numeric problem checksum mismatch")
    check(archived_hash == config["initial_data_sha256"], "Manifest problem checksum mismatch")
    check(String(instance["source_file_sha256"]) == config["source_file_sha256"], "Source dataset checksum mismatch")
    check(occursin(r"^[0-9a-f]{64}$", config["source_file_sha256"]), "Invalid dataset checksum")
    check(get(manifest["input_file_sha256"], config["source_file"], "") == config["source_file_sha256"], "Dataset checksum differs from manifest input hashes")
    x0 = numbers(instance["x0"])
    L = number(instance["L_global"])

    if problem == "collaborative_filtering"
        m, n, N = integer(instance["m"]), integer(instance["n"]), integer(instance["N"])
        rho, tau = number(instance["rho"]), number(instance["tau"])
        observed = integers(instance["observed_indices"])
        ratings, v0 = numbers(instance["observed_ratings"]), numbers(instance["lmo_v0"])
        check(length(observed) == length(ratings) == N, "Observed ratings count is inconsistent")
        check(length(x0) == m*n && length(v0) == min(m, n), "MovieLens dimensions are inconsistent")
        check(rho == 1.0 && tau == 5000.0 && same(L, 1/N), "MovieLens constants differ from intended experiment")
        check(length(unique(observed)) == N && all(i -> 1 <= i <= m*n, observed), "Invalid observed indices")
        f = x -> sum(t -> abs(t) <= rho ? t^2/2 : rho*(abs(t)-rho/2), ratings - x[observed]) / N
        gradient = x -> begin
            g = zeros(length(x)); g[observed] = -clamp.(ratings - x[observed], -rho, rho) / N; g
        end
        oracle = g -> begin
            iszero(norm(g)) && return zeros(length(g))
            u, _, v = svds(reshape(-g, m, n), nsv=1, v0=copy(v0))[1]
            tau * vec(u * v')
        end
        feasibility = x -> max(0.0, sum(svdvals(reshape(x, m, n))) - tau)
        feasibility_tolerance = 1e-8 * max(1.0, tau)
    else
        A, b = Matrix{Float64}(instance["A"]), numbers(instance["b"])
        blocks, width = integer(instance["n_blocks"]), integer(instance["block_size"])
        check(blocks == 33 && width == 20 && length(x0) == blocks*width, "Video simplex dimensions differ from intended experiment")
        check(size(A) == (length(x0), length(x0)) && length(b) == length(x0), "Video objective dimensions are inconsistent")
        check(same(L, opnorm(A, 2)), "Video smoothness bound differs from ||A||_2")
        f = x -> .5 * dot(x, A*x) + dot(b, x)
        gradient = x -> A*x+b
        oracle = g -> begin
            v = zeros(length(g))
            for i in 0:blocks-1
                v[i*width + argmin(@view g[i*width+1:(i+1)*width])] = 1.0
            end
            v
        end
        feasibility = x -> max(0.0, -minimum(x), maximum(abs.(vec(sum(reshape(x, width, blocks), dims=1)) .- 1)))
        feasibility_tolerance = 1e-9
    end
    initial_gradient = gradient(x0)
    initial_value = f(x0)
    initial_gap = dot(initial_gradient, x0 - oracle(initial_gradient))
    check(feasibility(x0) <= feasibility_tolerance, "Initial point is infeasible")
    summaries = Dict{String,Any}()
    inputs = Dict("run_manifest.toml" => file_sha256(manifest_path), "instances/seed_23.mat" => file_sha256(input_path))
    order_reference = Dict{Int,Vector{String}}()
    for method in methods
        prefix = "$method: "
        history_path = joinpath(directory, "$(problem)_iterations_$(matname(method)).mat")
        run = matread(history_path)
        inputs[basename(history_path)] = file_sha256(history_path)
        check(String(run["method"]) == method && String(run["problem"]) == problem, prefix * "Incorrect method/problem labels")
        check(String(run["initial_data_sha256"]) == archived_hash, prefix * "Problem checksum differs")
        check(String(run["source_file_sha256"]) == config["source_file_sha256"], prefix * "Dataset checksum differs")
        k = integer(run["iterations"])
        values, gaps = numbers(run["values"]), numbers(run["gaps"])
        indices, accepted_L = integers(run["iterate_indices"]), numbers(run["L_ks"])
        check(0 <= k <= cap && indices == collect(0:k), prefix * "Incorrect iteration indices")
        check(length(values) == length(gaps) == k+1 && length(accepted_L) == k, prefix * "Unaligned history lengths")
        check(all(isfinite, values) && all(isfinite, gaps) && all(x -> isfinite(x) && x>0, accepted_L), prefix * "Nonfinite histories or nonpositive accepted curvature")
        check(same(first(values), initial_value) && same(first(gaps), initial_gap), prefix * "Starting objective/gap differs from archived input")
        x = numbers(run["final_x"])
        check(length(x) == length(x0) && all(isfinite, x), prefix * "Invalid final iterate")
        grad = gradient(x)
        objective, gap, violation = f(x), dot(grad, x-oracle(grad)), feasibility(x)
        check(same(last(values), objective) && same(number(run["final_value"]), objective), prefix * "Final objective independently recomputed incorrectly")
        check(same(last(gaps), gap) && same(number(run["final_gap"]), gap), prefix * "Final Frank-Wolfe gap independently recomputed incorrectly")
        for (key, computed) in (("independently_verified_final_value", objective), ("independently_verified_final_gap", gap), ("independently_verified_final_gradnorm", norm(grad)), ("final_feasibility_violation", violation))
            check(same(number(run[key]), computed), prefix * "Independent verification field differs: $key")
        end
        check(violation <= feasibility_tolerance, prefix * "Final iterate is infeasible")
        check(integer(run["termination_iteration"]) == k && same(number(run["final_residual"]), gap), prefix * "Termination residual/index mismatch")
        for (key, computed) in (("accepted_L_max", isempty(accepted_L) ? NaN : maximum(accepted_L)), ("accepted_L_arithmetic_mean", isempty(accepted_L) ? NaN : mean(accepted_L)), ("accepted_L_harmonic_mean", harmonic_mean(accepted_L)))
            check(same(number(run[key]), computed), prefix * "Incorrect accepted curvature summary: $key")
        end
        elapsed = numbers(run["repeat_times"])
        outcomes = run["repeat_outcomes"]
        check(length(elapsed) == repeats && length(outcomes) == repeats, prefix * "Wrong repeat count")
        check(all(t -> isfinite(t) && t>0, elapsed) && same(number(run["total_time"]), median(elapsed)), prefix * "Invalid elapsed median")
        representative = integer(run["representative_repeat"])
        check(representative == argmin(abs.(elapsed .- median(elapsed))), prefix * "Representative repeat is not nearest median")
        positions, raw_statuses = Int[], String[]
        for repetition in 1:repeats
            raw = outcomes["repeat_$repetition"]
            status = String(raw["status"])
            push!(raw_statuses, status)
            check(same(number(raw["final_value"]), objective) && same(number(raw["final_gap"]), gap) && integer(raw["iterations"]) == k, prefix * "Nondeterministic repeat outcomes")
            check(integer(raw["termination_iteration"]) == k && same(number(raw["final_residual"]), gap), prefix * "Raw termination residual/index mismatch")
            solved = truth(raw["solved"])
            check(solved == (status == "converged") && solved == (gap <= epsilon), prefix * "Inconsistent raw stationarity success")
            check(status != "iteration_limit" || k == cap, prefix * "Iteration-limit status before cap")
            check(number(raw["compile_time"]) == 0 && number(raw["recompile_time"]) == 0, prefix * "Compilation occurred during measurement")
            outer, core = number(raw["outer_elapsed_time"]), number(raw["core_elapsed_time"])
            check(same(outer, elapsed[repetition]) && outer >= core && core >= 0, prefix * "External/core timing mismatch")
            check(0 <= number(raw["gc_time"]) <= outer && number(raw["allocated_bytes"]) >= 0, prefix * "Invalid timing diagnostics")
            order = strings(raw["execution_order"])
            position = integer(raw["execution_position"])
            check(Set(order) == Set(methods) && length(order) == 3 && order[position] == method, prefix * "Invalid execution order/position")
            check(!haskey(order_reference, repetition) || order_reference[repetition] == order, prefix * "Execution orders disagree across methods")
            order_reference[repetition] = order
            push!(positions, position)
        end
        check(all(p -> count(==(p), positions) == repeats÷3, 1:3), prefix * "Execution positions are unbalanced")
        check(truth(run["solved"]) == all(raw -> truth(raw["solved"]), valuesof(outcomes)), prefix * "Aggregate success is inconsistent")
        check(length(unique(raw_statuses)) == 1, prefix * "Repeat termination statuses differ")
        policy = run["measurement_policy"]
        check(truth(policy["external_timing"]) && truth(policy["gc_before_call"]) && String(policy["order_mode"]) == "balanced", prefix * "Wrong measurement policy")
        checkpoint = saved["runs"][matname(method)]
        check(numbers(checkpoint["final_x"]) == x && numbers(checkpoint["repeat_times"]) == elapsed && numbers(checkpoint["L_ks"]) == accepted_L, prefix * "Checkpoint differs from method file")
        summaries[method] = Dict("status" => unique(raw_statuses), "iterations" => k,
            "median_elapsed_seconds" => median(elapsed), "minimum_elapsed_seconds" => minimum(elapsed), "maximum_elapsed_seconds" => maximum(elapsed),
            "final_objective" => objective, "final_frank_wolfe_gap" => gap,
            "final_gradient_norm" => norm(grad), "final_feasibility_violation" => violation,
            "accepted_L_max" => number(run["accepted_L_max"]),
            "accepted_L_arithmetic_mean" => number(run["accepted_L_arithmetic_mean"]),
            "accepted_L_harmonic_mean" => number(run["accepted_L_harmonic_mean"]),
            "solved" => truth(run["solved"]))
    end
    for first_method in methods, second_method in methods
        first_method == second_method && continue
        adjacent = sum(count(i -> order[i] == first_method && order[i+1] == second_method, 1:2) for order in valuesof(order_reference))
        check(adjacent == repeats÷3, "Directed predecessor orders are unbalanced: $first_method -> $second_method")
    end
    push!(notes, "Curvature traces use up to $cap accepted steps with stationarity stopping at $epsilon; an iteration cap is a valid trace endpoint, not a success claim.")
    push!(notes, "Accepted L_k is recorded before multiplying by eta; gradient norms are diagnostic, while the constrained stopping residual is the Frank-Wolfe gap.")
    push!(notes, "Two warmups, balanced repeated order, external complete-call timing and pre-call garbage collection; generation, independent checks and output writing are outside timing.")
    report = Dict("validation" => isempty(problems) ? "PASS" : "FAIL", "problem" => problem,
        "created_utc" => string(now(UTC)), "scope" => cap == 500 && repeats == 6 && warmups == 2 ? "full curvature trace" : "pilot/custom trace",
        "notes" => notes, "problems" => problems, "methods" => summaries, "input_sha256" => inputs)
    open(joinpath(directory, "analysis_report.toml"), "w") do io
        TOML.print(io, report; sorted=true)
    end
    open(joinpath(directory, "analysis_report.txt"), "w") do io
        println(io, "Validation: ", report["validation"], "\nProblem: ", problem, "\nScope: ", report["scope"])
        foreach(note -> println(io, note), notes)
        for method in methods
            r = summaries[method]
            println(io, "\n", method, ": ", join(r["status"], ", "), "; iterations=", r["iterations"],
                "; median seconds=", r["median_elapsed_seconds"], "; final f=", r["final_objective"],
                "; FW gap=", r["final_frank_wolfe_gap"], "; feasibility violation=", r["final_feasibility_violation"],
                "; accepted L mean=", r["accepted_L_arithmetic_mean"], "; harmonic mean=", r["accepted_L_harmonic_mean"])
        end
        foreach(problem -> println(io, "ERROR: ", problem), problems)
    end
    println(read(joinpath(directory, "analysis_report.txt"), String))
    return isempty(problems)
end

# Avoid shadowing Base.values with the objective-history local variable above.
valuesof(dictionary) = Base.values(dictionary)
if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("Usage: julia --project=. scripts/analyze_curvature_run.jl OUTPUT_DIRECTORY")
    exit(analyze_curvature_run(ARGS[1]) ? 0 : 1)
end
