#!/usr/bin/env julia
# Saved-data analysis only: this script never includes a problem driver or solves it.
module ACGDRemainingAnalysis

using Dates, LinearAlgebra, MAT, Printf, SHA, Statistics, TOML
include(joinpath(@__DIR__, "benchmark_support.jl"))
using .ACGDBenchmarkSupport

const COMMON_METHODS = ["Adaptive constant", "Adaptive adjustable", "Pure backtracking", "Auto-conditioned"]
const SPECS = Dict(
    "least_squares" => (rule="objective", epsilon=1e-5, methods=vcat(COMMON_METHODS, ["Short-step", "AdaBB"]), metric="gradient norm", nonconvex=false),
    "Logistic_Regression_with_Regularization" => (rule="stationarity", epsilon=1e-3, methods=vcat(COMMON_METHODS, ["Short-step", "AdaBB"]), metric="gradient norm", nonconvex=false),
    "NonconvexQP_Simplex" => (rule="stationarity", epsilon=1e-3, methods=vcat(COMMON_METHODS, ["Short-step", "Open-loop"]), metric="Frank-Wolfe gap", nonconvex=true),
    "Sigmoid_w8a" => (rule="stationarity", epsilon=1e-3, methods=vcat(COMMON_METHODS, ["Short-step", "Open-loop"]), metric="Frank-Wolfe gap", nonconvex=true),
    "matrix_balancing" => (rule="either", epsilon=1e-5, methods=vcat(COMMON_METHODS, ["Open-loop"]), metric="Frank-Wolfe gap", nonconvex=false),
    "matrix_factorization" => (rule="either", epsilon=1e-5, methods=vcat(COMMON_METHODS, ["Short-step", "Open-loop"]), metric="Frank-Wolfe gap", nonconvex=true),
    "traffic_assignment" => (rule="either", epsilon=1e-3, methods=vcat(COMMON_METHODS, ["Open-loop"]), metric="normalized Frank-Wolfe gap", nonconvex=false),
)
const TERMINAL_STATUSES = Set(["converged", "iteration_limit", "time_limit", "nonfinite", "invalid_direction",
    "zero_probe", "nonfinite_probe", "stationary_without_target", "stagnation", "adabb_zero_curvature",
    "adabb_nonpositive_curvature", "invalid_curvature", "invalid_step", "backtracking_limit"])

matname(name) = replace(name, r"[^A-Za-z0-9_]" => "_")
scalar(value) = value isa AbstractArray ? only(value) : value
number(value) = Float64(scalar(value))
integer(value) = Int(scalar(value))
truth(value) = Bool(scalar(value))
flat(value) = value isa AbstractArray ? vec(value) : [value]
numbers(value) = Float64.(flat(value))
integers(value) = Int.(flat(value))
strings(value) = strip.(String.(flat(value)))
same_number(a, b) = isequal(a, b) || (isfinite(a) && isfinite(b) && isapprox(a, b; rtol=1e-9, atol=1e-10))
same_array(a, b) = size(a) == size(b) && all(same_number(x, y) for (x, y) in zip(a, b))
empty_stat(f, values) = isempty(values) ? NaN : f(values)
file_sha256(path) = open(io -> bytes2hex(sha256(io)), path)
benchmark_file_path(directory, problem) = joinpath(directory,
    problem == "matrix_factorization" ? "matrixfactorization_benchmark_all_methods.mat" : "benchmark_all_methods.mat")
stable_sigmoid(z) = z >= 0 ? 1 / (1 + exp(-z)) : exp(z) / (1 + exp(z))
stable_softplus(z) = z > 0 ? z + log1p(exp(-z)) : log1p(exp(z))

function run_fingerprint(run)
    fields = ("seed", "instance", "values", "gaps", "times", "L_ks", "backtracks", "iterate_indices",
        "iterations", "solved", "status", "total_time", "initial_data_sha256", "repeat_times",
        "final_x", "final_U", "final_V", "final_value", "final_gap", "final_residual", "stopping_criterion",
        "gamma_history", "gamma_iterations", "representative_repeat")
    return snapshot_sha256(Dict(key => run[key] for key in fields if haskey(run, key)))
end

function smoothness_bound(problem, data)
    if problem in ("least_squares", "Logistic_Regression_with_Regularization", "Sigmoid_w8a")
        A = data["A"]
        norm2 = eigmax(Symmetric(Matrix(A' * A)))
        problem == "least_squares" && return 2norm2
        problem == "Sigmoid_w8a" && return norm2 / (4size(A, 1))
        return norm2 / (4size(A, 1)) + 2number(data["omega"])
    elseif problem == "NonconvexQP_Simplex"
        return 2maximum(abs, eigvals(Symmetric(data["Q"])))
    elseif problem == "matrix_factorization"
        a, b = number(data["tau1"]), number(data["tau2"])
        return max(a^2, b^2) + opnorm(data["Y"]) + 2a*b
    end
    return NaN # Neither matrix balancing nor traffic includes a short-step baseline.
end

"Independent DAG shortest-path costs, avoiding the driver's graph library/LMO."
function traffic_diagnostics(data, x)
    n = integer(data["mn"])
    sources, targets = integers(data["edge_sources"]), integers(data["edge_targets"])
    ds, dt, demands = integers(data["demand_sources"]), integers(data["demand_targets"]), numbers(data["demand_values"])
    c, normalization = number(data["c"]), number(data["normalization"])
    X = reshape(x, n, n)
    flows = [X[s, t] for (s, t) in zip(sources, targets)]
    costs = (1 .+ 0.15 .* flows.^4 ./ c^4) ./ normalization
    value = sum(flows .+ 0.03 .* flows.^5 ./ c^4) / normalization
    outgoing = [Int[] for _ in 1:n]
    for e in eachindex(sources)
        sources[e] < targets[e] || error("archived traffic graph is not in DAG topological order")
        push!(outgoing[sources[e]], e)
    end
    lmo_cost = 0.0
    for source in unique(ds)
        distances = fill(Inf, n)
        distances[source] = 0.0
        for vertex in source:n
            for e in outgoing[vertex]
                distances[targets[e]] = min(distances[targets[e]], distances[vertex] + costs[e])
            end
        end
        for j in eachindex(ds)
            ds[j] == source && (lmo_cost += demands[j] * distances[dt[j]])
        end
    end
    gap = dot(flows, costs) - lmo_cost
    net = zeros(n)
    for (s, t, demand) in zip(ds, dt, demands)
        net[s] += demand; net[t] -= demand
    end
    mask = falses(n, n)
    for (s, t) in zip(sources, targets)
        mask[s, t] = true
    end
    conservation = maximum(abs.(vec(sum(X; dims=2)) - vec(sum(X; dims=1)) - net))
    violation = max(0.0, -minimum(x), conservation, any(.!mask) ? maximum(abs.(X[.!mask])) : 0.0)
    return (value=value, gap=gap, feasibility=violation)
end

"Recompute objective, stationarity and feasibility using analytic saved-data formulas."
function diagnostics(problem, data, state)
    if problem == "matrix_factorization"
        U, V = state
        Y = Matrix{Float64}(data["Y"])
        tau1, tau2 = number(data["tau1"]), number(data["tau2"])
        R = U * V' - Y
        GU, GV = R * V, R' * U
        value = 0.5 * sum(abs2, R)
        # The support function of the spectral-norm ball is tau times nuclear norm.
        gap = dot(U, GU) + dot(V, GV) + tau1 * sum(svdvals(GU)) + tau2 * sum(svdvals(GV))
        violation = max(0.0, opnorm(U) - tau1, opnorm(V) - tau2)
        return (value=value, gap=gap, feasibility=violation)
    end
    x = state
    problem == "traffic_assignment" && return traffic_diagnostics(data, x)
    if problem == "least_squares"
        A, b = data["A"], numbers(data["b"])
        residual = A * x - b
        return (value=sum(abs2, residual), gap=norm(2 * A' * residual), feasibility=0.0)
    elseif problem == "Logistic_Regression_with_Regularization"
        A, y, omega = data["A"], numbers(data["y"]), number(data["omega"])
        z = -y .* (A * x)
        value = mean(stable_softplus.(z)) + omega * sum(abs2, x)
        g = A' * (-y .* stable_sigmoid.(z)) / length(y) + 2omega * x
        return (value=value, gap=norm(g), feasibility=0.0)
    elseif problem == "Sigmoid_w8a"
        A, y, tau = data["A"], numbers(data["y"]), number(data["tau"])
        predictions = stable_sigmoid.(A * x)
        residual = predictions - y
        g = A' * (2 .* residual .* predictions .* (1 .- predictions)) / length(y)
        return (value=mean(abs2, residual), gap=dot(x, g) + tau * maximum(abs, g),
                feasibility=max(0.0, norm(x, 1) - tau))
    elseif problem == "NonconvexQP_Simplex"
        Q = data["Q"]
        g = 2 * Q * x
        return (value=dot(x, Q * x), gap=dot(g, x) - minimum(g),
                feasibility=max(0.0, -minimum(x), abs(sum(x) - 1)))
    elseif problem == "matrix_balancing"
        A, l, u = data["A"], number(data["l"]), number(data["u"])
        weighted = A .* exp.(x .- x')
        g = vec(sum(weighted; dims=2)) - vec(sum(weighted; dims=1))
        v = ifelse.(g .> 0, l, u)
        return (value=sum(weighted), gap=dot(g, x - v),
                feasibility=max(0.0, l - minimum(x), maximum(x) - u))
    end
    error("unsupported problem: $problem")
end

function validate_instance(check, problem, data, seed)
    prefix = "seed $seed"
    check(integer(data["seed"]) == seed, "$prefix: snapshot seed differs")
    if problem in ("least_squares", "Logistic_Regression_with_Regularization", "Sigmoid_w8a")
        expected = problem == "least_squares" ? (2000, 200) : problem == "Sigmoid_w8a" ? (2000, 300) : (2000, 123)
        check(size(data["A"]) == expected && length(data["x0"]) == expected[2], "$prefix: incorrect $problem instance dimensions")
        if problem == "least_squares"
            A, b = data["A"], numbers(data["b"])
            optimum = sum(abs2, A * (A \ b) - b)
            check(same_number(number(data["optimal_value"]), optimum), "$prefix: least-squares optimum differs from independent QR solve")
        elseif problem == "Sigmoid_w8a"
            check(number(data["tau"]) == 50.0 && all(y -> y in (0, 1), numbers(data["y"])), "$prefix: invalid sigmoid labels/constraint")
        else
            check(all(y -> y in (-1, 1), numbers(data["y"])), "$prefix: invalid logistic labels")
            check(same_number(number(data["omega"]), 1 / prod(expected)), "$prefix: incorrect logistic regularization")
        end
    elseif problem == "NonconvexQP_Simplex"
        check(size(data["Q"]) == (700, 700) && length(data["x0"]) == 700, "$prefix: StQO must have 700 variables")
        check(number(data["dens"]) == 0.75 && number(data["dvert"]) == 2.0, "$prefix: incorrect StQO generator parameters")
    elseif problem == "matrix_balancing"
        A = data["A"]
        check(size(A) == (100, 100) && length(data["x0"]) == 100, "$prefix: matrix balancing must have dimension 100")
        check(number(data["l"]) == 1.0 && number(data["u"]) == 10.0, "$prefix: incorrect matrix balancing box")
        check(same_array(A, A') && minimum(A) >= 0, "$prefix: matrix balancing requires symmetric nonnegative A")
        check(same_number(number(data["optimal_value"]), sum(A)), "$prefix: matrix balancing optimum differs from sum(A)")
    elseif problem == "matrix_factorization"
        check(size(data["Y"]) == (400, 300) && size(data["U0"]) == (400, 20) && size(data["V0"]) == (300, 20), "$prefix: incorrect matrix-factorization dimensions")
        check(number(data["optimal_value"]) == 0.0, "$prefix: matrix factorization optimum must be zero")
        planted = diagnostics(problem, data, (data["U_star"], data["V_star"]))
        check(planted.value <= 1e-20 && planted.feasibility <= 1e-9, "$prefix: planted optimum is not feasible/exact")
    elseif problem == "traffic_assignment"
        n = integer(data["mn"])
        check(n == 150 && length(data["x0"]) == n^2 && number(data["normalization"]) == n^2, "$prefix: traffic dimensions/normalization differ from 150 nodes and |V|^2")
        check(integer(data["m"]) == 15 && integer(data["n"]) == 10 && number(data["p"]) == 0.4, "$prefix: incorrect traffic generator parameters")
        check(length(data["edge_sources"]) == length(data["edge_targets"]) == integer(data["n_links"]), "$prefix: traffic edge arrays disagree")
        check(length(data["demand_sources"]) == length(data["demand_targets"]) == length(data["demand_values"]) == integer(data["n_od"]), "$prefix: traffic demand arrays disagree")
        check(same_number(number(data["c"]), integer(data["n_od"]) / n), "$prefix: traffic capacity differs from OD count / node count")
    end
    state = problem == "matrix_factorization" ? (Matrix{Float64}(data["U0"]), Matrix{Float64}(data["V0"])) : numbers(data["x0"])
    initial = diagnostics(problem, data, state)
    check(isfinite(initial.value) && isfinite(initial.gap) && initial.feasibility <= 1e-8, "$prefix: invalid or infeasible initial state")
    expected_L = smoothness_bound(problem, data)
    if isfinite(expected_L)
        field = problem == "matrix_factorization" ? "L_global" : problem == "Sigmoid_w8a" ? "L" : "L_estimate"
        check(same_number(number(data[field]), expected_L), "$prefix: archived smoothness bound differs from independent formula")
    end
    return merge(initial, (L=expected_L,))
end

function expected_criterion(value, gap, fstar, epsilon, rule)
    if rule != "stationarity" && isfinite(fstar) && value - fstar <= epsilon
        return "objective", value - fstar
    elseif rule != "objective" && isfinite(gap) && gap <= epsilon
        return "stationarity", gap
    end
    return "none", rule == "objective" ? value - fstar : gap
end

"Validate a completed saved benchmark and produce concise text plus detailed TOML."
function analyze_remaining_run(output_directory)
    directory = abspath(output_directory)
    manifest_path = joinpath(directory, "run_manifest.toml")
    manifest = TOML.parsefile(manifest_path)
    config = manifest["config"]
    problem = String(config["problem"])
    haskey(SPECS, problem) || error("Unsupported remaining benchmark: $problem")
    spec = SPECS[problem]
    methods, seeds = String.(config["method_order"]), Int.(config["seeds"])
    epsilon, cap = Float64(config["epsilon"]), Int(config["max_iter_cap"])
    repeats, warmups = Int(config["repeats"]), Int(config["warmups"])
    issues, notes, pilot_reasons = String[], String[], String[]
    check(condition, message) = condition || (push!(issues, message); false)
    check(!isempty(seeds) && length(unique(seeds)) == length(seeds), "manifest seeds are empty or duplicated")
    check(length(methods) == length(spec.methods) && Set(methods) == Set(spec.methods), "method set differs from the intended benchmark")
    check(config["stop_rule"] == spec.rule && epsilon == spec.epsilon, "stopping rule/tolerance differs from the documented problem")
    check(Float64(config["max_time"]) == Inf, "all methods must have the same unlimited time limit")
    check(warmups >= 1 && repeats >= 1 && cap >= 0, "invalid run controls")
    for (key, expected) in (("gamma", 0.25), ("beta", 2.0), ("eta", 0.9), ("block_size", 10), ("delta", 1e-10), ("probe_seed", 23))
        check(get(config, key, NaN) == expected, "manifest $key differs from the intended configuration")
    end
    "AdaBB" in methods && check(get(config, "alpha0", NaN) == 1e-10, "AdaBB initial step differs from 1e-10")
    check(manifest["julia_threads"] == 1 && manifest["blas_threads"] == 1, "Julia/BLAS thread counts differ from the one-thread protocol")
    check(occursin(r"^[0-9a-f]{64}$", String(manifest["manifest_sha256"])) && !isempty(manifest["package_versions"]), "resolved package environment is not archived")
    for filename in (String(config["script"]), "src/ACGDSolvers.jl", "scripts/benchmark_support.jl", "Project.toml", "Manifest.toml")
        check(haskey(manifest["source_sha256"], filename) && occursin(r"^[0-9a-f]{64}$", String(manifest["source_sha256"][filename])), "missing source checksum for $filename")
    end
    expected_repeats = iseven(length(methods)) ? length(methods) : 2length(methods)
    check(repeats % expected_repeats == 0, "repeat count cannot give position and directed-neighbor balance")
    seeds == collect(1:100) || push!(pilot_reasons, "seed set differs from 1:100")
    cap == 100000 || push!(pilot_reasons, "iteration cap differs from 100000")
    repeats == expected_repeats || push!(pilot_reasons, "repeat count differs from $expected_repeats")
    warmups == 2 || push!(pilot_reasons, "warmup count differs from two")
    scope = isempty(pilot_reasons) ? "full" : "pilot"

    benchmark_path, profile_path = benchmark_file_path(directory, problem), joinpath(directory, "profile_statistics.mat")
    benchmark, saved_profile = matread(benchmark_path), matread(profile_path)
    input_hashes = Dict(basename(path) => file_sha256(path) for path in (manifest_path, benchmark_path, profile_path))
    check(integers(benchmark["seeds"]) == seeds, "benchmark seeds differ from manifest")
    check(Set(keys(benchmark["times"])) == Set(matname.(methods)) && Set(keys(benchmark["solved"])) == Set(matname.(methods)), "benchmark method keys differ from manifest")
    times = hcat([numbers(benchmark["times"][matname(method)]) for method in methods]...)
    check(size(times) == (length(seeds), length(methods)), "timing matrix dimensions differ from manifest")
    instances = Dict{Int,Any}()
    for seed in seeds
        path = joinpath(directory, "instances", "seed_$seed.mat")
        input_hashes[relpath(path, directory)] = file_sha256(path)
        header = matopen(path) do file
            runs = read(file, "runs")
            (data=read(file, "problem"), input_hash=String(read(file, "initial_data_sha256")),
             seed=integer(read(file, "seed")), methods=strings(read(file, "method_order")),
             fingerprints=Dict(key => run_fingerprint(run) for (key, run) in runs))
        end
        check(header.seed == seed && header.methods == methods, "seed $seed: checkpoint seed/method order differs")
        hash = snapshot_sha256(header.data)
        check(hash == header.input_hash && occursin(r"^[0-9a-f]{64}$", hash), "seed $seed: input snapshot checksum mismatch")
        initial = validate_instance(check, problem, header.data, seed)
        instances[seed] = (data=header.data, input_hash=hash, initial=initial, fingerprints=header.fingerprints)
    end
    summaries, per_seed = Dict{String,Any}(), Dict{String,Any}()
    execution_orders = Dict{Tuple{Int,Int},Vector{String}}()
    for (column, method) in enumerate(methods)
        history_path = joinpath(directory, "$(problem)_iterations_$(matname(method)).mat")
        input_hashes[basename(history_path)] = file_sha256(history_path)
        history = matread(history_path)
        check(String(history["method"]) == method && integers(history["seeds"]) == seeds, "$method: history header differs")
        check(Set(keys(history["runs"])) == Set("seed_$seed" for seed in seeds), "$method: history seed keys differ")
        solved_flags = Bool.(flat(benchmark["solved"][matname(method)]))
        elapsed_all, iterations_all, values_all, gaps_all, subopt_all = Float64[], Int[], Float64[], Float64[], Float64[]
        variability, gc_fractions, overheads, allocations, feasible = Float64[], Float64[], Float64[], Float64[], Float64[]
        l_maxima, l_means, l_harmonics = Float64[], Float64[], Float64[]
        status_counts, failure_counts = Dict{String,Int}(), Dict{String,Int}()
        details = Dict{String,Any}()
        for (row, seed) in enumerate(seeds)
            prefix = "$method / seed $seed"
            run, instance = history["runs"]["seed_$seed"], instances[seed]
            data = instance.data
            check(run_fingerprint(run) == instance.fingerprints[matname(method)], "$prefix: final history differs from its per-instance checkpoint")
            k, fstar, elapsed = integer(run["iterations"]), number(run["fstar"]), number(run["total_time"])
            values, gaps, indices = numbers(run["values"]), numbers(run["gaps"]), integers(run["iterate_indices"])
            solved, repeat_times = truth(run["solved"]), numbers(run["repeat_times"])
            check(integer(run["seed"]) == seed && integer(run["instance"]) == row, "$prefix: seed/instance index mismatch")
            check(String(run["initial_data_sha256"]) == instance.input_hash, "$prefix: input checksum differs from snapshot")
            check(same_number(fstar, number(data["optimal_value"])), "$prefix: stopping target differs from archived problem")
            check(0 <= k <= cap && indices == collect(0:k), "$prefix: iteration count/indices violate common cap")
            check(length(values) == length(gaps) == length(indices) == k + 1, "$prefix: objective/residual/index histories are misaligned")
            check(integer(run["termination_iteration"]) == k, "$prefix: termination index differs from final iterate")
            check(same_number(first(values), instance.initial.value) && same_number(first(gaps), instance.initial.gap), "$prefix: initial objective/residual differs from paired input")
            check(same_number(number(run["final_value"]), last(values)) && same_number(number(run["final_gap"]), last(gaps)), "$prefix: final values differ from histories")
            check(same_number(number(run["final_subopt"]), last(values) - fstar), "$prefix: incorrect final objective-minus-target")
            for (field, expected) in (("independently_verified_final_value", last(values)), ("independently_verified_final_gap", last(gaps)))
                check(haskey(run, field) && same_number(number(run[field]), expected), "$prefix: independent driver $field disagrees")
            end
            final_state = problem == "matrix_factorization" ? (Matrix{Float64}(run["final_U"]), Matrix{Float64}(run["final_V"])) : numbers(run["final_x"])
            finite_state = problem == "matrix_factorization" ? all(all(isfinite, part) for part in final_state) : all(isfinite, final_state)
            solved && check(finite_state, "$prefix: solved final iterate is nonfinite")
            final = finite_state ? diagnostics(problem, data, final_state) : (value=NaN, gap=NaN, feasibility=NaN)
            check(same_number(final.value, last(values)) && same_number(final.gap, last(gaps)), "$prefix: objective/residual recomputed from archived final iterate disagrees")
            feasibility_key = problem == "traffic_assignment" ? "final_aggregate_flow_violation" : "final_feasibility_violation"
            violation = number(run[feasibility_key])
            check(same_number(final.feasibility, violation), "$prefix: independent feasibility diagnostic disagrees")
            finite_state && check(isfinite(violation) && 0 <= violation <= 1e-7, "$prefix: final feasibility check fails")
            push!(feasible, violation)
            check(integer(run["repeats"]) == repeats && integer(run["warmups"]) == warmups && String(run["aggregation"]) == "median", "$prefix: aggregation/repeat controls differ")
            outcomes = run["repeat_outcomes"]
            check(length(repeat_times) == repeats && Set(keys(outcomes)) == Set("repeat_$i" for i in 1:repeats), "$prefix: missing/extra repeats")
            raw = [outcomes["repeat_$i"] for i in 1:repeats]
            check(all(t -> isfinite(t) && t > 0, repeat_times), "$prefix: raw elapsed time is nonpositive/nonfinite")
            check(same_number(elapsed, median(repeat_times)) && same_number(number(run["elapsed_time"]), elapsed), "$prefix: complete elapsed time is not repeat median")
            check(solved == all(r -> truth(r["solved"]), raw) == solved_flags[row], "$prefix: aggregate success must require every repeat")
            check(solved ? same_number(times[row, column], elapsed) : times[row, column] == Inf, "$prefix: profile time must be successful elapsed or +Inf on failure")
            check(String(run["status"]) == (solved ? "converged" : "repeat_failure"), "$prefix: incorrect aggregate status")
            policy = run["measurement_policy"]
            check(String(policy["clock"]) == "external_complete_call" && truth(policy["external_timing"]) && truth(policy["gc_before_call"]) && truth(policy["gc_inside_call_included"]) && truth(policy["compilation_inside_call_included"]), "$prefix: complete-call/GC measurement convention differs")
            check(String(policy["order_mode"]) == "balanced" && integer(policy["instance_index"]) == row && !truth(policy["measured_order_depends_on_warmup_count"]), "$prefix: execution-order policy differs")
            check(Set(keys(policy["warmup_orders"])) == Set("warmup_$i" for i in 1:warmups) && Set(keys(policy["measured_orders"])) == Set("repeat_$i" for i in 1:repeats), "$prefix: warmup/measured execution orders missing or extra")
            for order in valuesof(policy["warmup_orders"])
                check(length(strings(order)) == length(methods) && Set(strings(order)) == Set(methods), "$prefix: invalid warmup method order")
            end
            for field in ("iteration_time", "initialization_time", "outer_elapsed_time", "core_elapsed_time", "gc_time", "allocated_bytes", "compile_time", "recompile_time")
                check(same_number(number(run[field]), median(number(r[field]) for r in raw)), "$prefix: incorrect median $field")
            end
            check(same_array(numbers(run["repeat_core_times"]), [number(r["core_elapsed_time"]) for r in raw]) && same_array(numbers(run["repeat_outer_times"]), repeat_times), "$prefix: saved timing vectors differ from repeats")
            for (repeat, r) in enumerate(raw)
                tag = "$prefix / repeat $repeat"
                status, criterion = String(r["status"]), String(r["stopping_criterion"])
                value, gap = number(r["final_value"]), number(r["final_gap"])
                expected_kind, expected_residual = expected_criterion(value, gap, fstar, epsilon, spec.rule)
                check(status in TERMINAL_STATUSES && truth(r["solved"]) == (status == "converged"), "$tag: unrecognized/inconsistent terminal status")
                startswith(status, "adabb_") && check(method == "AdaBB", "$tag: AdaBB failure on another method")
                check(integer(r["repeat"]) == repeat && 0 <= integer(r["iterations"]) == integer(r["termination_iteration"]) <= cap, "$tag: repeat/termination index mismatch")
                check(criterion == expected_kind && same_number(number(r["final_residual"]), expected_residual), "$tag: saved stopping residual/criterion differs from rule")
                if truth(r["solved"])
                    check(criterion != "none" && isfinite(expected_residual) && -1e-8 <= expected_residual <= epsilon, "$tag: success fails tolerance")
                    check(isfinite(value) && isfinite(gap) && gap >= -1e-8, "$tag: invalid successful objective/stationarity")
                end
                status == "iteration_limit" && check(integer(r["iterations"]) == cap, "$tag: iteration_limit before cap")
                check(status != "time_limit", "$tag: time-limit termination despite unlimited budget")
                for field in ("status", "stopping_criterion", "iterations", "termination_iteration", "final_value", "final_gap", "final_residual")
                    a, b = r[field], raw[1][field]
                    check(a isa AbstractString ? String(a) == String(b) : same_number(number(a), number(b)), "$tag: deterministic repeated outcomes disagree ($field)")
                end
                outer, core = number(r["elapsed_time"]), number(r["core_elapsed_time"])
                check(same_number(outer, repeat_times[repeat]) && same_number(number(r["outer_elapsed_time"]), outer), "$tag: outer timing disagrees")
                check(0 <= core <= outer + 1e-10 && number(r["initialization_time"]) >= 0 && number(r["iteration_time"]) >= 0 && number(r["initialization_time"]) + number(r["iteration_time"]) <= core + 1e-10, "$tag: invalid timing component boundaries")
                check(0 <= number(r["gc_time"]) <= outer + 1e-10 && isfinite(number(r["allocated_bytes"])) && number(r["allocated_bytes"]) >= 0, "$tag: invalid GC/allocation statistics")
                check(number(r["compile_time"]) == 0 && number(r["recompile_time"]) == 0, "$tag: compilation occurred inside measured call")
                push!(gc_fractions, number(r["gc_time"]) / outer); push!(overheads, (outer - core) / outer); push!(allocations, number(r["allocated_bytes"]))
                order, position = strings(r["execution_order"]), integer(r["execution_position"])
                check(length(order) == length(methods) && Set(order) == Set(methods) && 1 <= position <= length(order) && order[position] == method, "$tag: execution order/position invalid")
                check(order == strings(policy["measured_orders"]["repeat_$repeat"]), "$tag: raw execution order differs from policy")
                if haskey(execution_orders, (seed, repeat))
                    check(order == execution_orders[(seed, repeat)], "$tag: methods disagree on actual execution order")
                else
                    execution_orders[(seed, repeat)] = order
                end
                status_counts[status] = get(status_counts, status, 0) + 1
            end
            check(strings(run["repeat_statuses"]) == [String(r["status"]) for r in raw], "$prefix: repeat status vector differs")
            failures = sort!(unique(String(r["status"]) for r in raw if !truth(r["solved"])))
            for reason in failures
                failure_counts[reason] = get(failure_counts, reason, 0) + 1
            end
            representative = integer(run["representative_repeat"])
            check(representative == argmin(abs.(repeat_times .- elapsed)), "$prefix: representative is not nearest median")
            selected = raw[representative]
            for field in ("final_value", "final_gap", "final_residual", "termination_iteration")
                check(same_number(number(run[field]), number(selected[field])), "$prefix: $field differs from representative")
            end
            check(String(run["stopping_criterion"]) == String(selected["stopping_criterion"]), "$prefix: representative stopping criterion differs")
            trace_times = numbers(run["times"])
            check(all(t -> isfinite(t) && t >= 0, trace_times) && same_number(first(trace_times), number(selected["initialization_time"])) && same_number(sum(trace_times[2:end]), number(selected["iteration_time"])), "$prefix: representative attempt times do not match components")
            check(sum(trace_times) <= number(selected["core_elapsed_time"]) + 1e-10, "$prefix: attempts exceed representative core elapsed")
            curvatures, backtracks = numbers(run["L_ks"]), integers(run["backtracks"])
            check(length(curvatures) == (method == "Open-loop" ? 0 : k) && all(L -> isfinite(L) && L > 0, curvatures), "$prefix: invalid accepted-curvature history")
            method == "Short-step" && check(all(L -> same_number(L, instance.initial.L), curvatures), "$prefix: short-step curvature differs from independently verified smoothness bound")
            check(length(backtracks) == k && all(b -> b >= 0, backtracks), "$prefix: rejected-trial history invalid")
            maxima, average, harmonic = empty_stat(maximum, curvatures), empty_stat(mean, curvatures), harmonic_mean(curvatures)
            if method != "AdaBB"
                for (key, expected) in (("accepted_L_max", maxima), ("accepted_L_arithmetic_mean", average), ("accepted_L_harmonic_mean", harmonic))
                    check(same_number(number(run[key]), expected), "$prefix: incorrect $key")
                end
            end
            if method == "Adaptive adjustable"
                boundaries, history_gamma = integers(run["gamma_iterations"]), numbers(run["gamma_history"])
                check(boundaries == collect(0:10:k) && length(boundaries) == length(history_gamma), "$prefix: adjustable blocks must end at 10,20,...")
                expected_gamma = Float64[Float64(config["gamma"])]
                for endpoint in 10:10:k
                    rejected, previous = sum(backtracks[endpoint-9:endpoint]), last(expected_gamma)
                    push!(expected_gamma, rejected == 0 ? 0.9previous : rejected > 10 ? min(1.0, 1.1previous) : previous)
                end
                check(same_array(history_gamma, expected_gamma), "$prefix: adjustable counter duplicates/reuses rejected trials")
            end
            push!(elapsed_all, elapsed); push!(iterations_all, k); push!(values_all, last(values)); push!(gaps_all, last(gaps)); push!(subopt_all, last(values) - fstar)
            push!(variability, maximum(repeat_times) / minimum(repeat_times)); push!(l_maxima, maxima); push!(l_means, average); push!(l_harmonics, harmonic)
            details["seed_$seed"] = Dict("solved" => solved, "elapsed_median_seconds" => elapsed, "iterations" => k,
                "final_value" => last(values), "final_gap" => last(gaps), "final_objective_minus_target" => last(values)-fstar,
                "target_value" => fstar, "failure_reasons" => failures, "repeat_max_min_ratio" => last(variability),
                "initial_data_sha256" => instance.input_hash, "representative_feasibility_violation" => violation)
        end
        summaries[method] = Dict(
            "solved" => count(solved_flags), "instances" => length(seeds), "success_rate" => mean(solved_flags),
            "mean_seconds_successes" => empty_stat(mean, elapsed_all[solved_flags]), "median_seconds_successes" => empty_stat(median, elapsed_all[solved_flags]),
            "median_seconds_all_outcomes" => median(elapsed_all), "median_iterations_successes" => empty_stat(median, iterations_all[solved_flags]),
            "median_repeat_max_min_ratio" => median(variability), "maximum_repeat_max_min_ratio" => maximum(variability),
            "median_final_objective_all_outcomes" => median(values_all), "median_final_gap_all_outcomes" => median(gaps_all),
            "median_final_objective_successes" => empty_stat(median, values_all[solved_flags]), "median_final_gap_successes" => empty_stat(median, gaps_all[solved_flags]),
            "median_objective_minus_target_successes" => empty_stat(median, subopt_all[solved_flags]),
            "median_accepted_L_max" => empty_stat(median, filter(isfinite, l_maxima)),
            "median_accepted_L_arithmetic_mean" => empty_stat(median, filter(isfinite, l_means)),
            "median_accepted_L_harmonic_mean" => empty_stat(median, filter(isfinite, l_harmonics)),
            "L_interpretation" => method == "AdaBB" ? "reciprocal step size 1/alpha" : method == "Open-loop" ? "not applicable" : "accepted curvature before eta decay",
            "median_external_overhead_fraction" => median(overheads), "median_gc_fraction" => median(gc_fractions), "median_allocated_bytes" => median(allocations),
            "maximum_feasibility_violation" => maximum(feasible), "raw_repeat_status_counts" => status_counts, "failed_seed_reason_counts" => failure_counts)
        per_seed[method] = details
    end

    for seed in seeds
        positions = Dict((method, p) => 0 for method in methods for p in eachindex(methods))
        neighbors = Dict((a, b) => 0 for a in methods for b in methods if a != b)
        for repeat in 1:repeats
            order = execution_orders[(seed, repeat)]
            for (p, method) in enumerate(order)
                positions[(method, p)] += 1
            end
            for p in 1:length(order)-1
                neighbors[(order[p], order[p+1])] += 1
            end
        end
        check(all(==(repeats ÷ length(methods)), valuesof(positions)), "seed $seed: method positions are not balanced")
        check(all(==(repeats ÷ length(methods)), valuesof(neighbors)), "seed $seed: directed neighboring pairs are not balanced")
    end
    profile = finite_performance_profile(times; logscale=false)
    check(strings(saved_profile["methods"]) == methods, "profile method order differs")
    for (field, expected) in (("ratios", profile.ratios), ("thresholds", profile.thresholds), ("fractions", profile.fractions), ("success_rates", profile.success_rates), ("finite_ratio_coverage", profile.finite_ratio_coverage))
        actual = expected isa AbstractVector ? numbers(saved_profile[field]) : Float64.(saved_profile[field])
        check(same_array(actual, expected), "profile $field differs from saved successful timings/failure handling")
    end
    check(integer(saved_profile["all_failed_instances"]) == profile.all_failed_instances && same_number(number(saved_profile["max_ratio"]), profile.max_ratio), "profile failed count/max ratio differs")
    comparisons = Dict{String,Any}()
    for reference in COMMON_METHODS[1:2]
        paired = paired_speed_ratios(times; reference=findfirst(==(reference), methods))
        saved = saved_profile["paired_speed_comparisons"][matname(reference)]
        comparisons[reference] = Dict{String,Any}()
        for (column, method) in enumerate(methods)
            p, archived = paired[column], saved[matname(method)]
            check(integer(archived["paired_instances"]) == p.paired_instances && same_number(number(archived["baseline_over_reference_harmonic"]), p.baseline_over_reference_harmonic), "$reference/$method: paired denominator/speed ratio differs")
            comparisons[reference][method] = Dict("paired_instances" => p.paired_instances, "baseline_over_reference_harmonic" => p.baseline_over_reference_harmonic, "baseline_over_reference_median" => empty_stat(median, p.ratios))
        end
    end
    winners = profile.ratios .== 1
    split_wins = zeros(length(methods))
    for row in axes(winners, 1)
        selected = findall(winners[row, :])
        isempty(selected) || (split_wins[selected] .+= 1 / length(selected))
    end
    for (column, method) in enumerate(methods)
        summaries[method]["fastest_including_ties_count"] = count(winners[:, column])
        summaries[method]["fastest_including_ties_share"] = mean(winners[:, column])
        summaries[method]["fastest_split_ties_share"] = split_wins[column] / length(seeds)
        summaries[method]["finite_ratio_coverage"] = profile.finite_ratio_coverage[column]
    end
    push!(notes, "Successful timing summaries exclude failures. All-outcome elapsed medians include time to failure and are not time-to-solution comparisons.")
    push!(notes, "The profile uses raw ratios on a logarithmic axis; all instances remain in the denominator and failed methods retain infinite ratios. Finite-ratio coverage and overall success rates are distinct saved fields.")
    push!(notes, "Paired speedups use only instances solved by both methods; values above one favor the named adaptive method. No historical/cross-hardware speedup is inferred.")
    push!(notes, "Input hashes bind numeric problem data and starting points. Final objective, stationarity metric and feasibility were recomputed independently from saved iterates, outside all solver timers.")
    push!(notes, "Stopping rule: $(spec.rule); tolerance=$(spec.epsilon); stationarity diagnostic=$(spec.metric). For either, objective accuracy OR stationarity suffices.")
    spec.nonconvex && push!(notes, "This is nonconvex: attainment of a stationarity tolerance does not establish a common attained objective or global optimality. Per-seed final objective and stationarity are reported separately.")
    problem == "traffic_assignment" && push!(notes, "Traffic objective and FW gap are divided by |V|^2=22500. Zero is a lower bound, not an established optimum. Feasibility diagnostics check aggregate conservation, nonnegativity and absent-edge flow only; they do not independently certify commodity-wise decomposability.")
    "AdaBB" in methods && push!(notes, "AdaBB zero/nonpositive curvature quotients are explicit failed runs; their early failure times are excluded from successful timing comparisons.")
    report = Dict{String,Any}("problem" => problem, "scope" => scope, "pilot_reasons" => pilot_reasons,
        "validation_passed" => isempty(issues), "validation_errors" => issues, "analyzed_utc" => string(now(UTC)),
        "output_directory" => directory, "seeds" => seeds, "methods" => methods, "config" => config,
        "source_revision" => manifest["source_revision"], "source_dirty" => manifest["source_dirty"],
        "julia_version" => manifest["julia_version"], "julia_threads" => manifest["julia_threads"], "blas_threads" => manifest["blas_threads"],
        "input_sha256" => input_hashes, "summary" => summaries, "per_seed" => per_seed, "paired_speedups" => comparisons,
        "all_failed_instances" => profile.all_failed_instances, "profile_max_finite_ratio" => profile.max_ratio, "notes" => notes)
    open(joinpath(directory, "analysis_report.toml"), "w") do io
        TOML.print(io, report; sorted=true)
    end
    io = IOBuffer()
    println(io, "$problem $(uppercase(scope)) run: $(length(seeds)) instances; $warmups warmups, $repeats balanced measured repeats; median complete-call elapsed time.")
    println(io, "Validation: ", isempty(issues) ? "PASS" : "FAIL ($(length(issues)) issues)")
    isempty(pilot_reasons) || println(io, "Pilot configuration: ", join(pilot_reasons, "; "))
    println(io, "Julia $(manifest["julia_version"]); Julia/BLAS threads $(manifest["julia_threads"])/$(manifest["blas_threads"]).")
    @printf(io, "%-22s %9s %12s %12s %12s %10s\n", "Method", "Solved", "Median sec", "Median iter", "Repeat max/min", "Fastest %")
    for method in methods
        s = summaries[method]
        @printf(io, "%-22s %9s %12s %12s %12.3f %10.1f\n", method, "$(s["solved"])/$(s["instances"])",
            s["solved"] == 0 ? "N/A" : @sprintf("%.6g", s["median_seconds_successes"]),
            s["solved"] == 0 ? "N/A" : @sprintf("%.1f", s["median_iterations_successes"]), s["median_repeat_max_min_ratio"], 100s["fastest_including_ties_share"])
    end
    println(io, "\nComplete-call timing includes in-call GC; measured compilation/recompilation must be zero. Initialization, probes, stopping and in-memory history are included; data generation, warmup, pre-call GC, analysis, plotting and file writes are excluded.")
    println(io, "\nRepresentative final diagnostics across ALL outcomes (including failures):")
    @printf(io, "%-22s %14s %14s %14s %14s\n", "Method", "Median f", "Median gap", "Max feas viol", "Median GC %")
    for method in methods
        s = summaries[method]
        @printf(io, "%-22s %14.6g %14.6g %14.6g %14.6g\n", method, s["median_final_objective_all_outcomes"], s["median_final_gap_all_outcomes"], s["maximum_feasibility_violation"], 100s["median_gc_fraction"])
        println(io, "  Raw repeat terminal statuses: ", join(["$status=$count" for (status, count) in sort!(collect(s["raw_repeat_status_counts"]); by=first)], ", "))
        isempty(s["failed_seed_reason_counts"]) || println(io, "  Failed instances by reason: ", join(["$status=$count" for (status, count) in sort!(collect(s["failed_seed_reason_counts"]); by=first)], ", "))
    end
    println(io, "\nAccepted-L summaries (medians of representative per-instance histories; AdaBB uses 1/alpha, Open-loop has no L):")
    @printf(io, "%-22s %14s %14s %14s %14s\n", "Method", "Max L", "Mean L", "Harmonic L", "Finite cover %")
    for method in methods
        s = summaries[method]
        @printf(io, "%-22s %14.6g %14.6g %14.6g %14.2f\n", method, s["median_accepted_L_max"], s["median_accepted_L_arithmetic_mean"], s["median_accepted_L_harmonic_mean"], 100s["finite_ratio_coverage"])
    end
    println(io, "\nPaired speedups (baseline / adaptive time; values above 1 favor the named adaptive method):")
    for reference in COMMON_METHODS[1:2]
        println(io, reference, ":")
        for method in methods
            method == reference && continue
            p = comparisons[reference][method]
            if p["paired_instances"] == 0
                println(io, "  $method: N/A (0/$(length(seeds)) paired successes)")
            else
                @printf(io, "  %-22s median=%9.4g harmonic=%9.4g pairs=%d/%d\n", method, p["baseline_over_reference_median"], p["baseline_over_reference_harmonic"], p["paired_instances"], length(seeds))
            end
        end
    end
    println(io, "\nAll-method failures: $(profile.all_failed_instances)/$(length(seeds)).")
    foreach(note -> println(io, note), notes)
    foreach(issue -> println(io, "ERROR: ", issue), issues)
    rendered = String(take!(io))
    write(joinpath(directory, "analysis_report.txt"), rendered)
    print(rendered)
    return report
end

# Avoid local variables named `values` hiding Base.values inside the audit.
valuesof(dictionary) = Base.values(dictionary)

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: julia --project=. scripts/analyze_remaining_run.jl OUTPUT_DIRECTORY")
    try
        report = ACGDRemainingAnalysis.analyze_remaining_run(only(ARGS))
        report["validation_passed"] || exit(1)
    catch error
        directory = abspath(only(ARGS))
        message = sprint(showerror, error, catch_backtrace())
        if isdir(directory)
            write(joinpath(directory, "analysis_report.txt"), "Validation: FAIL (analysis could not finish)\n$message\n")
            open(joinpath(directory, "analysis_report.toml"), "w") do io
                ACGDRemainingAnalysis.TOML.print(io, Dict("validation_passed" => false, "validation_errors" => [message], "output_directory" => directory))
            end
        end
        println(stderr, message)
        exit(1)
    end
end
