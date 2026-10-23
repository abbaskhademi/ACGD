#!/usr/bin/env julia
# Postprocessing only: never includes a problem driver or runs a solver.
using Dates, LinearAlgebra, MAT, Printf, SHA, Statistics, TOML
include(joinpath(@__DIR__, "benchmark_support.jl"))
using .ACGDBenchmarkSupport

const POWER_METHODS = ["Adaptive constant", "Adaptive adjustable", "Pure backtracking",
                       "Auto-conditioned", "Short-step", "AdaBB"]
matname(name) = replace(name, r"[^A-Za-z0-9_]" => "_")
scalar(value) = value isa AbstractArray ? only(value) : value
number(value) = Float64(scalar(value))
integer(value) = Int(scalar(value))
truth(value) = Bool(scalar(value))
flat(value) = value isa AbstractArray ? vec(value) : [value]
numbers(value) = Float64.(flat(value))
integers(value) = Int.(flat(value))
# MAT string vectors may round-trip as fixed-width, space-padded character rows.
strings(value) = strip.(String.(flat(value)))
same_number(a, b) = isequal(a, b) || (isfinite(a) && isfinite(b) && isapprox(a, b; rtol=1e-10, atol=1e-12))
same_array(a, b) = size(a) == size(b) && all(same_number(x, y) for (x, y) in zip(a, b))
empty_stat(function_, values) = isempty(values) ? NaN : function_(values)
file_sha256(path) = open(io -> bytes2hex(sha256(io)), path)

"Validate saved power-shifted quadratic results and write separate human/machine-readable reports."
function analyze_power_run(output_directory)
    directory = abspath(output_directory)
    isdir(directory) || throw(ArgumentError("output directory does not exist: $directory"))
    manifest_path = joinpath(directory, "run_manifest.toml")
    manifest = TOML.parsefile(manifest_path)
    config = manifest["config"]
    methods = String.(config["method_order"])
    seeds = Int.(config["seeds"])
    epsilon = Float64(config["epsilon"])
    cap = Int(config["max_iter_cap"])
    repeats = Int(config["repeats"])
    warmups = Int(config["warmups"])
    problem = String(config["problem"])
    problems = String[]
    notes = String[]
    function check(condition, message)
        condition || push!(problems, message)
    end
    check(problem in ("power_quadratic", "Power_Shifted_Quadratic"), "manifest problem is not power-shifted quadratic: $problem")
    check(length(seeds) > 0 && length(unique(seeds)) == length(seeds), "manifest seeds must be nonempty and unique")
    check(length(methods) == 6 && Set(methods) == Set(POWER_METHODS), "expected all six power-shifted quadratic methods exactly once")
    check(epsilon == 1e-5, "Power-shifted quadratic tolerance differs from the documented 1e-5")
    check(config["stop_rule"] == "objective", "Power-shifted quadratic must use objective-only stopping")
    check(warmups >= 1 && repeats >= 1, "invalid warmup/repeat count")
    check(get(config, "dimension", get(config, "n", 0)) == 100, "instance dimension must be archived as 100")
    check(Float64(config["max_time"]) == Inf, "expected the common unlimited wall-time limit")
    for (key, expected) in (("gamma", 0.25), ("beta", 2.0), ("eta", 0.9), ("block_size", 10),
                            ("delta", 1e-10), ("alpha0", 1e-10), ("probe_seed", 23))
        check(get(config, key, NaN) == expected, "manifest $key differs from the intended power experiment")
    end
    check(config["p_range"] == [0.05, 0.45] && config["c_range"] == [0.1, 1.0], "generator parameter ranges differ from the intended experiment")
    check(manifest["julia_threads"] == 1 && manifest["blas_threads"] == 1, "Julia/BLAS thread counts differ from the one-thread protocol")

    pilot_reasons = String[]
    seeds == collect(1:100) || push!(pilot_reasons, "seed set differs from 1:100")
    cap == 100000 || push!(pilot_reasons, "iteration cap differs from 100000")
    repeats == 6 || push!(pilot_reasons, "measured repeat count differs from six position-balanced orders")
    warmups == 2 || push!(pilot_reasons, "warmup count differs from two")
    Set(methods) == Set(POWER_METHODS) || push!(pilot_reasons, "method set differs from the full experiment")
    scope = isempty(pilot_reasons) ? "full" : "pilot"

    benchmark_path = joinpath(directory, "benchmark_all_methods.mat")
    profile_path = joinpath(directory, "profile_statistics.mat")
    benchmark = matread(benchmark_path)
    saved_profile = matread(profile_path)
    input_hashes = Dict(basename(path) => file_sha256(path)
                        for path in (manifest_path, benchmark_path, profile_path))
    check(integers(benchmark["seeds"]) == seeds, "benchmark seed vector differs from manifest")
    check(Set(keys(benchmark["times"])) == Set(matname.(methods)), "benchmark timing method keys differ from manifest")
    check(Set(keys(benchmark["solved"])) == Set(matname.(methods)), "benchmark solved method keys differ from manifest")
    times = hcat([numbers(benchmark["times"][matname(name)]) for name in methods]...)
    check(size(times) == (length(seeds), length(methods)), "benchmark timing dimensions differ from manifest")
    summaries = Dict{String,Any}()
    per_seed = Dict{String,Any}()
    paired_initial_states = Dict{Int,Vector{Float64}}()
    paired_input_hashes = Dict{Int,String}()
    paired_execution_orders = Dict{Tuple{Int,Int},Vector{String}}()
    instances = Dict{Int,Any}()
    for seed in seeds
        path = joinpath(directory, "instances", "seed_$seed.mat")
        input_hashes[relpath(path, directory)] = file_sha256(path)
        data = matopen(path) do file
            read(file, "problem")
        end
        Q, x0 = Matrix{Float64}(data["Q"]), numbers(data["x0"])
        p, c = number(data["p"]), number(data["c"])
        check(size(Q) == (100, 100) && length(x0) == 100 && integer(data["dimension"]) == 100,
              "seed $seed: archived input dimensions are incorrect")
        check(all(isfinite, Q) && all(isfinite, x0), "seed $seed: nonfinite problem inputs")
        computed_hash = bytes2hex(sha256(vcat(reinterpret(UInt8, vec(Q)), reinterpret(UInt8, x0), reinterpret(UInt8, [p, c]))))
        check(computed_hash == String(data["initial_data_sha256"]), "seed $seed: archived problem input checksum mismatch")
        check(same_number(number(data["fstar"]), c^p), "seed $seed: archived optimum differs from c^p")
        expected_L = 2p * c^(p - 1) * eigmax(Symmetric(Q))
        check(same_number(number(data["L_estimate"]), expected_L), "seed $seed: archived smoothness bound is incorrect")
        instances[seed] = (Q=Q, x0=x0, p=p, c=c, fstar=c^p, input_hash=computed_hash, L=expected_L)
    end

    for (column, method) in enumerate(methods)
        history_path = joinpath(directory, "$(problem)_iterations_$(matname(method)).mat")
        input_hashes[basename(history_path)] = file_sha256(history_path)
        history = matread(history_path)
        check(String(history["method"]) == method, "$method: file identifies a different method")
        check(integers(history["seeds"]) == seeds, "$method: history seed vector differs from manifest")
        check(Set(keys(history["runs"])) == Set("seed_$seed" for seed in seeds), "$method: history seed keys differ from manifest")
        solved_flags = Bool.(flat(benchmark["solved"][matname(method)]))
        check(length(solved_flags) == length(seeds), "$method: wrong solved flag count")
        elapsed_all, iterations_all, objective_all, gap_all = Float64[], Int[], Float64[], Float64[]
        suboptimality_all, accepted_l_max, accepted_l_mean, accepted_l_harmonic = Float64[], Float64[], Float64[], Float64[]
        variability_ratios, variability_cvs, feasibility = Float64[], Float64[], Float64[]
        status_counts = Dict{String,Int}()
        failure_seed_counts = Dict{String,Int}()
        external_overheads, gc_fractions, allocated_bytes = Float64[], Float64[], Float64[]
        method_seed_details = Dict{String,Any}()

        for (row, seed) in enumerate(seeds)
            prefix = "$method / seed $seed"
            run = history["runs"]["seed_$seed"]
            k = integer(run["iterations"])
            values, gaps = numbers(run["values"]), numbers(run["gaps"])
            indices = integers(run["iterate_indices"])
            fstar = number(run["fstar"])
            solved = truth(run["solved"])
            elapsed = number(run["total_time"])
            repeat_times = numbers(run["repeat_times"])
            outcomes = run["repeat_outcomes"]
            check(integer(run["seed"]) == seed && integer(run["instance"]) == row, "$prefix: seed/instance mismatch")
            check(isfinite(fstar) && fstar > 0, "$prefix: known optimum c^p must be finite and positive")
            if haskey(run, "p") && haskey(run, "c")
                p, c = number(run["p"]), number(run["c"])
                check(0.05 <= p <= 0.45 && 0.10 <= c <= 1.0, "$prefix: generator p/c outside declared ranges")
                check(same_number(fstar, c^p), "$prefix: fstar does not equal archived c^p")
            else
                check(false, "$prefix: missing p/c needed to verify fstar=c^p")
            end
            initial_state = [fstar, first(values), first(gaps), number(get(run, "p", NaN)), number(get(run, "c", NaN))]
            if haskey(paired_initial_states, seed)
                check(same_array(initial_state, paired_initial_states[seed]), "$prefix: initial objective/gradient or instance parameters differ across methods")
            else
                paired_initial_states[seed] = initial_state
            end
            check(integer(get(run, "dimension", 0)) == 100, "$prefix: saved instance dimension is not 100")
            input_hash = String(get(run, "initial_data_sha256", ""))
            check(occursin(r"^[0-9a-f]{64}$", input_hash), "$prefix: invalid or missing instance input SHA256")
            instance = instances[seed]
            check(input_hash == instance.input_hash, "$prefix: input checksum differs from archived problem data")
            check(same_number(fstar, instance.fstar) && same_number(number(run["p"]), instance.p) &&
                  same_number(number(run["c"]), instance.c) && same_number(number(run["L_estimate"]), instance.L),
                  "$prefix: saved parameters differ from archived problem data")
            if haskey(paired_input_hashes, seed)
                check(input_hash == paired_input_hashes[seed], "$prefix: instance input hash differs across methods")
            else
                paired_input_hashes[seed] = input_hash
            end
            if haskey(run, "final_x")
                final_x = numbers(run["final_x"])
                check(length(final_x) == 100, "$prefix: saved final iterate has wrong dimension")
                solved && check(all(isfinite, final_x), "$prefix: successful final iterate is nonfinite")
                if length(final_x) == 100
                    base = dot(final_x, instance.Q, final_x) + instance.c
                    recomputed_f = base^instance.p
                    recomputed_norm = norm(2 * instance.p * base^(instance.p - 1) .* (instance.Q * final_x))
                    check(same_number(recomputed_f, last(values)), "$prefix: objective recomputed from archived final iterate disagrees")
                    check(same_number(recomputed_norm, last(gaps)), "$prefix: gradient norm recomputed from archived final iterate disagrees")
                end
            else
                check(false, "$prefix: final iterate was not archived")
            end
            check(0 <= k <= cap, "$prefix: iteration count outside common limit")
            check(indices == collect(0:k), "$prefix: iterate indices are not 0:k")
            check(length(values) == length(gaps) == length(indices) == k + 1, "$prefix: residual/objective/iterate histories are misaligned")
            check(integer(run["termination_iteration"]) == k, "$prefix: termination index differs from final iterate")
            check(same_number(number(run["final_value"]), last(values)), "$prefix: saved final objective differs from history")
            check(same_number(number(run["final_gap"]), last(gaps)), "$prefix: saved final gap differs from history")
            check(same_number(number(run["final_subopt"]), last(values) - fstar), "$prefix: incorrect final suboptimality")
            check(haskey(run, "independently_verified_final_value") &&
                  same_number(number(run["independently_verified_final_value"]), last(values)),
                  "$prefix: independently evaluated final objective differs from saved history")
            check(haskey(run, "independently_verified_final_gradnorm") &&
                  same_number(number(run["independently_verified_final_gradnorm"]), last(gaps)),
                  "$prefix: independently evaluated final gradient norm differs from saved history")
            check(integer(run["repeats"]) == repeats && integer(run["warmups"]) == warmups, "$prefix: repeat/warmup counts differ from manifest")
            check(String(run["aggregation"]) == "median", "$prefix: aggregation is not median")
            check(length(repeat_times) == repeats && Set(keys(outcomes)) == Set("repeat_$r" for r in 1:repeats), "$prefix: missing or extra repeat outcomes")
            check(all(t -> isfinite(t) && t > 0, repeat_times), "$prefix: nonpositive/nonfinite raw elapsed time")
            check(same_number(elapsed, median(repeat_times)), "$prefix: reported elapsed time is not the raw-repeat median")
            check(same_number(number(run["elapsed_time"]), elapsed), "$prefix: elapsed/total-time disagreement")
            raw = [outcomes["repeat_$repeat"] for repeat in 1:repeats]
            if haskey(run, "measurement_policy")
                policy = run["measurement_policy"]
                check(String(policy["clock"]) == "external_complete_call", "$prefix: clock is not complete external call")
                check(truth(policy["external_timing"]) && truth(policy["gc_before_call"]) &&
                      truth(policy["gc_inside_call_included"]) && truth(policy["compilation_inside_call_included"]),
                      "$prefix: measurement policy differs from the declared complete-call/GC protocol")
                check(String(policy["order_mode"]) == "balanced", "$prefix: measured method order is not balanced")
                check(integer(policy["instance_index"]) == row, "$prefix: order policy has incorrect instance index")
                check(Set(keys(policy["warmup_orders"])) == Set("warmup_$r" for r in 1:warmups), "$prefix: warmup orders missing or extra")
                check(Set(keys(policy["measured_orders"])) == Set("repeat_$r" for r in 1:repeats), "$prefix: measured orders missing or extra")
            else
                check(false, "$prefix: measurement policy was not archived")
            end
            for r in raw[2:end]
                check(integer(r["iterations"]) == integer(raw[1]["iterations"]) &&
                      String(r["status"]) == String(raw[1]["status"]) &&
                      same_number(number(r["final_value"]), number(raw[1]["final_value"])) &&
                      same_number(number(r["final_gap"]), number(raw[1]["final_gap"])),
                      "$prefix: repeated deterministic calls disagree in trajectory outcome")
            end
            check(solved == all(r -> truth(r["solved"]), raw), "$prefix: aggregate success does not require every repeat to solve")
            check(solved == solved_flags[row], "$prefix: history and benchmark success disagree")
            check(solved ? same_number(times[row, column], elapsed) : times[row, column] == Inf,
                  "$prefix: benchmark must store positive elapsed time for success and +Inf for failure")
            check(String(run["status"]) == (solved ? "converged" : "repeat_failure"), "$prefix: incorrect aggregate status")
            check(same_number(number(run["iteration_time"]), median(number(r["iteration_time"]) for r in raw)), "$prefix: iteration-time median mismatch")
            check(same_number(number(run["initialization_time"]), median(number(r["initialization_time"]) for r in raw)), "$prefix: initialization-time median mismatch")
            for field in ("outer_elapsed_time", "core_elapsed_time", "gc_time", "allocated_bytes", "compile_time", "recompile_time")
                check(haskey(run, field) && all(r -> haskey(r, field), raw), "$prefix: missing $field timing audit metadata")
                if haskey(run, field) && all(r -> haskey(r, field), raw)
                    check(same_number(number(run[field]), median(number(r[field]) for r in raw)), "$prefix: $field aggregate is not the repeat median")
                end
            end
            check(haskey(run, "repeat_core_times") &&
                  same_array(numbers(run["repeat_core_times"]), [number(r["core_elapsed_time"]) for r in raw]),
                  "$prefix: saved core timing vector differs from repeat outcomes")
            check(haskey(run, "repeat_outer_times") && same_array(numbers(run["repeat_outer_times"]), repeat_times),
                  "$prefix: saved outer timing vector differs from repeat_times")

            for (repeat, r) in enumerate(raw)
                tag = "$prefix / repeat $repeat"
                criterion = String(r["stopping_criterion"])
                value, gap, residual = number(r["final_value"]), number(r["final_gap"]), number(r["final_residual"])
                status = String(r["status"])
                status_counts[status] = get(status_counts, status, 0) + 1
                check(integer(r["repeat"]) == repeat, "$tag: incorrect repeat index")
                check(same_number(number(r["elapsed_time"]), repeat_times[repeat]), "$tag: elapsed time differs from repeat_times")
                check(haskey(r, "outer_elapsed_time") && same_number(number(r["outer_elapsed_time"]), repeat_times[repeat]),
                      "$tag: complete external timing differs from reported elapsed time")
                order = strings(get(r, "execution_order", String[]))
                position = integer(get(r, "execution_position", 0))
                check(length(order) == length(methods) && Set(order) == Set(methods), "$tag: execution order does not contain each method once")
                check(1 <= position <= length(order) && order[position] == method, "$tag: incorrect execution position")
                order_key = (seed, repeat)
                if haskey(paired_execution_orders, order_key)
                    check(order == paired_execution_orders[order_key], "$tag: different methods record conflicting execution orders")
                else
                    paired_execution_orders[order_key] = order
                end
                if haskey(run, "measurement_policy")
                    check(order == strings(run["measurement_policy"]["measured_orders"]["repeat_$repeat"]),
                          "$tag: execution order differs from measurement policy")
                end
                check(integer(r["iterations"]) == integer(r["termination_iteration"]) <= cap, "$tag: termination/iteration mismatch")
                check(truth(r["solved"]) == (status == "converged"), "$tag: status/success disagreement")
                check(status in ("converged", "iteration_limit", "time_limit", "nonfinite", "invalid_direction",
                      "zero_probe", "nonfinite_probe", "stationary_without_target", "stagnation",
                      "adabb_zero_curvature", "adabb_nonpositive_curvature", "invalid_curvature",
                      "invalid_step", "backtracking_limit"), "$tag: unrecognized terminal solver status")
                startswith(status, "adabb_") && check(method == "AdaBB", "$tag: AdaBB failure reported for another method")
                check(number(r["initialization_time"]) >= 0 && number(r["iteration_time"]) >= 0 &&
                      number(r["initialization_time"]) + number(r["iteration_time"]) <= number(r["elapsed_time"]) + 1e-10,
                      "$tag: timing components exceed complete elapsed time")
                check(criterion in ("objective", "none"), "$tag: unknown stopping criterion")
                expected_residual = value - fstar
                check(same_number(residual, expected_residual), "$tag: residual does not match the stated criterion")
                if truth(r["solved"])
                    check(criterion == "objective" && isfinite(residual) && -1e-10 <= residual <= epsilon,
                          "$tag: successful repeat does not meet its stopping tolerance")
                    check(isfinite(value) && isfinite(gap) && gap >= 0,
                          "$tag: successful repeat has invalid final objective/gradient norm")
                    check(value >= fstar - 1e-10, "$tag: objective is below the known optimum")
                end
                if haskey(r, "core_elapsed_time")
                    external, core = number(r["elapsed_time"]), number(r["core_elapsed_time"])
                    check(isfinite(external) && isfinite(core) && 0 <= core <= external + 1e-10,
                          "$tag: invalid complete-call/core timing relationship")
                    check(number(r["initialization_time"]) + number(r["iteration_time"]) <= core + 1e-10,
                          "$tag: timing components exceed internal core elapsed time")
                    push!(external_overheads, (external - core) / external)
                end
                if haskey(r, "gc_time")
                    gc = number(r["gc_time"])
                    check(isfinite(gc) && 0 <= gc <= number(r["elapsed_time"]) + 1e-10,
                          "$tag: invalid GC time")
                    push!(gc_fractions, gc / number(r["elapsed_time"]))
                end
                if haskey(r, "allocated_bytes")
                    allocated = number(r["allocated_bytes"])
                    check(isfinite(allocated) && allocated >= 0, "$tag: invalid allocation count")
                    push!(allocated_bytes, allocated)
                end
                for key in ("compile_time", "recompile_time")
                    haskey(r, key) && check(number(r[key]) == 0.0, "$tag: measured call included compilation ($key)")
                end
            end
            failure_reasons = sort!(unique(String(r["status"]) for r in raw if !truth(r["solved"])))
            if repeats == length(methods)
                check(sort([integer(r["execution_position"]) for r in raw]) == collect(1:length(methods)),
                      "$prefix: six measured repeats do not cover every execution position once")
            end
            for reason in failure_reasons
                failure_seed_counts[reason] = get(failure_seed_counts, reason, 0) + 1
            end
            check(strings(run["repeat_statuses"]) == [String(r["status"]) for r in raw], "$prefix: repeat status vector differs from detailed outcomes")
            representative = integer(run["representative_repeat"])
            check(1 <= representative <= repeats, "$prefix: representative repeat index is out of range")
            check(representative == argmin(abs.(repeat_times .- elapsed)), "$prefix: representative is not nearest the median elapsed time")
            selected = raw[representative]
            for field in ("final_value", "final_gap", "final_residual", "termination_iteration")
                check(same_number(number(run[field]), number(selected[field])), "$prefix: $field differs from representative repeat")
            end
            check(String(run["stopping_criterion"]) == String(selected["stopping_criterion"]), "$prefix: representative stopping criterion mismatch")
            trace_times = numbers(run["times"])
            check(all(t -> isfinite(t) && t >= 0, trace_times), "$prefix: invalid per-attempt timing")
            check(sum(trace_times) <= number(selected["elapsed_time"]) + 1e-10, "$prefix: representative timing history exceeds its elapsed time")
            check(same_number(first(trace_times), number(selected["initialization_time"])), "$prefix: initialization history differs from representative repeat")
            check(same_number(sum(trace_times[2:end]), number(selected["iteration_time"])), "$prefix: iteration history differs from representative repeat")
            violation = number(run["final_feasibility_violation"])
            check(violation == 0.0, "$prefix: unconstrained feasibility placeholder must be zero")
            curvatures = numbers(run["L_ks"])
            check(length(curvatures) == k, "$prefix: accepted-curvature count differs from accepted steps")
            check(all(L -> isfinite(L) && L > 0, curvatures), "$prefix: invalid accepted curvature")
            method == "Short-step" && check(all(L -> same_number(L, instance.L), curvatures),
                                            "$prefix: short-step curvature differs from the verified smoothness bound")
            check(length(flat(run["backtracks"])) == k, "$prefix: backtrack counts differ from accepted steps")
            check(all(b -> b >= 0, integers(run["backtracks"])), "$prefix: negative rejected-trial count")
            l_max, l_mean, l_harmonic = empty_stat(maximum, curvatures), empty_stat(mean, curvatures), harmonic_mean(curvatures)
            if method != "AdaBB"
                for (key, expected) in (("accepted_L_max", l_max), ("accepted_L_arithmetic_mean", l_mean),
                                        ("accepted_L_harmonic_mean", l_harmonic))
                    check(haskey(run, key) && same_number(number(run[key]), expected), "$prefix: incorrect $key summary")
                end
            end
            if method == "Adaptive adjustable"
                check(integers(run["gamma_iterations"]) == collect(0:10:k), "$prefix: adjustable-scaling block boundaries are incorrect")
                check(length(flat(run["gamma_history"])) == length(flat(run["gamma_iterations"])), "$prefix: scaling history is misaligned")
            end
            cv = length(repeat_times) > 1 ? std(repeat_times) / mean(repeat_times) : 0.0
            variability = maximum(repeat_times) / minimum(repeat_times)
            push!(elapsed_all, elapsed); push!(iterations_all, k)
            push!(objective_all, last(values)); push!(gap_all, last(gaps))
            push!(suboptimality_all, last(values) - fstar)
            push!(accepted_l_max, l_max); push!(accepted_l_mean, l_mean); push!(accepted_l_harmonic, l_harmonic)
            push!(variability_ratios, variability); push!(variability_cvs, cv); push!(feasibility, violation)
            method_seed_details["seed_$seed"] = Dict(
                "solved" => solved, "elapsed_median_seconds" => elapsed,
                "iterations" => k, "final_value" => last(values), "final_gap" => last(gaps),
                "final_suboptimality" => last(values) - fstar, "fstar" => fstar,
                "initial_data_sha256" => input_hash,
                "failure_reasons" => failure_reasons,
                "repeat_max_min_ratio" => variability, "repeat_coefficient_of_variation" => cv,
                "representative_repeat" => representative, "representative_feasibility_violation" => violation)
        end
        successful_times = elapsed_all[solved_flags]
        successful_iterations = iterations_all[solved_flags]
        summaries[method] = Dict(
            "solved" => count(solved_flags), "instances" => length(seeds),
            "success_rate" => count(solved_flags) / length(seeds),
            "mean_seconds_successes" => empty_stat(mean, successful_times),
            "median_seconds_successes" => empty_stat(median, successful_times),
            "median_seconds_all_outcomes" => median(elapsed_all),
            "median_iterations_successes" => empty_stat(median, successful_iterations),
            "median_iterations_all_outcomes" => median(iterations_all),
            "median_repeat_max_min_ratio" => median(variability_ratios),
            "maximum_repeat_max_min_ratio" => maximum(variability_ratios),
            "median_repeat_coefficient_of_variation" => median(variability_cvs),
            "maximum_repeat_coefficient_of_variation" => maximum(variability_cvs),
            "median_final_objective" => median(objective_all), "median_final_gap" => median(gap_all),
            "median_final_suboptimality" => median(suboptimality_all),
            "maximum_final_suboptimality" => maximum(suboptimality_all),
            "median_final_gradient_norm" => median(gap_all), "maximum_final_gradient_norm" => maximum(gap_all),
            "median_final_suboptimality_successes" => empty_stat(median, suboptimality_all[solved_flags]),
            "median_final_gradient_norm_successes" => empty_stat(median, gap_all[solved_flags]),
            "median_accepted_L_max" => empty_stat(median, filter(isfinite, accepted_l_max)),
            "median_accepted_L_arithmetic_mean" => empty_stat(median, filter(isfinite, accepted_l_mean)),
            "median_accepted_L_harmonic_mean" => empty_stat(median, filter(isfinite, accepted_l_harmonic)),
            "L_interpretation" => method == "AdaBB" ? "reciprocal accepted step size 1/alpha" : "accepted curvature before eta decay",
            "median_external_overhead_fraction" => empty_stat(median, external_overheads),
            "median_gc_fraction" => empty_stat(median, gc_fractions),
            "median_allocated_bytes" => empty_stat(median, allocated_bytes),
            "maximum_representative_feasibility_violation" => maximum(feasibility),
            "raw_repeat_status_counts" => status_counts,
            "failed_seed_reason_counts" => failure_seed_counts)
        per_seed[method] = method_seed_details
    end

    profile = finite_performance_profile(times; logscale=false)
    check(strings(saved_profile["methods"]) == methods, "profile method order differs from manifest")
    check(same_array(Float64.(saved_profile["ratios"]), profile.ratios), "profile ratios disagree with successful timing ratios or failure handling")
    check(same_array(numbers(saved_profile["thresholds"]), profile.thresholds), "profile thresholds disagree with finite ratios")
    check(same_array(Float64.(saved_profile["fractions"]), profile.fractions), "profile fractions use incorrect coverage/denominator")
    check(same_array(numbers(saved_profile["success_rates"]), profile.success_rates), "profile success rates disagree with outcomes")
    check(same_array(numbers(saved_profile["finite_ratio_coverage"]), profile.finite_ratio_coverage), "finite-ratio coverage disagrees with profile")
    check(integer(saved_profile["all_failed_instances"]) == profile.all_failed_instances, "all-failed instance count mismatch")
    check(same_number(number(saved_profile["max_ratio"]), profile.max_ratio), "profile maximum finite ratio mismatch")
    comparisons = Dict{String,Any}()
    for reference in ("Adaptive constant", "Adaptive adjustable")
        paired = paired_speed_ratios(times; reference=findfirst(==(reference), methods))
        saved_pairs = saved_profile["paired_speed_comparisons"][matname(reference)]
        comparisons[reference] = Dict{String,Any}()
        for (column, method) in enumerate(methods)
            comparison = paired[column]
            saved = saved_pairs[matname(method)]
            check(integer(saved["paired_instances"]) == comparison.paired_instances, "$reference/$method: paired-speedup denominator mismatch")
            check(same_number(number(saved["baseline_over_reference_harmonic"]), comparison.baseline_over_reference_harmonic), "$reference/$method: harmonic speedup mismatch")
            comparisons[reference][method] = Dict(
                "paired_instances" => comparison.paired_instances,
                "baseline_over_reference_harmonic" => comparison.baseline_over_reference_harmonic,
                "baseline_over_reference_median" => empty_stat(median, comparison.ratios))
        end
    end
    fastest = profile.ratios .== 1.0
    fractional_wins = zeros(length(methods))
    for row in axes(fastest, 1)
        winners = findall(fastest[row, :])
        for column in winners
            fractional_wins[column] += 1 / length(winners)
        end
    end
    for (column, method) in enumerate(methods)
        summaries[method]["fastest_including_ties_count"] = count(fastest[:, column])
        summaries[method]["fastest_including_ties_share"] = count(fastest[:, column]) / length(seeds)
        summaries[method]["fastest_split_ties_share"] = fractional_wins[column] / length(seeds)
    end
    push!(notes, "This is an unconstrained problem: saved feasibility=0 is a placeholder, not an additional numerical feasibility test.")
    push!(notes, "The only convergence target is f(x)-c^p <= 1e-5. Gradient norms are reported as diagnostics, not treated as an alternative stopping target.")
    push!(notes, "AdaBB nonpositive/zero curvature statuses are failed runs; their early elapsed times do not contribute to successful timing or speedup summaries.")
    push!(notes, "Paired-instance validation checks input SHA256 against archived Q/x0/p/c and compares initial objective/gradient norms across methods. Final objective and gradient norms are recomputed from the archived final iterates; the smoothness bound is independently recomputed from Q.")
    push!(notes, "Successful timing summaries exclude failures; all-outcome elapsed/iteration medians are reported separately.")
    push!(notes, "Ratios and speedups compare paired results in this run only; no historical or cross-hardware timing comparison is made.")
    push!(notes, "Profile fractions include all instances in their denominator; failed methods retain infinite ratios.")
    report = Dict{String,Any}(
        "scope" => scope, "pilot_reasons" => pilot_reasons,
        "validation_passed" => isempty(problems), "validation_errors" => problems,
        "analyzed_utc" => string(now(UTC)), "output_directory" => directory,
        "seeds" => seeds, "methods" => methods, "epsilon" => epsilon,
        "max_iter_cap" => cap, "warmups" => warmups, "repeats" => repeats,
        "config" => config,
        "source_revision" => manifest["source_revision"], "source_dirty" => manifest["source_dirty"],
        "julia_version" => manifest["julia_version"], "julia_threads" => manifest["julia_threads"],
        "blas_threads" => manifest["blas_threads"], "input_sha256" => input_hashes,
        "summary" => summaries, "per_seed" => per_seed,
        "paired_speedups" => comparisons, "all_failed_instances" => profile.all_failed_instances,
        "profile_max_finite_ratio" => profile.max_ratio, "notes" => notes)
    open(joinpath(directory, "analysis_report.toml"), "w") do io
        TOML.print(io, report; sorted=true)
    end
    text_report = IOBuffer()
    println(text_report, "Power-shifted quadratic $(uppercase(scope)) run: $(length(seeds)) instances; $warmups warmup(s), $repeats measured repeats, median elapsed time.")
    println(text_report, "Validation: ", isempty(problems) ? "PASS" : "FAIL ($(length(problems)) issues)")
    isempty(pilot_reasons) || println(text_report, "Pilot configuration: ", join(pilot_reasons, "; "))
    println(text_report, "Julia $(manifest["julia_version"]); Julia/BLAS threads $(manifest["julia_threads"])/$(manifest["blas_threads"]).")
    println(text_report, "Times/iterations below summarize successful instances; repeat range is the median per-instance max/min ratio.")
    @printf(text_report, "%-22s %9s %12s %12s %12s %11s %10s\n", "Method", "Solved", "Mean sec", "Median sec", "Median iter", "Repeat max/min", "Fastest %")
    for method in methods
        s = summaries[method]
        mean_time = s["solved"] == 0 ? "N/A" : @sprintf("%.6g", s["mean_seconds_successes"])
        median_time = s["solved"] == 0 ? "N/A" : @sprintf("%.6g", s["median_seconds_successes"])
        median_iteration = s["solved"] == 0 ? "N/A" : @sprintf("%.1f", s["median_iterations_successes"])
        @printf(text_report, "%-22s %9s %12s %12s %12s %11.3f %10.1f\n",
                method, "$(s["solved"])/$(s["instances"])", mean_time,
                median_time, median_iteration,
                s["median_repeat_max_min_ratio"], 100s["fastest_including_ties_share"])
    end
    println(text_report, "\nComplete-call timing audit: compilation must be zero in every measured call; GC inside a call is included.")
    @printf(text_report, "%-22s %16s %16s %16s\n", "Method", "Median overhead%", "Median GC%", "Median bytes")
    for method in methods
        s = summaries[method]
        @printf(text_report, "%-22s %16.6g %16.6g %16.6g\n", method,
                100s["median_external_overhead_fraction"], 100s["median_gc_fraction"], s["median_allocated_bytes"])
    end
    println(text_report, "\nFinal diagnostics (representative repeat, ALL outcomes including failures):")
    @printf(text_report, "%-22s %12s %12s %12s %12s\n", "Method", "Median f-f*", "Max f-f*", "Median |grad|", "Elapsed sec")
    for method in methods
        s = summaries[method]
        @printf(text_report, "%-22s %12.6g %12.6g %12.6g %12.6g\n", method,
                s["median_final_suboptimality"], s["maximum_final_suboptimality"],
                s["median_final_gradient_norm"], s["median_seconds_all_outcomes"])
    end
    println(text_report, "The all-outcome elapsed column includes time to failure and is not a time-to-solution comparison.")
    println(text_report, "\nRaw repeat terminal statuses (repeat_failure is only the aggregate label):")
    for method in methods
        s = summaries[method]
        counts = join(["$status=$count" for (status, count) in sort!(collect(s["raw_repeat_status_counts"]); by=first)], ", ")
        println(text_report, method, ": ", counts)
        isempty(s["failed_seed_reason_counts"]) && continue
        reasons = join(["$status=$count" for (status, count) in sort!(collect(s["failed_seed_reason_counts"]); by=first)], ", ")
        println(text_report, "  Failed instances by reason: ", reasons)
    end
    println(text_report, "\nAccepted-L summaries: median of each instance's representative history; AdaBB stores 1/alpha.")
    @printf(text_report, "%-22s %12s %12s %12s\n", "Method", "Max L", "Mean L", "Harmonic L")
    for method in methods
        s = summaries[method]
        @printf(text_report, "%-22s %12.6g %12.6g %12.6g\n", method,
                s["median_accepted_L_max"], s["median_accepted_L_arithmetic_mean"], s["median_accepted_L_harmonic_mean"])
    end
    println(text_report, "\nPaired speedups: baseline time / adaptive time; values above 1 favor the named adaptive method.")
    for reference in ("Adaptive constant", "Adaptive adjustable")
        println(text_report, reference, ":")
        for method in methods
            method == reference && continue
            c = comparisons[reference][method]
            if c["paired_instances"] == 0
                @printf(text_report, "  %-22s N/A: no paired successes (0/%d)\n", method, length(seeds))
                continue
            end
            @printf(text_report, "  %-22s harmonic=%8.4g median=%8.4g pairs=%d/%d\n",
                    method, c["baseline_over_reference_harmonic"], c["baseline_over_reference_median"],
                    c["paired_instances"], length(seeds))
        end
    end
    println(text_report, "\nAll-method failures: $(profile.all_failed_instances)/$(length(seeds)).")
    for note in notes
        println(text_report, note)
    end
    for problem in problems
        println(text_report, "ERROR: ", problem)
    end
    rendered = String(take!(text_report))
    write(joinpath(directory, "analysis_report.txt"), rendered)
    print(rendered)
    return report
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: julia --project=. scripts/analyze_power_run.jl OUTPUT_DIRECTORY")
    report = analyze_power_run(only(ARGS))
    report["validation_passed"] || exit(1)
end
