#!/usr/bin/env julia
# Postprocessing only: never includes a problem driver or runs a solver.
using Dates, MAT, Printf, SHA, Statistics, TOML
include(joinpath(@__DIR__, "benchmark_support.jl"))
using .ACGDBenchmarkSupport

const LASSO_METHODS = ["Adaptive constant", "Adaptive adjustable", "Pure backtracking",
                       "Auto-conditioned", "Short-step", "Open-loop"]
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

"Validate saved Lasso results and write separate human/machine-readable reports."
function analyze_lasso_run(output_directory)
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
    check(problem in ("lasso", "Lasso_Problem"), "manifest problem is not Lasso: $problem")
    check(length(seeds) > 0 && length(unique(seeds)) == length(seeds), "manifest seeds must be nonempty and unique")
    check(length(methods) == 6 && Set(methods) == Set(LASSO_METHODS), "expected all six Lasso methods exactly once")
    check(epsilon == 1e-5, "Lasso tolerance differs from the documented 1e-5")
    check(config["stop_rule"] == "either", "Lasso must use either objective or stationarity stopping")
    check(warmups >= 1 && repeats >= 1, "invalid warmup/repeat count")

    pilot_reasons = String[]
    seeds == collect(1:100) || push!(pilot_reasons, "seed set differs from 1:100")
    cap == 100000 || push!(pilot_reasons, "iteration cap differs from 100000")
    repeats >= 3 || push!(pilot_reasons, "fewer than three measured repeats")
    warmups >= 1 || push!(pilot_reasons, "no warmup")
    Set(methods) == Set(LASSO_METHODS) || push!(pilot_reasons, "method set differs from the full experiment")
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
        variability_ratios, variability_cvs, feasibility = Float64[], Float64[], Float64[]
        status_counts = Dict{String,Int}()
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
            check(fstar == 0.0, "$prefix: planted Lasso optimum should be zero")
            check(0 <= k <= cap, "$prefix: iteration count outside common limit")
            check(indices == collect(0:k), "$prefix: iterate indices are not 0:k")
            check(length(values) == length(gaps) == length(indices) == k + 1, "$prefix: residual/objective/iterate histories are misaligned")
            check(integer(run["termination_iteration"]) == k, "$prefix: termination index differs from final iterate")
            check(same_number(number(run["final_value"]), last(values)), "$prefix: saved final objective differs from history")
            check(same_number(number(run["final_gap"]), last(gaps)), "$prefix: saved final gap differs from history")
            check(same_number(number(run["final_subopt"]), last(values) - fstar), "$prefix: incorrect final suboptimality")
            check(integer(run["repeats"]) == repeats && integer(run["warmups"]) == warmups, "$prefix: repeat/warmup counts differ from manifest")
            check(String(run["aggregation"]) == "median", "$prefix: aggregation is not median")
            check(length(repeat_times) == repeats && Set(keys(outcomes)) == Set("repeat_$r" for r in 1:repeats), "$prefix: missing or extra repeat outcomes")
            check(all(t -> isfinite(t) && t > 0, repeat_times), "$prefix: nonpositive/nonfinite raw elapsed time")
            check(same_number(elapsed, median(repeat_times)), "$prefix: reported elapsed time is not the raw-repeat median")
            check(same_number(number(run["elapsed_time"]), elapsed), "$prefix: elapsed/total-time disagreement")
            raw = [outcomes["repeat_$repeat"] for repeat in 1:repeats]
            check(solved == all(r -> truth(r["solved"]), raw), "$prefix: aggregate success does not require every repeat to solve")
            check(solved == solved_flags[row], "$prefix: history and benchmark success disagree")
            check(solved ? same_number(times[row, column], elapsed) : times[row, column] == Inf,
                  "$prefix: benchmark must store positive elapsed time for success and +Inf for failure")
            check(String(run["status"]) == (solved ? "converged" : "repeat_failure"), "$prefix: incorrect aggregate status")
            check(same_number(number(run["iteration_time"]), median(number(r["iteration_time"]) for r in raw)), "$prefix: iteration-time median mismatch")
            check(same_number(number(run["initialization_time"]), median(number(r["initialization_time"]) for r in raw)), "$prefix: initialization-time median mismatch")

            for (repeat, r) in enumerate(raw)
                tag = "$prefix / repeat $repeat"
                criterion = String(r["stopping_criterion"])
                value, gap, residual = number(r["final_value"]), number(r["final_gap"]), number(r["final_residual"])
                status = String(r["status"])
                status_counts[status] = get(status_counts, status, 0) + 1
                check(integer(r["repeat"]) == repeat, "$tag: incorrect repeat index")
                check(same_number(number(r["elapsed_time"]), repeat_times[repeat]), "$tag: elapsed time differs from repeat_times")
                check(integer(r["iterations"]) == integer(r["termination_iteration"]) <= cap, "$tag: termination/iteration mismatch")
                check(truth(r["solved"]) == (status == "converged"), "$tag: status/success disagreement")
                check(number(r["initialization_time"]) >= 0 && number(r["iteration_time"]) >= 0 &&
                      number(r["initialization_time"]) + number(r["iteration_time"]) <= number(r["elapsed_time"]) + 1e-10,
                      "$tag: timing components exceed complete elapsed time")
                check(criterion in ("objective", "stationarity", "none"), "$tag: unknown stopping criterion")
                expected_residual = criterion == "objective" ? value - fstar : gap
                check(same_number(residual, expected_residual), "$tag: residual does not match the stated criterion")
                if truth(r["solved"])
                    check(criterion != "none" && isfinite(residual) && residual <= epsilon,
                          "$tag: successful repeat does not meet its stopping tolerance")
                    check(isfinite(value) && isfinite(gap) && gap >= -epsilon,
                          "$tag: successful repeat has invalid final objective/gap")
                    check(value >= fstar - 1e-10, "$tag: objective is below the known optimum")
                end
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
            check(isfinite(violation) && 0 <= violation <= 1e-8, "$prefix: representative L1 feasibility check failed")
            curvatures = numbers(run["L_ks"])
            check(method == "Open-loop" ? isempty(curvatures) : length(curvatures) == k, "$prefix: accepted-curvature count differs from accepted steps")
            check(all(L -> isfinite(L) && L > 0, curvatures), "$prefix: invalid accepted curvature")
            check(length(flat(run["backtracks"])) == k, "$prefix: backtrack counts differ from accepted steps")
            if method == "Adaptive adjustable"
                check(integers(run["gamma_iterations"]) == collect(0:10:k), "$prefix: adjustable-scaling block boundaries are incorrect")
            end
            cv = length(repeat_times) > 1 ? std(repeat_times) / mean(repeat_times) : 0.0
            variability = maximum(repeat_times) / minimum(repeat_times)
            push!(elapsed_all, elapsed); push!(iterations_all, k)
            push!(objective_all, last(values)); push!(gap_all, last(gaps))
            push!(variability_ratios, variability); push!(variability_cvs, cv); push!(feasibility, violation)
            method_seed_details["seed_$seed"] = Dict(
                "solved" => solved, "elapsed_median_seconds" => elapsed,
                "iterations" => k, "final_value" => last(values), "final_gap" => last(gaps),
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
            "maximum_representative_feasibility_violation" => maximum(feasibility),
            "raw_repeat_status_counts" => status_counts)
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
    push!(notes, "Feasibility validates the saved representative-repeat violation; final iterates and other-repeat feasibility are not stored.")
    push!(notes, "Successful timing summaries exclude failures; all-outcome elapsed/iteration medians are reported separately.")
    push!(notes, "Ratios and speedups compare paired results in this run only; no historical or cross-hardware timing comparison is made.")
    push!(notes, "Profile fractions include all instances in their denominator; failed methods retain infinite ratios.")
    report = Dict{String,Any}(
        "scope" => scope, "pilot_reasons" => pilot_reasons,
        "validation_passed" => isempty(problems), "validation_errors" => problems,
        "analyzed_utc" => string(now(UTC)), "output_directory" => directory,
        "seeds" => seeds, "methods" => methods, "epsilon" => epsilon,
        "max_iter_cap" => cap, "warmups" => warmups, "repeats" => repeats,
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
    println(text_report, "Lasso $(uppercase(scope)) run: $(length(seeds)) instances; $warmups warmup(s), $repeats measured repeats, median elapsed time.")
    println(text_report, "Validation: ", isempty(problems) ? "PASS" : "FAIL ($(length(problems)) issues)")
    isempty(pilot_reasons) || println(text_report, "Pilot configuration: ", join(pilot_reasons, "; "))
    println(text_report, "Julia $(manifest["julia_version"]); Julia/BLAS threads $(manifest["julia_threads"])/$(manifest["blas_threads"]).")
    println(text_report, "Times/iterations below summarize successful instances; repeat range is the median per-instance max/min ratio.")
    @printf(text_report, "%-22s %9s %12s %12s %12s %11s %10s\n", "Method", "Solved", "Mean sec", "Median sec", "Median iter", "Repeat max/min", "Fastest %")
    for method in methods
        s = summaries[method]
        @printf(text_report, "%-22s %9s %12.6g %12.6g %12.1f %11.3f %10.1f\n",
                method, "$(s["solved"])/$(s["instances"])", s["mean_seconds_successes"],
                s["median_seconds_successes"], s["median_iterations_successes"],
                s["median_repeat_max_min_ratio"], 100s["fastest_including_ties_share"])
    end
    println(text_report, "\nPaired speedups: baseline time / adaptive time; values above 1 favor the named adaptive method.")
    for reference in ("Adaptive constant", "Adaptive adjustable")
        println(text_report, reference, ":")
        for method in methods
            method == reference && continue
            c = comparisons[reference][method]
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
    length(ARGS) == 1 || error("usage: julia --project=. scripts/analyze_lasso_run.jl OUTPUT_DIRECTORY")
    report = analyze_lasso_run(only(ARGS))
    report["validation_passed"] || exit(1)
end
