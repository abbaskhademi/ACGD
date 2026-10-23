#!/usr/bin/env julia
# Remeasure the paused StQO instance after all other timing jobs have finished.
# Original results and frozen source are read-only inputs; output must be fresh.
using Dates, LinearAlgebra, MAT, SHA, TOML

const STQO_PROBLEM = "NonconvexQP_Simplex"
const SEED_DECLARATION = "seeds  = 1:parse(Int, get(ENV, \"ACGD_SEED_COUNT\", \"100\"))"
const EXPORT_MARKER = "# 3b. Save per-iteration histories: one .mat per method"
const BENCHMARK_END = "end # ACGD_RUN_BENCHMARKS"

scalar(x) = x isa AbstractArray ? only(x) : x
integer(x) = Int(scalar(x))
number(x) = Float64(scalar(x))
truth(x) = Bool(scalar(x))
flat(x) = x isa AbstractArray ? vec(x) : [x]
matname(name) = replace(name, r"[^A-Za-z0-9_]" => "_")
file_hash(path) = open(io -> bytes2hex(sha256(io)), path)

function exact_replace(source, old, new)
    length(findall(old, source)) == 1 || error("expected exactly one source marker: $old")
    return replace(source, old => new; count=1)
end

"Change only seed selection, timing-order index, saved instance index and an input guard."
function replacement_driver(source::String, seed::Int, expected_hash::String)
    occursin(r"^[0-9a-f]{64}$", expected_hash) || error("invalid original input checksum")
    text = exact_replace(source, SEED_DECLARATION, "seeds = [$seed]")
    text = exact_replace(text, "runs = benchmark_methods(calls; instance_index=idx,",
                         "runs = benchmark_methods(calls; instance_index=$seed,")
    text = exact_replace(text, "\"instance\"     => idx,", "\"instance\"     => $seed,")
    guard = "input_hash = snapshot_sha256(snapshot)"
    text = exact_replace(text, guard, guard * "\n    input_hash == \"$expected_hash\" || error(\"repair instance differs from original checkpoint\")")
    return text
end

"Reuse the frozen driver's exports and figures with an assembled full result set."
function export_tail(source::String)
    length(findall(EXPORT_MARKER, source)) == 1 || error("export marker missing or duplicated")
    start = first(findfirst(EXPORT_MARKER, source))
    text = source[start:end]
    text = exact_replace(text, BENCHMARK_END, "")
    # All measured work precedes this marker. A changed driver must be reviewed.
    occursin("benchmark_methods(", text) && error("export section unexpectedly contains solver measurements")
    return text
end

function hashes(directory)
    result = Dict{String,String}()
    for (root, directories, files) in walkdir(directory)
        for name in sort(files)
            path = joinpath(root, name)
            isfile(path) && (result[relpath(path, directory)] = file_hash(path))
        end
    end
    return result
end

function write_toml(path, data)
    temporary = path * ".tmp"
    open(temporary, "w") do io
        TOML.print(io, data; sorted=true)
    end
    mv(temporary, path; force=true)
end

function state!(root, status; extra=Dict{String,Any}())
    record = Dict{String,Any}("status" => status, "updated_utc" => string(now(UTC)))
    merge!(record, extra)
    write_toml(joinpath(root, "repair_status.toml"), record)
    write(joinpath(root, "status.txt"), status * "\n")
end

function execution_module(name)
    mod = Module(name)
    isdefined(mod, :include) || Core.eval(mod, :(include(path) = Base.include(@__MODULE__, path)))
    isdefined(mod, :eval) || Core.eval(mod, :(eval(expression) = Core.eval(@__MODULE__, expression)))
    return mod
end

function install_globals(mod; variables...)
    for (name, value) in variables
        Core.eval(mod, Expr(:(=), name, QuoteNode(value)))
    end
end

function report_line(name, run, seed)
    status = truth(run["solved"]) ? "solved" : "NOT solved"
    line = string("instance=", seed, "  seed=", seed, "  ", rpad(name, 22), status,
        "  iters=", integer(run["iterations"]),
        "  time=", round(number(run["total_time"]), sigdigits=6), "s",
        "  f_last=", round(number(run["final_value"]), digits=6),
        "  gap_last=", round(number(run["final_gap"]), digits=6),
        "  feasibility_check=", number(run["final_feasibility_violation"]),
        "  status=", String(run["status"]))
    if !truth(run["solved"])
        reasons = sort(unique(String(raw["status"]) for raw in values(run["repeat_outcomes"])))
        line *= "  reasons=" * join(reasons, ",")
    end
    return line
end

function read_arguments(arguments)
    length(arguments) % 2 == 0 || error("arguments must be --name VALUE pairs")
    options = Dict(arguments[index] => arguments[index + 1] for index in 1:2:length(arguments))
    allowed = Set(["--source-dir", "--original-dir", "--output-dir", "--seed"])
    isempty(setdiff(Set(keys(options)), allowed)) || error("unknown argument")
    for key in ("--source-dir", "--original-dir", "--output-dir")
        haskey(options, key) || error("required argument $key")
    end
    return (source=abspath(options["--source-dir"]), original=abspath(options["--original-dir"]),
            output=abspath(options["--output-dir"]), seed=parse(Int, get(options, "--seed", "54")))
end

function repair_stqo(options)
    source, original, output, seed = options.source, options.original, options.output, options.seed
    seed == 54 || error("this repair is specifically authorized for interrupted seed 54")
    isdir(source) && isdir(original) || error("source and completed original results must exist")
    ispath(output) && error("repair output already exists; preserve it and choose a fresh directory")
    is_subdir(path, parent) = path == parent || startswith(path, parent * "/")
    (is_subdir(output, original) || is_subdir(output, source) || is_subdir(original, output) || is_subdir(source, output)) &&
        error("repair output must be separate from source and original results")
    manifest_path = joinpath(original, "run_manifest.toml")
    manifest = TOML.parsefile(manifest_path)
    config = manifest["config"]
    String(config["problem"]) == STQO_PROBLEM || error("original problem is not StQO")
    Int.(config["seeds"]) == collect(1:100) || error("original must contain the full 100-seed benchmark")
    config["max_iter_cap"] == 100000 && config["warmups"] == 2 && config["repeats"] == 6 || error("unexpected original run controls")
    config["order_seed"] == 20261021 && config["order_mode"] == "balanced" || error("unexpected original ordering policy")
    config["epsilon"] == 1e-3 && config["stop_rule"] == "stationarity" || error("unexpected stopping target")
    string(VERSION) == manifest["julia_version"] || error("Julia version differs from original run")
    Threads.nthreads() == manifest["julia_threads"] == 1 || error("Julia threads must match original one-thread run")
    BLAS.get_num_threads() == manifest["blas_threads"] == 1 || error("BLAS threads must match original one-thread run")
    for relative in ("STQO.jl", "src/ACGDSolvers.jl", "scripts/benchmark_support.jl", "Project.toml", "Manifest.toml")
        file_hash(joinpath(source, relative)) == manifest["source_sha256"][relative] || error("frozen source differs: $relative")
    end
    for s in 1:100
        isfile(joinpath(original, "instances", "seed_$s.mat")) || error("original checkpoint $s is missing; wait for the full run")
    end
    methods = String.(config["method_order"])
    for name in methods
        isfile(joinpath(original, "$(STQO_PROBLEM)_iterations_$(matname(name)).mat")) || error("original final history missing: $name")
    end
    original_checkpoint_path = joinpath(original, "instances", "seed_$seed.mat")
    original_checkpoint = matread(original_checkpoint_path)
    expected_hash = String(original_checkpoint["initial_data_sha256"])
    driver_path = joinpath(source, "STQO.jl")
    driver_source = read(driver_path, String)
    transformed = replacement_driver(driver_source, seed, expected_hash)
    exports = export_tail(driver_source)
    baseline_hashes = hashes(original)

    mkpath(output)
    write(joinpath(output, "replacement_driver.jl"), transformed)
    write(joinpath(output, "reused_export_tail.jl"), exports)
    original_manifest_copy = joinpath(output, "original_run_manifest.toml")
    cp(manifest_path, original_manifest_copy)
    write_toml(joinpath(output, "original_output_checksums.toml"), baseline_hashes)
    provenance = Dict{String,Any}(
        "kind" => "composite StQO result: 99 original seeds plus remeasured interrupted seed 54",
        "started_utc" => string(now(UTC)), "seed" => seed,
        "reason" => "systemd freezer pause contaminated the complete-call elapsed measurement for this seed",
        "original_results" => original, "frozen_source" => source,
        "original_manifest_sha256" => file_hash(manifest_path),
        "original_checkpoint_sha256" => file_hash(original_checkpoint_path),
        "initial_data_sha256" => expected_hash,
        "repair_script_sha256" => file_hash(@__FILE__),
        "original_driver_sha256" => file_hash(driver_path),
        "replacement_driver_sha256" => file_hash(joinpath(output, "replacement_driver.jl")),
        "reused_export_tail_sha256" => file_hash(joinpath(output, "reused_export_tail.jl")),
        "control_changes" => ["select only seed 54", "preserve measurement instance_index=54",
                               "preserve saved instance index=54", "assert original input hash before measurement"],
        "warmups" => 2, "repeats" => 6, "order_seed" => 20261021, "instance_index" => 54,
        "method_order" => methods,
        "frozen_analyzer_sha256" => file_hash(joinpath(source, "scripts", "analyze_remaining_run.jl")),
    )
    write_toml(joinpath(output, "repair_provenance.toml"), provenance)
    try
        state!(output, "measuring_seed_54")
        replacement_root = joinpath(output, "replacement")
        driver_module = execution_module(:StQOPauseRepairDriver)
        withenv("ACGD_RUN_BENCHMARKS" => "true", "ACGD_OUTPUT_ROOT" => replacement_root,
                "ACGD_SEED_COUNT" => "1", "ACGD_MAX_ITER" => "100000",
                "ACGD_WARMUPS" => "2", "ACGD_REPEATS" => "6", "ACGD_PROGRESS" => "true") do
            # Use the original filename so @__DIR__ continues resolving frozen includes.
            Base.include_string(driver_module, transformed, driver_path)
        end
        replacement = joinpath(replacement_root, STQO_PROBLEM)
        checkpoint_path = joinpath(replacement, "instances", "seed_$seed.mat")
        checkpoint = matread(checkpoint_path)
        String(checkpoint["initial_data_sha256"]) == expected_hash || error("replacement input hash differs")
        for name in methods
            run = checkpoint["runs"][matname(name)]
            integer(run["instance"]) == seed || error("replacement saved instance index differs")
            integer(run["measurement_policy"]["instance_index"]) == seed || error("replacement ordering index differs")
            original_run = original_checkpoint["runs"][matname(name)]
            for repeat in 1:6
                fresh = run["repeat_outcomes"]["repeat_$repeat"]
                prior = original_run["repeat_outcomes"]["repeat_$repeat"]
                String.(flat(fresh["execution_order"])) == String.(flat(prior["execution_order"])) ||
                    error("replacement execution order differs for $name repeat $repeat")
            end
        end
        provenance["replacement_manifest_sha256"] = file_hash(joinpath(replacement, "run_manifest.toml"))
        provenance["replacement_checkpoint_sha256"] = file_hash(checkpoint_path)
        provenance["replacement_completed_utc"] = string(now(UTC))
        write_toml(joinpath(output, "repair_provenance.toml"), provenance)

        state!(output, "assembling_corrected_results")
        corrected = joinpath(output, "results", STQO_PROBLEM)
        mkpath(dirname(corrected))
        cp(original, corrected)
        # Never leave an old PASS or a stale original-run publication hold in the new copy.
        for filename in ("analysis_report.txt", "analysis_report.toml", "output_checksums.json",
                         "output_checksums.sha256", "output_checksums.toml", "TIMING_REPAIR_REQUIRED.json")
            path = joinpath(corrected, filename)
            isfile(path) && rm(path)
        end
        write(joinpath(corrected, "REPAIR_VALIDATION_PENDING.txt"),
              "Composite result under construction; use only after repair_status.toml reports complete.\n")
        per_iter = Dict{String,Any}()
        all_times = Dict{String,Vector{Float64}}()
        all_solved = Dict{String,BitVector}()
        for name in methods
            path = joinpath(original, "$(STQO_PROBLEM)_iterations_$(matname(name)).mat")
            histories = matread(path)["runs"]
            histories["seed_$seed"] = checkpoint["runs"][matname(name)]
            per_iter[name] = histories
            solved = BitVector([truth(histories["seed_$s"]["solved"]) for s in 1:100])
            all_solved[name] = solved
            all_times[name] = [solved[s] ? number(histories["seed_$s"]["total_time"]) : Inf for s in 1:100]
        end
        report_lines = [report_line(name, per_iter[name]["seed_$s"], s) for s in 1:100 for name in methods]
        cp(checkpoint_path, joinpath(corrected, "instances", "seed_$seed.mat"); force=true)
        install_globals(driver_module; output_dir=corrected, instance_dir=joinpath(corrected, "instances"),
                        seeds=collect(1:100), method_order=methods, per_iter=per_iter,
                        all_times=all_times, all_solved=all_solved, report_lines=report_lines)
        # This tail only writes histories, reports, summary statistics and profile figures.
        Base.include_string(driver_module, exports, driver_path)
        corrected_manifest = deepcopy(manifest)
        corrected_manifest["composite_result"] = merge(copy(provenance), Dict(
            "corrected_results" => corrected,
            "untouched_seeds" => [s for s in 1:100 if s != seed]))
        write_toml(joinpath(corrected, "run_manifest.toml"), corrected_manifest)
        for s in 1:100
            s == seed && continue
            relative = joinpath("instances", "seed_$s.mat")
            file_hash(joinpath(corrected, relative)) == baseline_hashes[relative] || error("untouched checkpoint changed: seed $s")
        end
        hashes(original) == baseline_hashes || error("original archive changed during the repair")
        state!(output, "validating_corrected_results")
        audit_module = execution_module(:StQOPauseRepairAudit)
        Base.include(audit_module, joinpath(source, "scripts", "analyze_remaining_run.jl"))
        analyzer = getfield(getfield(audit_module, :ACGDRemainingAnalysis), :analyze_remaining_run)
        report = Base.invokelatest(analyzer, corrected)
        report["validation_passed"] || error("corrected full-result validation failed; see analysis_report.txt")
        report["scope"] == "full" || error("corrected result is not a full benchmark")
        hashes(original) == baseline_hashes || error("original archive changed during validation")
        rm(joinpath(corrected, "REPAIR_VALIDATION_PENDING.txt"))
        write_toml(joinpath(output, "corrected_output_checksums.toml"), hashes(corrected))
        provenance["completed_utc"] = string(now(UTC))
        provenance["corrected_results"] = corrected
        provenance["validation_passed"] = true
        provenance["untouched_checkpoint_count"] = 99
        write_toml(joinpath(output, "repair_provenance.toml"), provenance)
        state!(output, "complete"; extra=Dict("corrected_results" => corrected, "validation_passed" => true))
        println("Completed seed 54 timing repair: ", corrected)
        return corrected
    catch exception
        state!(output, "failed"; extra=Dict("error" => sprint(showerror, exception)))
        rethrow()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if ARGS == ["--help"]
        println("Usage: julia --project=FROZEN_SOURCE scripts/repair_stqo_pause.jl --source-dir FROZEN_SOURCE --original-dir ORIGINAL_STQO_RESULTS --output-dir FRESH_REPAIR_DIRECTORY [--seed 54]")
    else
        repair_stqo(read_arguments(ARGS))
    end
end
