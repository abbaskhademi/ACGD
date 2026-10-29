module ACGDBenchmarkSupport

using Dates, LinearAlgebra, Pkg, Random, SHA, Statistics, TOML

export benchmark_methods, benchmark_metadata, run_manifest, write_run_manifest,
       finite_performance_profile, profile_statistics, paired_speed_ratios,
       harmonic_mean, benchmark_repeats, problem_snapshot, snapshot_sha256,
       save_instance_checkpoint

"""
Return measured execution orders without using the caller's random-number stream.
Each Williams block balances positions and directed adjacent pairs. Even method
counts need n orders; odd counts need 2n orders (the cyclic square and its reverse).
Random label and row permutations preserve balance. Warmups never advance this RNG.
"""
function balanced_orders(count, repeats, order_seed, instance_index)
    count >= 1 || throw(ArgumentError("at least one method is required"))
    block_size = count == 1 ? 1 : iseven(count) ? count : 2count
    repeats > 0 && repeats % block_size == 0 ||
        throw(ArgumentError("balanced order for $count methods requires repeats divisible by $block_size"))
    digest = sha256("ACGD balanced orders:$(order_seed):$(instance_index)")
    seed = foldl((acc, byte) -> (acc << 8) | UInt64(byte), digest[1:8]; init=UInt64(0))
    rng = Xoshiro(seed)
    base = [isodd(j) ? count - div(j - 3, 2) : div(j, 2) + 1 for j in 1:count]
    base[1] = 1
    square = [[mod1(symbol + row - 1, count) for symbol in base] for row in 1:count]
    design = isodd(count) && count > 1 ? vcat(square, reverse.(square)) : square
    orders = Vector{Int}[]
    for _ in 1:div(repeats, block_size)
        labels = randperm(rng, count)
        rows = randperm(rng, block_size)
        append!(orders, [[labels[symbol] for symbol in design[row]] for row in rows])
    end
    return orders
end

"Default to a complete balanced block; explicit overrides must still balance."
benchmark_repeats(count::Integer) = parse(Int, get(ENV, "ACGD_REPEATS",
    string(count == 1 ? 1 : iseven(count) ? count : 2count)))

# Numerical inputs only: functions and custom objects are not portable MAT data.
snapshot_value(value::Bool) = Int64(value)
snapshot_value(value::Integer) = Int64(value)
snapshot_value(value::AbstractFloat) = Float64(value)
snapshot_value(value::AbstractString) = String(value)
snapshot_value(value::Symbol) = String(value)
snapshot_value(value::AbstractArray{<:Integer}) = Int64.(Array(value))
snapshot_value(value::AbstractArray{<:AbstractFloat}) = Float64.(Array(value))
snapshot_value(value::Tuple) = problem_snapshot(Dict("component_$i" => v for (i, v) in enumerate(value)))
snapshot_value(value::NamedTuple) = problem_snapshot(value)
snapshot_value(value::AbstractDict) = all(k -> k isa Union{Symbol,AbstractString}, keys(value)) ? problem_snapshot(value) : nothing
snapshot_value(value) = nothing

"Archive numeric data and metadata without serializing closures or graph objects."
function problem_snapshot(problem::Union{NamedTuple,AbstractDict})
    result = Dict{String,Any}()
    for (key, value) in pairs(problem)
        saved = snapshot_value(value)
        saved === nothing || (result[string(key)] = saved)
    end
    return result
end

"Canonical checksum of a numerical snapshot, stable across its MAT round trip."
function snapshot_sha256(data::AbstractDict)
    io = IOBuffer()
    function encode(value)
        if value isa AbstractDict
            keys_ = sort!(String.(collect(keys(value))))
            print(io, "D", length(keys_), ":")
            for key in keys_
                encode(key)
                encode(value[key])
            end
        elseif value isa AbstractString
            bytes = codeunits(String(value))
            print(io, "S", length(bytes), ":")
            write(io, bytes)
        elseif value isa Real || value isa AbstractArray{<:Real}
            array = value isa Real ? fill(Float64(value), 1, 1) : Float64.(Array(value))
            dimensions = isempty(array) ? (0,) : ndims(array) == 1 ? (length(array), 1) : size(array)
            print(io, "N", join(dimensions, ","), ":")
            write(io, reinterpret(UInt8, vec(array)))
        else
            throw(ArgumentError("unsupported snapshot value $(typeof(value))"))
        end
    end
    encode(data)
    return bytes2hex(sha256(take!(io)))
end

"Write a completed instance atomically; the driver supplies MAT.matwrite."
function save_instance_checkpoint(path, data; writer)
    mkpath(dirname(path))
    temporary = path * ".tmp"
    writer(temporary, data)
    mv(temporary, path; force=true)
    return path
end

"""
Warm every solver path, then repeat paired calls. Defaults preserve the original
rotating order and solver-reported complete elapsed time. `external_timing=true`
uses `@timed` around the complete supplied call, retaining the core solver time
separately. Allocations, garbage collection and compilation during that call are
included. Optional `gc_before_call` collects outside the clock before every call.

`order_mode=:rotate` requires complete Williams blocks (n repeats for even n,
2n for odd n). Its
measured Williams orders are independent of the number of warmups. Every warmup
round calls all paths before measurement begins. The local order RNG is keyed by
`order_seed` and `instance_index` without changing the caller's RNG.

Each call must construct its own solver state from the same immutable instance.
The median is the reported time; success requires every measured repeat to succeed.
Histories and final values belong to the repeat nearest the reported median.
Warmups, instance generation, report formatting, plotting and writes are untimed.
"""
function benchmark_methods(calls; warmups=1, repeats=3, instance_index=1,
                           external_timing=false, gc_before_call=false,
                           order_mode=:rotate, order_seed=20261021,
                           progress=get(ENV, "ACGD_PROGRESS", "false") == "true")
    warmups >= 1 || throw(ArgumentError("at least one warmup is required"))
    repeats >= 1 || throw(ArgumentError("at least one repeat is required"))
    order_mode in (:rotate, :balanced) || throw(ArgumentError("unknown execution order mode"))
    entries = collect(pairs(calls))
    if calls isa AbstractVector
        entries = collect(calls)
    end
    isempty(entries) && throw(ArgumentError("no solver calls supplied"))
    names = String[first(p) for p in entries]
    length(unique(names)) == length(names) || throw(ArgumentError("duplicate method name"))
    outcomes = [Any[] for _ in entries]
    count = length(entries)
    rotating_order(repetition) = [mod1(j + mod(instance_index + repetition - 2, count), count)
                                  for j in 1:count]
    warmup_orders = [rotating_order(repetition) for repetition in 1:warmups]
    measured_orders = order_mode == :balanced ?
        balanced_orders(count, repeats, order_seed, instance_index) :
        [rotating_order(warmups + repetition) for repetition in 1:repeats]
    policy = Dict{String,Any}(
        "clock" => external_timing ? "external_complete_call" : "solver_reported_complete_elapsed",
        "external_timing" => external_timing, "gc_before_call" => gc_before_call,
        "gc_inside_call_included" => true, "compilation_inside_call_included" => true,
        "order_mode" => String(order_mode), "order_seed" => order_seed,
        "instance_index" => instance_index,
        "order_design" => order_mode == :balanced ?
            "Williams design; independent label and row permutations per block" :
            "cyclic rotation across warmups and measured repeats",
        "measured_order_depends_on_warmup_count" => order_mode == :rotate,
        "warmup_orders" => Dict("warmup_$(i)" => names[order] for (i, order) in enumerate(warmup_orders)),
        "measured_orders" => Dict("repeat_$(i)" => names[order] for (i, order) in enumerate(measured_orders)),
        "aggregation" => "median", "history_selection" => "repeat nearest reported elapsed median",
    )
    for (repetition, order) in enumerate(vcat(warmup_orders, measured_orders))
        for index in order
            gc_before_call && GC.gc(true)
            result = if external_timing
                measured = @timed last(entries[index])()
                merge(measured.value, (
                    core_elapsed_time=Float64(measured.value.total_time),
                    total_time=Float64(measured.time), elapsed_time=Float64(measured.time),
                    outer_elapsed_time=Float64(measured.time),
                    gc_time=Float64(measured.gctime), allocated_bytes=measured.bytes,
                    compile_time=Float64(measured.compile_time),
                    recompile_time=Float64(measured.recompile_time),
                ))
            else
                last(entries[index])()
            end
            if progress
                phase = repetition <= warmups ? "warmup" : "measured"
                repeat_index = repetition <= warmups ? repetition : repetition - warmups
                println("[", now(UTC), "] ", phase, " repeat=", repeat_index,
                        " method=", names[index], " status=", result.status,
                        " iterations=", result.k, " elapsed=", result.total_time)
                flush(stdout)
            end
            repetition > warmups && push!(outcomes[index], result)
        end
    end
    results = Dict{String,Any}()
    for (index, name) in enumerate(names)
        samples = outcomes[index]
        elapsed = Float64[r.total_time for r in samples]
        all(t -> isfinite(t) && t >= 0, elapsed) ||
            throw(ArgumentError("solver elapsed times must be finite and nonnegative"))
        aggregate = median(elapsed)
        representative = argmin(abs.(elapsed .- aggregate))
        result = samples[representative]
        success = all(r -> r.solved, samples)
        repeated = [Dict{String,Any}(
            "repeat" => i, "elapsed_time" => Float64(r.total_time),
            "iteration_time" => Float64(r.iteration_time),
            "initialization_time" => Float64(r.initialization_time),
            "status" => String(r.status), "solved" => r.solved,
            "iterations" => r.k, "final_value" => r.final_value,
            "final_gap" => r.final_gap, "final_residual" => r.final_residual,
            "termination_iteration" => r.termination_iteration,
            "stopping_criterion" => String(r.stopping_criterion),
            "execution_order" => names[measured_orders[i]],
            "execution_position" => findfirst(==(index), measured_orders[i]),
        ) for (i, r) in enumerate(samples)]
        timing_summary = NamedTuple()
        if external_timing
            for (raw, sample) in zip(repeated, samples)
                for field in (:core_elapsed_time, :outer_elapsed_time, :gc_time,
                              :allocated_bytes, :compile_time, :recompile_time)
                    raw[string(field)] = getproperty(sample, field)
                end
            end
            timing_summary = (
                outer_elapsed_time=aggregate,
                core_elapsed_time=median([r.core_elapsed_time for r in samples]),
                gc_time=median([r.gc_time for r in samples]),
                allocated_bytes=median([r.allocated_bytes for r in samples]),
                compile_time=median([r.compile_time for r in samples]),
                recompile_time=median([r.recompile_time for r in samples]),
            )
        end
        results[name] = merge(result, (
            total_time=aggregate, elapsed_time=aggregate,
            iteration_time=median([r.iteration_time for r in samples]),
            initialization_time=median([r.initialization_time for r in samples]),
            solved=success,
            status=success ? result.status : "repeat_failure",
            raw_repeats=repeated, representative_repeat=representative,
            warmups=warmups, repeats=repeats, aggregation="median",
            measurement_policy=policy,
        ), timing_summary)
    end
    return results
end

"Export repeat metadata; histories remain those of representative_repeat."
function benchmark_metadata(result)
    hasproperty(result, :raw_repeats) || return Dict{String,Any}()
    metadata = Dict{String,Any}(
        "repeat_outcomes" => Dict("repeat_$(r["repeat"])" => r for r in result.raw_repeats),
        "repeat_times" => [r["elapsed_time"] for r in result.raw_repeats],
        "repeat_statuses" => [r["status"] for r in result.raw_repeats],
        "representative_repeat" => result.representative_repeat,
        "warmups" => result.warmups, "repeats" => result.repeats,
        "aggregation" => result.aggregation,
        "history_timing" => "representative repeat; scalar times are repeat medians",
    )
    hasproperty(result, :measurement_policy) &&
        (metadata["measurement_policy"] = result.measurement_policy)
    if hasproperty(result, :outer_elapsed_time)
        for field in (:outer_elapsed_time, :core_elapsed_time, :gc_time,
                      :allocated_bytes, :compile_time, :recompile_time)
            metadata[string(field)] = getproperty(result, field)
        end
        metadata["repeat_core_times"] = [r["core_elapsed_time"] for r in result.raw_repeats]
        metadata["repeat_outer_times"] = [r["outer_elapsed_time"] for r in result.raw_repeats]
    end
    return metadata
end

function git_output(root, arguments...)
    try
        return strip(read(Cmd(["git", "-C", root, arguments...]), String))
    catch
        return "unavailable"
    end
end

# TOML has no null or Symbol values; normalize arbitrary benchmark configuration.
toml_value(value::NamedTuple) = Dict(string(k) => toml_value(v) for (k, v) in pairs(value))
toml_value(value::AbstractDict) = Dict(string(k) => toml_value(v) for (k, v) in value)
toml_value(value::Tuple) = [toml_value(v) for v in value]
toml_value(value::AbstractArray) = [toml_value(v) for v in value]
toml_value(value::Symbol) = string(value)
toml_value(::Nothing) = "nothing"
toml_value(value) = value

"Capture actual runtime, source hashes and configuration without changing them."
function run_manifest(config; root=normpath(joinpath(@__DIR__, "..")))
    hashes = Dict{String,String}()
    for (directory, subdirectories, files) in walkdir(root)
        filter!(name -> !(name in (".git", "corrected_results")), subdirectories)
        for file in files
            if endswith(file, ".jl") || endswith(file, ".toml") || file == "requirements-plot.txt"
                path = joinpath(directory, file)
                hashes[relpath(path, root)] = bytes2hex(sha256(read(path)))
            end
        end
    end
    project = Base.active_project()
    manifest = project === nothing ? "" : joinpath(dirname(project), "Manifest.toml")
    inputs = Dict{String,String}()
    for relative in ("movielens100k.csv", "video_colocalization/aeroplane_data_small.mat")
        path = joinpath(root, relative)
        isfile(path) && (inputs[relative] = bytes2hex(sha256(read(path))))
    end
    dataset_directory = get(ENV, "LIBSVMDATA_HOME", "")
    for name in ("a9a", "w8a")
        path = joinpath(dataset_directory, name)
        !isempty(dataset_directory) && isfile(path) &&
            (inputs["libsvm/" * name] = bytes2hex(sha256(read(path))))
    end
    plotting = Dict{String,String}("status" => "PyPlot not loaded")
    for (identifier, loaded) in Base.loaded_modules
        identifier.name == "PyPlot" || continue
        try
            pycall = getproperty(loaded, :PyCall)
            plotting = Dict("status" => "loaded",
                            "python" => string(getproperty(pycall, :pyversion)),
                            "matplotlib" => string(getproperty(getproperty(loaded, :matplotlib), :__version__)))
        catch exception
            plotting = Dict("status" => "loaded; version lookup failed", "error" => sprint(showerror, exception))
        end
    end
    packages = try
        Dict(info.name => (info.version === nothing ? "stdlib/unversioned" : string(info.version))
             for info in values(Pkg.dependencies()))
    catch
        Dict("status" => "unavailable: no resolved project environment")
    end
    return Dict{String,Any}(
        "created_utc" => string(now(UTC)),
        "source_revision" => git_output(root, "rev-parse", "HEAD"),
        "source_dirty" => !isempty(git_output(root, "status", "--porcelain")),
        "source_sha256" => hashes,
        "input_file_sha256" => inputs,
        "plotting_environment" => plotting,
        "julia_version" => string(VERSION), "julia_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "blas_configuration" => sprint(show, BLAS.get_config()),
        "cpu_name" => Sys.CPU_NAME, "architecture" => string(Sys.ARCH),
        "kernel" => string(Sys.KERNEL), "word_size" => Sys.WORD_SIZE,
        "active_project" => something(project, "none"),
        "manifest_sha256" => isfile(manifest) ? bytes2hex(sha256(read(manifest))) : "missing",
        "package_versions" => packages,
        "thread_environment" => Dict(key => get(ENV, key, "unset") for key in
            ("JULIA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")),
        "timing_convention" => "complete solver elapsed, including initialization, stopping and in-memory histories",
        "failure_policy" => "any failed measured repeat makes the method unsuccessful",
        "config" => toml_value(config),
    )
end

function write_run_manifest(path, config; root=normpath(joinpath(@__DIR__, "..")))
    metadata = run_manifest(config; root=root)
    mkpath(dirname(path))
    open(path, "w") do io
        TOML.print(io, metadata; sorted=true)
    end
    return metadata
end

"""
Construct empirical performance profiles without assigning finite ratios to failures.

Rows are instances; columns are methods. Nonfinite entries are failures. Finite
entries must be strictly positive. All-failed rows stay in the denominator.
`max_ratio` limits finite-ratio coverage, which can differ from overall success.
For `logscale=true`, x coordinates are log2(ratio), not the ratio itself.
"""
function finite_performance_profile(times::AbstractMatrix; logscale=true, max_ratio=nothing)
    np, ns = size(times)
    np > 0 && ns > 0 || throw(ArgumentError("performance matrix must not be empty"))
    any(t -> isfinite(t) && t <= 0, times) &&
        throw(ArgumentError("finite solver times must be positive; do not shift zero times"))
    ratios = fill(Inf, np, ns)
    for p in 1:np
        successful = filter(isfinite, times[p, :])
        isempty(successful) && continue
        best = minimum(successful)
        for s in 1:ns
            isfinite(times[p, s]) && (ratios[p, s] = times[p, s] / best)
        end
    end
    finite = filter(isfinite, vec(ratios))
    cutoff = max_ratio === nothing ? (isempty(finite) ? 1.0 : maximum(finite)) : Float64(max_ratio)
    isfinite(cutoff) && cutoff >= 1 || throw(ArgumentError("max_ratio must be finite and at least one"))
    thresholds = sort!(unique(vcat([1.0, cutoff], filter(r -> r <= cutoff, finite))))
    fractions = [count(r -> r <= threshold, ratios[:, s]) / np
                 for threshold in thresholds, s in 1:ns]
    counts = [count(isfinite, times[:, s]) for s in 1:ns]
    return (x=logscale ? log2.(thresholds) : thresholds,
            thresholds=thresholds, fractions=fractions, ratios=ratios,
            solved_counts=counts, success_rates=counts ./ np,
            finite_ratio_coverage=vec(fractions[end, :]),
            all_failed_instances=count(p -> all(t -> !isfinite(t), times[p, :]), 1:np),
            logscale=logscale, max_ratio=cutoff)
end

profile_statistics(times; max_ratio=nothing) = let p = finite_performance_profile(times; max_ratio=max_ratio)
    (solved_counts=p.solved_counts, success_rates=p.success_rates,
     finite_ratio_coverage=p.finite_ratio_coverage, max_ratio=p.max_ratio,
     all_failed_instances=p.all_failed_instances)
end

function harmonic_mean(values)
    isempty(values) && return NaN
    all(v -> isfinite(v) && v > 0, values) || throw(ArgumentError("harmonic mean requires finite positive values"))
    return length(values) / sum(inv, values)
end

"Baseline/reference speed ratios use only paired successes; preserve pair counts."
function paired_speed_ratios(times::AbstractMatrix; reference=1)
    1 <= reference <= size(times, 2) || throw(ArgumentError("invalid reference column"))
    finite_performance_profile(times) # Validate times with the same policy.
    return [let pairs = [times[p, s] / times[p, reference] for p in axes(times, 1)
                         if isfinite(times[p, s]) && isfinite(times[p, reference])]
        (paired_instances=length(pairs), total_instances=size(times, 1),
         baseline_over_reference_harmonic=harmonic_mean(pairs), ratios=pairs)
    end for s in axes(times, 2)]
end

end # module
