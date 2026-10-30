using Test, Random, Statistics
include(joinpath(@__DIR__, "..", "scripts", "benchmark_support.jl"))
using .ACGDBenchmarkSupport

function mock_result(time=1e-6)
    return (total_time=time, elapsed_time=time, iteration_time=0.75time,
            initialization_time=0.25time, solved=true, status="converged",
            k=1, final_value=0.0, final_gap=0.0, final_residual=0.0,
            termination_iteration=1, stopping_criterion="objective")
end

@testset "Default timing and rotation remain compatible" begin
    order = String[]
    counts = Dict("a" => 0, "b" => 0)
    function measured(name)
        push!(order, name)
        counts[name] += 1
        return mock_result([99.0, 1.0, 3.0, 2.0][counts[name]])
    end
    results = benchmark_methods(["a" => () -> measured("a"), "b" => () -> measured("b")])
    @test order == ["a", "b", "b", "a", "a", "b", "b", "a"]
    @test results["a"].total_time == 2.0
    @test results["a"].representative_repeat == 3
    @test benchmark_metadata(results["a"])["repeat_times"] == [1.0, 3.0, 2.0]
    @test !hasproperty(results["a"], :core_elapsed_time)
    @test results["a"].measurement_policy["clock"] == "solver_reported_complete_elapsed"
end

@testset "Williams blocks balance position and directed adjacent pairs" begin
    names = string.(1:6)
    calls = [name => () -> mock_result() for name in names]
    for repeats in (6, 12), instance in (1, 2)
        order = String[]
        tracked_calls = [name => () -> (push!(order, name); mock_result()) for name in names]
        Random.seed!(841)
        expected_next = rand()
        Random.seed!(841)
        results = benchmark_methods(tracked_calls; warmups=2, repeats=repeats,
                                    instance_index=instance, order_mode=:balanced)
        @test rand() == expected_next
        orders = [results["1"].raw_repeats[i]["execution_order"] for i in 1:repeats]
        @test order[1:12] == vcat([results["1"].measurement_policy["warmup_orders"]["warmup_$i"] for i in 1:2]...)
        @test order[13:end] == vcat(orders...)
        for start in 1:6:repeats
            block = orders[start:start+5]
            @test all(position -> sort([row[position] for row in block]) == names, 1:6)
            adjacent = [(row[j], row[j+1]) for row in block for j in 1:5]
            @test length(unique(adjacent)) == 30
            @test Set(adjacent) == Set((a, b) for a in names for b in names if a != b)
        end
        for name in names, i in 1:repeats
            raw = results[name].raw_repeats[i]
            @test raw["execution_order"][raw["execution_position"]] == name
        end
        other = benchmark_methods(calls; warmups=1, repeats=repeats,
                                  instance_index=instance, order_mode=:balanced)
        @test other["1"].measurement_policy["measured_orders"] ==
              results["1"].measurement_policy["measured_orders"]
    end
    @test_throws ArgumentError benchmark_methods(calls; repeats=3, order_mode=:balanced)
    @test_throws ArgumentError benchmark_methods(calls[1:5]; repeats=6, order_mode=:balanced)
    @test_throws ArgumentError benchmark_methods(calls; order_mode=:unknown)
end

@testset "Complete call is measured and raw diagnostics survive aggregation" begin
    function wrapped_call()
        sleep(0.002) # Deliberate work before the mock solver's internally timed body.
        return mock_result()
    end
    result = benchmark_methods(["wrapped" => wrapped_call]; warmups=2, repeats=3,
                               external_timing=true, gc_before_call=true)["wrapped"]
    metadata = benchmark_metadata(result)
    @test result.total_time == result.elapsed_time == result.outer_elapsed_time
    @test result.total_time >= 0.002
    @test result.core_elapsed_time == 1e-6
    @test metadata["outer_elapsed_time"] == median(metadata["repeat_outer_times"])
    @test metadata["core_elapsed_time"] == median(metadata["repeat_core_times"])
    @test metadata["measurement_policy"]["clock"] == "external_complete_call"
    @test metadata["measurement_policy"]["gc_before_call"]
    for raw in result.raw_repeats
        @test raw["elapsed_time"] == raw["outer_elapsed_time"]
        @test raw["elapsed_time"] >= 0.002
        @test raw["core_elapsed_time"] == 1e-6
        @test 0 <= raw["gc_time"] <= raw["elapsed_time"]
        @test raw["allocated_bytes"] >= 0
        @test raw["compile_time"] >= 0
        @test raw["recompile_time"] >= 0
    end
    @test result.compile_time == 0.0
    @test result.recompile_time == 0.0
end
