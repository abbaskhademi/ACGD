using Test, Random, SparseArrays, MAT
include(joinpath(@__DIR__, "..", "scripts", "benchmark_support.jl"))
using .ACGDBenchmarkSupport

@testset "General Williams schedules" begin
    for n in (1, 2, 3, 5, 6, 8)
        block_size = n == 1 ? 1 : iseven(n) ? n : 2n
        multiplicity = div(block_size, n)
        orders = ACGDBenchmarkSupport.balanced_orders(n, 2block_size, 20261021, 7)
        @test length(orders) == 2block_size
        for start in (1, block_size + 1)
            block = orders[start:start + block_size - 1]
            @test all(row -> sort(row) == collect(1:n), block)
            for position in 1:n, method in 1:n
                @test count(row -> row[position] == method, block) == multiplicity
            end
            for first in 1:n, second in 1:n
                first == second && continue
                @test sum(count(i -> row[i] == first && row[i+1] == second, 1:n-1)
                          for row in block) == multiplicity
            end
        end
        Random.seed!(123)
        expected = rand()
        Random.seed!(123)
        ACGDBenchmarkSupport.balanced_orders(n, block_size, 7, 12)
        @test rand() == expected
        @test_throws ArgumentError ACGDBenchmarkSupport.balanced_orders(n, 0, 7, 12)
    end
    @test_throws ArgumentError ACGDBenchmarkSupport.balanced_orders(5, 6, 7, 12)
end

@testset "Numerical checkpoints survive MAT round trip" begin
    problem = (A=sparse([1.0 0 2; 0 3 0]), x0=[0.0, 1.0, 2.0],
               seed=23, flag=true, value=NaN, empty=Float64[],
               dataset="public input", factors=(ones(2, 1), zeros(3, 1)),
               callback=identity, unsupported=Dict((1,2) => 3.0))
    snapshot = problem_snapshot(problem)
    @test !haskey(snapshot, "callback")
    @test !haskey(snapshot, "unsupported")
    @test snapshot["A"] == Matrix(problem.A)
    digest = snapshot_sha256(snapshot)
    @test snapshot_sha256(Dict(reverse(collect(snapshot)))) == digest
    mktempdir() do directory
        path = joinpath(directory, "instances", "seed_23.mat")
        save_instance_checkpoint(path, Dict("problem" => snapshot); writer=matwrite)
        @test isfile(path) && !isfile(path * ".tmp")
        reread = matread(path)["problem"]
        @test snapshot_sha256(reread) == digest
        reread["A"][1, 1] += 1
        @test snapshot_sha256(reread) != digest
        @test_throws ErrorException save_instance_checkpoint(path, Dict();
            writer=(file, data) -> error("interrupted write"))
        @test snapshot_sha256(matread(path)["problem"]) == digest
    end
end
