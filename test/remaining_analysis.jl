using Test, LinearAlgebra
include(joinpath(@__DIR__, "..", "scripts", "analyze_remaining_run.jl"))
const R = ACGDRemainingAnalysis

@testset "Problem-specific benchmark artifact selection" begin
    mktempdir() do directory
        # Matrix factorization retains the driver's established prefixed filename.
        # Keep a standard-name decoy to ensure analysis selects the intended data.
        R.MAT.matwrite(joinpath(directory, "matrixfactorization_benchmark_all_methods.mat"), Dict("seeds" => [11]))
        R.MAT.matwrite(joinpath(directory, "benchmark_all_methods.mat"), Dict("seeds" => [22]))
        factorization = R.MAT.matread(R.benchmark_file_path(directory, "matrix_factorization"))
        @test R.integers(factorization["seeds"]) == [11]
        for problem in keys(R.SPECS)
            problem == "matrix_factorization" && continue
            benchmark = R.MAT.matread(R.benchmark_file_path(directory, problem))
            @test R.integers(benchmark["seeds"]) == [22]
        end
    end
end

@testset "Independent saved-iterate diagnostics" begin
    ls = Dict("A" => [1.0 0; 0 2], "b" => [1.0, 1.0])
    d = R.diagnostics("least_squares", ls, [1.0, 0.0])
    @test d.value == 1.0
    @test d.gap == 4.0
    @test d.feasibility == 0.0

    logistic = Dict("A" => [1.0 0; 0 1], "y" => [1.0, -1.0], "omega" => 0.1)
    d = R.diagnostics("Logistic_Regression_with_Regularization", logistic, zeros(2))
    @test d.value ≈ log(2)
    @test d.gap ≈ sqrt(2) / 4
    @test isfinite(R.stable_softplus(1000.0))
    @test R.stable_sigmoid(-1000.0) == 0

    sigmoid = Dict("A" => [1.0 0; 0 1], "y" => [1.0, 0.0], "tau" => 2.0)
    d = R.diagnostics("Sigmoid_w8a", sigmoid, zeros(2))
    @test d.value == 0.25
    @test d.gap == 0.25
    @test d.feasibility == 0

    stqo = Dict("Q" => [1.0 0; 0 2])
    d = R.diagnostics("NonconvexQP_Simplex", stqo, [1.0, 0.0])
    @test d.value == 1.0
    @test d.gap == 2.0
    @test d.feasibility == 0

    balancing = Dict("A" => [1.0 2; 2 1], "l" => 1.0, "u" => 10.0)
    d = R.diagnostics("matrix_balancing", balancing, [2.0, 2.0])
    @test d.value == 6.0
    @test d.gap == 0.0
    @test d.feasibility == 0
    d = R.diagnostics("matrix_balancing", balancing, [2.0, 3.0])
    @test d.gap ≈ 40sinh(1.0)

    factorization = Dict("Y" => zeros(2, 2), "tau1" => 2.0, "tau2" => 3.0)
    d = R.diagnostics("matrix_factorization", factorization, (ones(2, 1), ones(2, 1)))
    @test d.value == 2.0
    @test d.gap ≈ 8 + 10sqrt(2)
    @test d.feasibility == 0

    # One OD pair has two routes; the archive places its flow on the costly direct edge.
    traffic = Dict("mn" => 3, "edge_sources" => [1, 1, 2], "edge_targets" => [2, 3, 3],
        "demand_sources" => [1], "demand_targets" => [3], "demand_values" => [10.0],
        "c" => 1.0, "normalization" => 9.0)
    X = zeros(3, 3); X[1, 3] = 10.0
    d = R.diagnostics("traffic_assignment", traffic, vec(X))
    @test d.value ≈ (10 + 0.03 * 10^5) / 9
    @test d.gap ≈ 10 * ((1 + 0.15 * 10^4) - 2) / 9
    @test d.feasibility == 0

    @test R.expected_criterion(1.0, 1.0, 1.0, 1e-5, "objective") == ("objective", 0.0)
    @test R.expected_criterion(1.0, 0.0, 0.0, 1e-5, "either") == ("stationarity", 0.0)
    @test R.expected_criterion(1.0, 0.0, 1.0, 1e-5, "either") == ("objective", 0.0)
    @test R.expected_criterion(1.0, 1.0, NaN, 1e-3, "stationarity") == ("none", 1.0)
end
