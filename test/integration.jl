# Run with the resolved repository environment. No dataset downloads or plots.
using Test, LinearAlgebra, Random, Statistics, SparseArrays
using Graphs, Zygote, MAT
include(joinpath(@__DIR__, "..", "src", "ACGDSolvers.jl"))
include(joinpath(@__DIR__, "..", "scripts", "benchmark_support.jl"))
using .ACGDSolvers, .ACGDBenchmarkSupport
include("load_definitions.jl")

function ad_definitions(filename)
    definitions = load_problem_definitions(joinpath(@__DIR__, "..", filename))
    Core.eval(definitions, :(using Graphs, Zygote))
    return definitions
end

function check_gradient(problem)
    x = copy(problem.x0)
    d = collect(range(-0.5, 0.75; length=length(x)))
    d ./= norm(d)
    h = 1e-6
    actual = (problem.f(x + h*d) - problem.f(x - h*d)) / (2h)
    @test dot(problem.grad_f(x), d) ≈ actual rtol=1e-5 atol=1e-7
end

@testset "Matrix balancing: actual Zygote gradient and feasibility" begin
    definitions = ad_definitions("matrix_balancing.jl")
    p = Base.invokelatest(definitions.generate_problem, 12; n=6, a=3, l=1.0, u=2.0)
    check_gradient(p)
    B = p.A .* exp.(p.x0 .- p.x0')
    @test p.grad_f(p.x0) ≈ vec(sum(B; dims=2)) - vec(sum(B; dims=1))
    for routine in (definitions.conditional_gradient_adaptive,
                    definitions.conditional_gradient_pure_backtracking)
        args = routine === definitions.conditional_gradient_adaptive ?
            (p.f, p.grad_f, p.lmo, p.x0, 0.25) : (p.f, p.grad_f, p.lmo, p.x0)
        r = Base.invokelatest(routine, args...; fstar=p.optimal_value, max_iter_cap=25)
        @test all(p.l - 1e-12 .<= r.x .<= p.u + 1e-12)
        @test r.final_value ≈ p.f(r.x)
        @test r.final_gap ≈ dot(p.grad_f(r.x), r.x - p.lmo(p.grad_f(r.x)))
    end
end

@testset "Traffic: normalized objective, actual oracle, AD and flow balance" begin
    definitions = ad_definitions("traffic_assignment.jl")
    p = Base.invokelatest(definitions.generate_problem, 17; m=2, n=3, p=1.0)
    @test p.normalization == p.mn^2
    @test p.n_od > 0 && p.n_links > 0
    check_gradient(p)
    c = p.n_od / p.mn
    X = reshape(p.x0, p.mn, p.mn)
    raw = sum(X[i,j] + 0.03X[i,j]^5 / c^4 for (i,j) in p.edge_indices)
    @test p.f(p.x0) ≈ raw / p.normalization
    net_demand = zeros(p.mn)
    for ((s,t), demand) in p.demands
        net_demand[s] += demand
        net_demand[t] -= demand
    end
    initial = copy(p.x0)
    for routine in (definitions.conditional_gradient_adaptive,
                    definitions.conditional_gradient_pure_backtracking)
        args = routine === definitions.conditional_gradient_adaptive ?
            (p.f, p.grad_f, p.lmo, p.x0, 0.25) : (p.f, p.grad_f, p.lmo, p.x0)
        r = Base.invokelatest(routine, args...; max_iter_cap=10)
        flows = reshape(r.x, p.mn, p.mn)
        @test minimum(flows) >= -1e-12
        @test vec(sum(flows; dims=2)) - vec(sum(flows; dims=1)) ≈ net_demand atol=1e-10
        @test all(flows[i,j] == 0 for i in 1:p.mn, j in 1:p.mn if !((i,j) in p.edge_indices))
        @test p.x0 == initial
        @test r.final_value ≈ p.f(r.x)
        @test r.final_gap ≈ dot(p.grad_f(r.x), r.x - p.lmo(p.grad_f(r.x))) atol=1e-12
    end
end

@testset "Regression generators: synthetic fixed data through actual AD paths" begin
    for filename in ("Logistic_Regression_with_Regularization.jl", "Sigmoid.jl")
        definitions = ad_definitions(filename)
        A = sparse(reshape(sin.(collect(1.0:72.0)), 12, 6))
        binary = Float64[isodd(i) for i in 1:12]
        y = filename == "Sigmoid.jl" ? binary : 2binary .- 1
        Core.eval(definitions, :(A_full = $A))
        Core.eval(definitions, :(y_full = $y))
        Core.eval(definitions, :(m_total = 12))
        Core.eval(definitions, :(n_total = 6))
        p = Base.invokelatest(definitions.generate_problem, 31; m_sub=10, n_sub=5, tau=2.0)
        check_gradient(p)
        @test isfinite(p.f(fill(1e4, 5)))
        @test all(isfinite, p.grad_f(fill(1e4, 5)))
        args = filename == "Sigmoid.jl" ? (p.f, p.grad_f, p.lmo, p.x0) :
               (p.f, p.grad_f, p.lmo, p.x0, p.rho)
        r = Base.invokelatest(definitions.conditional_gradient_pure_backtracking, args...; max_iter_cap=20)
        @test isfinite(r.final_value) && r.final_value ≈ p.f(r.x)
        if filename == "Sigmoid.jl"
            @test norm(r.x, 1) <= p.tau + 1e-12
            @test r.final_gap ≈ dot(p.grad_f(r.x), r.x - p.lmo(p.grad_f(r.x))) atol=1e-12
        else
            @test r.final_gap ≈ norm(p.grad_f(r.x))
        end
    end
end

@testset "MAT exports preserve repeat outcomes and final iterate metadata" begin
    f(x) = dot(x, x) / 2
    results = benchmark_methods(["adaptive" => () -> solve(f, identity, g -> -g / norm(g), [1.0];
                                  constrained=false, method=:adaptive, max_iter_cap=5)];
                                warmups=1, repeats=3)
    r = results["adaptive"]
    payload = merge(Dict{String,Any}("values" => r.values, "gaps" => r.gaps,
                                    "times" => r.times, "total_time" => r.total_time),
                    result_metadata(r), benchmark_metadata(r))
    mktempdir() do directory
        path = joinpath(directory, "test.mat")
        matwrite(path, Dict("runs" => Dict("seed_1" => payload)))
        restored = matread(path)["runs"]["seed_1"]
        @test vec(restored["values"]) == r.values
        @test restored["termination_iteration"] == r.termination_iteration
        @test restored["final_value"] == r.final_value
        @test length(restored["repeat_times"]) == 3
        @test length(restored["repeat_outcomes"]) == 3
        @test restored["aggregation"] == "median"
        @test restored["total_time"] == median(restored["repeat_times"])
    end
end
