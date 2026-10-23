using Test, LinearAlgebra, Random, Statistics, TOML
include(joinpath(@__DIR__, "..", "src", "ACGDSolvers.jl"))
include(joinpath(@__DIR__, "..", "scripts", "benchmark_support.jl"))
using .ACGDSolvers, .ACGDBenchmarkSupport
include("load_definitions.jl")

const METHODS = (:adaptive, :adjustable, :backtracking, :auto, :short, :open, :adabb)
ball_oracle(g) = -g / norm(g)
quadratic(x) = dot(x, x) / 2

@testset "Exactly ten accepted iterations per scaling block" begin
    gamma, total = 0.25, 0
    for k in 1:9
        gamma, total = update_scaling(gamma, 0, k, total)
        @test gamma == 0.25
    end
    gamma, total = update_scaling(gamma, 0, 10, total)
    @test gamma ≈ 0.225
    @test total == 0
    for k in 11:20
        gamma, total = update_scaling(gamma, k == 11 ? 11 : 0, k, total)
        k < 20 && @test gamma ≈ 0.225
    end
    @test gamma ≈ 0.2475
    @test total == 0
    # Exactly ten rejected trials must not cross the strictly >10 threshold.
    for k in 21:30
        gamma, total = update_scaling(gamma, 1, k, total)
    end
    @test gamma ≈ 0.2475
    @test total == 0
    gamma, total = update_scaling(0.99, 11, 10, 0)
    @test gamma == 1.0
    @test total == 0
end

@testset "Stationary initial point and shared limits" begin
    for method in METHODS
        oracle_called = Ref(false)
        oracle(g) = (oracle_called[] = true; error("stationary oracle must not be called"))
        r = solve(quadratic, identity, oracle, zeros(2);
                  method=method, constrained=method != :adabb, L=1.0,
                  stop_rule=:stationarity)
        @test r.solved && r.k == 0
        @test !oracle_called[]
        @test r.final_gap == 0
        @test r.termination_iteration == 0
        @test r.objective_evaluations == 1
        @test r.iterate_indices == [0]
        @test r.values == [0.0] && r.gaps == [0.0]
        @test r.total_time == r.elapsed_time >= r.initialization_time > 0

        for (limits, expected) in (( (; max_iter_cap=0), "iteration_limit"),
                                   ((; max_time=0.0), "time_limit"))
            limited = solve(quadratic, identity, ball_oracle, [2.0, 1.0];
                            method=method, constrained=false, L=1.0,
                            epsilon=0.0, stop_rule=:stationarity, limits...)
            @test !limited.solved && limited.k == 0
            @test limited.status == expected
            @test limited.objective_evaluations == 1
        end
    end
end

@testset "Accepted objectives are cached; histories describe the returned iterate" begin
    A = Diagonal([1.0, 3.0])
    for method in (:adaptive, :adjustable, :backtracking)
        calls = Ref(0)
        f(x) = (calls[] += 1; dot(x, A * x) / 2)
        r = solve(f, x -> A * x, ball_oracle, [2.0, 1.0];
                  method=method, constrained=false, epsilon=0.0,
                  stop_rule=:stationarity, max_iter_cap=7)
        @test r.objective_evaluations == calls[] == 1 + r.k + sum(r.backtracks)
        @test r.k == 7
        @test length(r.values) == length(r.gaps) == length(r.iterate_indices) == r.k + 1
        @test r.iterate_indices == collect(0:r.k)
        @test r.final_value == last(r.values) ≈ dot(r.x, A * r.x) / 2
        @test r.final_gap == last(r.gaps) ≈ norm(A * r.x)
        @test r.termination_iteration == r.k
        @test length(r.L_ks) == length(r.backtracks) == r.k
        @test all(isfinite, r.L_ks)
    end
end

@testset "Post-step objective stopping and accepted curvature for tuple states" begin
    f(z) = (sum(abs2, z[1]) + sum(abs2, z[2])) / 2
    grad(z) = z
    lmo(z) = (-sign.(z[1]), -2sign.(z[2]))
    r = solve(f, grad, lmo, ([1.0], [2.0]); method=:backtracking,
              eta=0.5, beta=2.0, stop_rule=:objective, fstar=0.0, epsilon=0.7)
    @test r.solved && r.k == 1
    @test r.stopping_criterion == "objective"
    @test r.final_residual <= 0.7
    # Infer accepted curvature from the actual step: the logged value must be
    # this value, not its eta-scaled initial estimate for the next iteration.
    step = (1.0 - r.x[1][1]) / 2
    @test r.L_ks[1] ≈ 0.5 / step
    @test r.objective_evaluations == 1 + r.k + sum(r.backtracks)
    @test r.termination_iteration == 1
    @test last(r.values) == r.final_value == f(r.x)
end

@testset "Degenerate and nonfinite calculations terminate" begin
    for method in (:adaptive, :adjustable, :backtracking, :auto)
        # A linear objective has zero probe curvature; the positive floor permits a step.
        r = solve(x -> x[1], x -> [1.0, 0.0], g -> [0.0, 1.0], [1.0, 0.0];
                  method=method, stop_rule=:stationarity, epsilon=0.0, max_iter_cap=5)
        @test r.solved && r.k == 1
        @test r.final_value == 0.0
        @test all(isfinite, r.L_ks)
    end

    zero_direction = solve(x -> x[1], x -> ones(1), g -> zeros(1), [1.0];
                           constrained=false, stop_rule=:objective, fstar=0.0)
    @test !zero_direction.solved && zero_direction.k == 0
    @test zero_direction.status == "stationary_without_target"

    nonfinite = solve(x -> NaN, identity, ball_oracle, [1.0]; stop_rule=:stationarity)
    @test !nonfinite.solved && nonfinite.status == "nonfinite"
    invalid_initial = solve(x -> NaN, x -> error("invalid objective must stop before gradient"),
                            g -> error("invalid objective must stop before oracle"), [1.0])
    @test !invalid_initial.solved && invalid_initial.status == "nonfinite"
    nonfinite_grad = solve(quadratic, x -> [NaN], ball_oracle, [1.0])
    @test !nonfinite_grad.solved && nonfinite_grad.status == "nonfinite"
    probe = solve(quadratic, x -> x == [1.0] ? [1.0] : [NaN], ball_oracle, [1.0])
    @test !probe.solved && probe.status == "nonfinite_probe"

    for method in (:adaptive, :adjustable, :backtracking)
        r = solve(x -> x == [1.0] ? 1.0 : Inf, x -> ones(1), ball_oracle, [1.0];
                  method=method, constrained=false, max_backtracks=3,
                  stop_rule=:stationarity, epsilon=0.0)
        @test !r.solved && r.k == 0
        @test r.status == "backtracking_limit"
        @test r.failed_backtracks <= 4
        @test r.objective_evaluations <= 5
        @test r.final_value == 1.0
        @test isempty(r.L_ks)
    end

    adabb = solve(x -> -dot(x, x) / 2, x -> -x, ball_oracle, [1.0];
                  method=:adabb, constrained=false, alpha0=0.1,
                  stop_rule=:stationarity, epsilon=0.0, max_iter_cap=5)
    @test !adabb.solved && adabb.status == "adabb_nonpositive_curvature"
    @test adabb.k == 1
    flat = solve(sum, x -> ones(1), ball_oracle, [1.0]; method=:adabb,
                 constrained=false, alpha0=0.1, stop_rule=:stationarity,
                 epsilon=0.0, max_iter_cap=5)
    @test !flat.solved && flat.status == "adabb_zero_curvature"
end

@testset "Local probe RNG and complete elapsed timing" begin
    for method in METHODS
        Random.seed!(281)
        expected = rand(5)
        Random.seed!(281)
        r = solve(quadratic, identity, ball_oracle, [2.0, 1.0];
                  method=method, constrained=false, L=1.0, max_iter_cap=2)
        @test rand(5) == expected
        @test r.total_time >= r.initialization_time + r.iteration_time
        @test r.times[1] == r.initialization_time
    end
    for method in (:adaptive, :backtracking)
        slow_f(x) = (sleep(0.003); quadratic(x))
        slow_grad(x) = (sleep(0.003); copy(x))
        r = solve(slow_f, slow_grad, ball_oracle, [2.0, 1.0];
                  method=method, constrained=false, max_iter_cap=1)
        @test r.initialization_time >= 0.009 # objective + gradient + curvature probe
        @test r.total_time >= r.initialization_time + r.iteration_time
        @test r.iteration_time >= 0.006 # accepted objective and post-step stopping gradient
    end
    delayed_calls = Ref(0)
    late_f(x) = (delayed_calls[] += 1; delayed_calls[] > 1 && sleep(0.04); quadratic(x))
    solve(late_f, identity, ball_oracle, [1.0]; method=:short,
          L=1.0, constrained=false, max_time=Inf, fstar=0.0) # compile this path
    delayed_calls[] = 0
    late = solve(late_f, identity, ball_oracle, [1.0]; method=:short,
                 L=1.0, constrained=false, max_time=0.02, fstar=0.0)
    @test !late.solved && late.status == "time_limit"
    @test late.k == 1 && late.final_value == 0.0
    @test late.total_time >= 0.02
end

@testset "Warmup, paired rotation, repeat aggregation and failure policy" begin
    order = String[]
    counts = Dict("a" => 0, "b" => 0)
    function measured(name; failure=false)
        push!(order, name)
        counts[name] += 1
        i = counts[name]
        t = Float64(i == 1 ? 99 : (i == 2 ? 1 : (i == 3 ? 3 : 2)))
        solved = !(failure && i == 3)
        return (x=[i], values=[Float64(i)], gaps=[0.0], times=[t], k=0,
                total_time=t, elapsed_time=t, iteration_time=0.75t,
                initialization_time=0.25t, solved=solved,
                status=solved ? "converged" : "iteration_limit", final_value=Float64(i),
                final_gap=0.0, final_residual=0.0, termination_iteration=0,
                stopping_criterion="stationarity")
    end
    results = benchmark_methods(["a" => () -> measured("a"),
                                 "b" => () -> measured("b"; failure=true)];
                                warmups=1, repeats=3)
    @test order == ["a", "b", "b", "a", "a", "b", "b", "a"]
    @test results["a"].total_time == 2.0
    @test results["a"].solved
    @test results["a"].representative_repeat == 3
    @test results["a"].final_value == 4.0
    @test !results["b"].solved && results["b"].status == "repeat_failure"
    @test benchmark_metadata(results["a"])["repeat_times"] == [1.0, 3.0, 2.0]
    @test length(results["b"].raw_repeats) == 3
    @test_throws ArgumentError benchmark_methods(["a" => () -> measured("a")]; warmups=0)
end

@testset "Finite performance ratios and honest success coverage" begin
    times = [1.0 4.0 Inf; 2.0 Inf 1.0; Inf Inf Inf]
    p = finite_performance_profile(times; max_ratio=2.0)
    @test p.thresholds == [1.0, 2.0]
    @test p.x == [0.0, 1.0]
    @test p.solved_counts == [2, 1, 1]
    @test p.success_rates ≈ [2/3, 1/3, 1/3]
    @test p.finite_ratio_coverage ≈ [2/3, 0, 1/3]
    @test p.all_failed_instances == 1
    @test all(isinf, p.ratios[3, :])
    @test all(==(0.0), finite_performance_profile(fill(Inf, 2, 2)).fractions)
    @test finite_performance_profile(ones(2, 2)).fractions == ones(1, 2)
    @test_throws ArgumentError finite_performance_profile([0.0 1.0])
    @test_throws ArgumentError finite_performance_profile([-1.0 1.0])
    @test_throws ArgumentError finite_performance_profile(zeros(0, 2))
    speed = paired_speed_ratios(times; reference=1)
    @test speed[2].paired_instances == 1
    @test speed[2].baseline_over_reference_harmonic == 4.0
    @test harmonic_mean([1.0, 3.0]) == 1.5
    @test isnan(harmonic_mean(Float64[]))
end

@testset "Manifest records actual runtime and source content" begin
    mktempdir() do directory
        path = joinpath(directory, "run.toml")
        manifest = write_run_manifest(path, Dict("seed" => 7, "method" => :adaptive,
                                                  "preprocessing_seconds" => 0.2))
        saved = TOML.parsefile(path)
        @test saved["julia_version"] == string(VERSION)
        @test saved["julia_threads"] == Threads.nthreads()
        @test saved["blas_threads"] == BLAS.get_num_threads()
        @test haskey(saved["source_sha256"], "src/ACGDSolvers.jl")
        @test saved["source_sha256"] == manifest["source_sha256"]
        @test saved["config"]["method"] == "adaptive"
        @test saved["config"]["preprocessing_seconds"] == 0.2
        @test haskey(saved, "package_versions")
        @test haskey(saved["source_sha256"], "docs/backtracking_candidates.toml")
        @test haskey(saved["input_file_sha256"], "movielens100k.csv")
        @test haskey(saved["input_file_sha256"], "video_colocalization/aeroplane_data_small.mat")
        @test haskey(saved, "plotting_environment")
    end
end

@testset "All benchmark scripts parse without executing their drivers" begin
    root = normpath(joinpath(@__DIR__, ".."))
    scripts = filter(path -> endswith(path, ".jl"), readdir(root; join=true))
    @test length(scripts) == 11
    for path in scripts
        @test load_problem_definitions(path) isa Module
    end
end

@testset "Small generated instances: wrappers, pairing, gradients and feasibility" begin
    root = normpath(joinpath(@__DIR__, ".."))
    configurations = [
        ("lasso_problem.jl", (; m=12, n=8, tau=3), true),
        ("Least_Squares.jl", (; m=12, n=6, k_star=2, cond_num=3.0), false),
        ("STQO.jl", (; n=7), true),
        ("matrix_balancing.jl", (; n=8, a=3, l=1.0, u=2.0), true),
        ("Power_Shifted_Quadratic.jl", (;), false),
    ]
    for (filename, kwargs, constrained) in configurations
        @testset "$filename" begin
            definitions = load_problem_definitions(joinpath(root, filename))
            args = filename == "Power_Shifted_Quadratic.jl" ? (71, 6) : (71,)
            original = Base.invokelatest(definitions.generate_problem, args...; kwargs...)
            problem = original
            if filename == "matrix_balancing.jl"
                # An independent analytic gradient lets this stdlib suite check
                # the generated objective/oracle without mocking Zygote. The
                # production AD dependency is checked separately when installed.
                analytic(x) = let B = problem.A .* exp.(x .- x')
                    vec(sum(B; dims=2)) - vec(sum(B; dims=1))
                end
                problem = merge(problem, (grad_f=analytic,))
            end
            initial = copy(problem.x0)
            direction = collect(1.0:length(initial))
            direction ./= norm(direction)
            h = 1e-6
            differential = (problem.f(initial + h * direction) -
                            problem.f(initial - h * direction)) / (2h)
            @test dot(problem.grad_f(initial), direction) ≈ differential rtol=1e-5 atol=1e-6
            L = if filename == "STQO.jl"
                Base.invokelatest(definitions.estimate_L, problem.Q)
            elseif filename == "Power_Shifted_Quadratic.jl"
                problem.L_estimate
            elseif filename == "matrix_balancing.jl"
                Base.invokelatest(definitions.estimate_L, problem.A; τ=1.0)
            else
                Base.invokelatest(definitions.estimate_L, problem.A)
            end
            method_names = [:conditional_gradient_adaptive, :conditional_gradient_adjustable_scaling,
                            :conditional_gradient_pure_backtracking, :conditional_gradient_auto_conditioned,
                            :conditional_gradient_short_step]
            constrained && push!(method_names, :conditional_gradient_open)
            !constrained && push!(method_names, :adabb)
            for name in method_names
                routine = getproperty(definitions, name)
                positional = if name == :adabb
                    (problem.f, problem.grad_f, problem.x0, 1e-3)
                else
                    base = constrained ? (problem.f, problem.grad_f, problem.lmo, problem.x0) :
                                         (problem.f, problem.grad_f, problem.lmo, problem.x0, problem.rho)
                    name in (:conditional_gradient_adaptive, :conditional_gradient_adjustable_scaling) ?
                        (base..., 0.25) : name == :conditional_gradient_short_step ? (base..., L) : base
                end
                r = Base.invokelatest(routine, positional...; max_iter_cap=20, fstar=problem.optimal_value)
                @test problem.x0 == initial
                @test isfinite(r.final_value) && all(isfinite, r.x)
                @test r.final_value ≈ problem.f(r.x)
                gradient = problem.grad_f(r.x)
                gap = constrained ? dot(gradient, r.x - problem.lmo(gradient)) : norm(gradient)
                @test r.final_gap ≈ gap atol=1e-9
                @test r.termination_iteration == r.k <= 20
                @test r.values[1] == problem.f(initial)
                if constrained
                    if filename == "lasso_problem.jl"
                        @test norm(r.x, 1) <= problem.tau + 1e-10
                    elseif filename == "STQO.jl"
                        @test minimum(r.x) >= -1e-12 && sum(r.x) ≈ 1
                    else
                        @test all(problem.l - 1e-12 .<= r.x .<= problem.u + 1e-12)
                    end
                end
                if name == :conditional_gradient_adjustable_scaling
                    @test r.gamma_iterations == collect(0:10:r.k)
                end
                if r.solved
                    if filename == "STQO.jl"
                        @test r.final_gap <= 1e-3
                    elseif !constrained
                        @test r.final_value - problem.optimal_value <= 1e-5
                    else
                        @test r.final_gap <= 1e-5 || r.final_value - problem.optimal_value <= 1e-5
                    end
                end
            end
            regenerated = Base.invokelatest(definitions.generate_problem, args...; kwargs...)
            @test regenerated.x0 == initial
            @test regenerated.f(initial) == original.f(initial)
        end
    end
end

@testset "Generated matrix factorization: product constraints and tuple wrappers" begin
    definitions = load_problem_definitions(joinpath(@__DIR__, "..", "matrix_factorization.jl"))
    p = Base.invokelatest(definitions.generate_problem, 19; m=8, n=6, r=2, sigma_max=2.0, sigma_min=0.5)
    initial_U, initial_V = copy(p.U0), copy(p.V0)
    @test p.f(p.U_star, p.V_star) <= 1e-25
    @test opnorm(p.U_star) <= p.tau1 && opnorm(p.V_star) <= p.tau2
    dU, dV = ones(size(p.U0)), ones(size(p.V0))
    gU, gV = p.grad_f(p.U0, p.V0)
    h = 1e-6
    differential = (p.f(p.U0 + h*dU, p.V0 + h*dV) -
                    p.f(p.U0 - h*dU, p.V0 - h*dV)) / (2h)
    @test dot(gU, dU) + dot(gV, dV) ≈ differential rtol=1e-5
    for name in (:adaptive, :adjustable_scaling, :pure_backtracking, :auto_conditioned, :short_step, :open)
        routine = getproperty(definitions, Symbol("conditional_gradient_", name, "_matfac"))
        args = (p.f, p.grad_f, p.lmo, p.U0, p.V0)
        args = name in (:adaptive, :adjustable_scaling) ? (args..., 0.25) :
               name == :short_step ? (args..., p.L_global) : args
        r = Base.invokelatest(routine, args...; max_iter_cap=20, fstar=p.optimal_value)
        @test p.U0 == initial_U && p.V0 == initial_V
        @test opnorm(r.U) <= p.tau1 + 1e-10 && opnorm(r.V) <= p.tau2 + 1e-10
        @test r.final_value ≈ p.f(r.U, r.V)
        gU, gV = p.grad_f(r.U, r.V)
        vU, vV = p.lmo(gU, gV)
        @test r.final_gap ≈ dot(gU, r.U - vU) + dot(gV, r.V - vV) atol=1e-9
        @test r.termination_iteration == r.k <= 20
        r.solved && @test r.final_gap <= 1e-5 || r.final_value - p.optimal_value <= 1e-5
    end
end

@testset "Wrapper default stopping rules" begin
    for filename in ("Least_Squares.jl", "Power_Shifted_Quadratic.jl")
        definitions = load_problem_definitions(joinpath(@__DIR__, "..", filename))
        routine = definitions.conditional_gradient_pure_backtracking
        r = Base.invokelatest(routine, x -> 5e-6, x -> ones(1), ball_oracle, [1.0], 1.0;
                             fstar=0.0, max_iter_cap=0)
        @test r.solved && r.stopping_criterion == "objective"
        r = Base.invokelatest(routine, x -> 1.0, x -> zeros(1), ball_oracle, [1.0], 1.0;
                             fstar=0.0, max_iter_cap=1)
        @test !r.solved && r.stopping_criterion == "none"
    end
    definitions = load_problem_definitions(joinpath(@__DIR__, "..", "STQO.jl"))
    r = Base.invokelatest(definitions.conditional_gradient_pure_backtracking,
                         x -> 5.0, x -> [8e-4], g -> zeros(1), [1.0];
                         fstar=0.0, max_iter_cap=0)
    @test r.solved && r.stopping_criterion == "stationarity"
    r = Base.invokelatest(definitions.conditional_gradient_pure_backtracking,
                         x -> 0.0, x -> ones(1), g -> zeros(1), [1.0];
                         fstar=0.0, max_iter_cap=0)
    @test !r.solved # An objective equal to supplied fstar is irrelevant for StQO.
end
