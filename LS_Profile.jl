# ==========================================================================
# Least-Squares Frank-Wolfe: all 5 methods x N seeds -> combined performance profile
# ==========================================================================
# "Solved" = EITHER of the method's own convergence criteria fires:
#     (a) gradient norm <= epsilon           (grad_norm <= epsilon)
#     (b) objective is within epsilon of the TRUE optimum
#         (current_f - fstar <= epsilon), where fstar = prob.optimal_value
#         (the unconstrained least-squares optimum, obtained via the
#         normal equations x_ls = (A'A)\(A'b), NOT 0 -- there is noise).
#
# Every solver's inner loop is `while true` (no fixed max_iter bound).
# A safety net (max_iter_cap / max_time) prevents a genuine infinite loop
# on a run that never satisfies epsilon -- hitting it does NOT count as
# solving (time = Inf, excluded from the profile at that problem).
#
# Since the problem is UNCONSTRAINED and the LMO returns a vector on the
# unit L2-ball (rho = 1), we have the exact identity
#     ||x^k - x^{k-1}|| = ||t_{k-1} v^{k-1}|| = t_{k-1} * rho = t_{k-1}
# so the "Auto-conditioned" realized-Lipschitz estimate uses
#     norms2 = t_k^2 * rho^2
# in closed form instead of recomputing norm(x_new - x_curr) numerically.
#
# The final plot is a true Dolan-Moré performance profile (ratio-based
# x-axis), built with BenchmarkProfiles.jl. See:
# https://tmigot.github.io/posts/2024/06/teaching/
# ==========================================================================

using LinearAlgebra
using Statistics
using Random
using Plots
using Measures
using PyPlot
using MAT
using SparseArrays
using BenchmarkProfiles

pyplot()
PyPlot.PyCall.pyimport("warnings").filterwarnings("ignore")

output_dir = "/content/Least_Squares_performance_output"
mkpath(output_dir)

# --------------------------------------------------------------------
# 1. Problem generator — one seed -> one independent LS instance
# --------------------------------------------------------------------
# NOTE: original sizes were m=20000, n=1000 which is far too heavy to
# repeat over many seeds x 5 solvers. Defaults below are lighter but the
# structure (ill-conditioning, sparse ground truth, noise) is identical.
# Bump m/n back up if you have the compute budget.
function generate_problem(seed; m=2000, n=200, cond_num=50.0, k_star=10, noise_level=1e-2)
    Random.seed!(seed)

    U, _ = qr(randn(m, n))
    V, _ = qr(randn(n, n))
    Σ = diagm(0 => [cond_num^(-(i - 1) / (n - 1)) for i in 1:n])
    A = U * Σ * V'

    x_star = zeros(n)
    indices = randperm(n)[1:k_star]
    x_star[indices] = randn(k_star)

    b = A * x_star + noise_level * randn(m)

    f(x) = sum((A * x - b) .^ 2)
    grad_f(x) = 2 * A' * (A * x - b)

    rho = 1.0
    function lmo(g)
        ng = norm(g)
        return ng > 0 ? -rho * g / ng : zeros(length(g))
    end

    x0 = zeros(n)

    # True (unconstrained) least-squares optimum via normal equations —
    # this is fstar, NOT 0, because of the additive noise.
    # x_ls = (A' * A) \ (A' * b)
    x_ls = A \ b
    fstar = sum((A * x_ls - b) .^ 2)

    return (f=f, grad_f=grad_f, lmo=lmo, x0=x0, rho=rho, seed=seed,
            optimal_value=fstar)
end

# --------------------------------------------------------------------
# 2. Solvers — all `while true`, all return (..., solved, total_time)
#    All accept `fstar` and stop when
#    grad_norm <= epsilon  OR  current_f - fstar <= epsilon.
# --------------------------------------------------------------------
function conditional_gradient_adaptive(f, grad_f, lmo, x0, rho, gamma;
        epsilon=1e-5, delta=1e-10, beta=2, max_iter_cap=200000, max_time=500.0,
        fstar=0.0)
    x_prev = copy(x0); x_curr = copy(x0)
    values = [f(x_curr)]; times = [0.0]; gaps = Float64[]; L_ks = Float64[]
    k = 0; solved = false; prev_t_k = 0.0
    prev_grad = grad_f(x_prev); current_f = f(x_curr)
    t_start_total = time()

    while true
        start = time()
        current_grad = grad_f(x_curr)
        v = lmo(current_grad)
        grad_norm = norm(current_grad)

        if grad_norm <= epsilon || (current_f - fstar) <= epsilon
            push!(times, time() - start); push!(gaps, grad_norm)
            solved = true
            break
        end
        if k >= max_iter_cap # || (time() - t_start_total) > max_time
            solved = false
            break
        end

        if k == 0
            Random.seed!(23)
            d0 = randn(length(x0))
            L_k = gamma * ((norm(grad_f(x0) - grad_f(x0 + 1e-3 * d0)) / (1e-3 * norm(d0))) + delta)
        else
            grad_diff = norm(current_grad - prev_grad)
            L_k = gamma * (grad_diff / (rho * prev_t_k) + delta)
        end

        t_k = grad_norm / (rho * L_k)
        while true
            x_new = x_curr + t_k * v
            new_f = f(x_new)
            if current_f - new_f >= rho * t_k * grad_norm - (L_k / 2) * t_k^2 * rho^2
                x_prev = copy(x_curr); x_curr = x_new
                break
            else
                L_k *= beta
                t_k = grad_norm / (rho * L_k)
            end
        end

        k += 1
        push!(times, time() - start)
        current_f = f(x_curr); prev_grad = current_grad; prev_t_k = t_k
        push!(gaps, grad_norm); push!(values, current_f); push!(L_ks, L_k)
    end
    return (x=x_curr, values=values, times=times, gaps=gaps, L_ks=L_ks,
            k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_adjustable_scaling(f, grad_f, lmo, x0, rho, gamma;
        epsilon=1e-5, delta=1e-10, beta=2, max_iter_cap=200000, max_time=500.0,
        fstar=0.0)
    x_prev = copy(x0); x_curr = copy(x0)
    values = [f(x_curr)]; times = [0.0]; gaps = Float64[]; L_ks = Float64[]
    gamma_history = [gamma]
    k = 0; solved = false; prev_t_k = 0.0
    prev_grad = grad_f(x_prev); current_f = f(x_curr)
    recent_backtracks = Int[]
    t_start_total = time()

    while true
        start = time()
        current_grad = grad_f(x_curr)
        v = lmo(current_grad)
        grad_norm = norm(current_grad)

        if grad_norm <= epsilon || (current_f - fstar) <= epsilon
            push!(times, time() - start); push!(gaps, grad_norm)
            solved = true
            break
        end
        if k >= max_iter_cap # || (time() - t_start_total) > max_time
            solved = false
            break
        end

        if k == 0
            Random.seed!(23)
            d0 = randn(length(x0))
            L_k = gamma * ((norm(grad_f(x0) - grad_f(x0 + 1e-3 * d0)) / (1e-3 * norm(d0))) + delta)
        else
            grad_diff = norm(current_grad - prev_grad)
            L_k = gamma * (grad_diff / (rho * prev_t_k) + delta)
        end

        t_k = grad_norm / (rho * L_k)
        i = 0
        while true
            x_new = x_curr + t_k * v
            new_f = f(x_new)
            if current_f - new_f >= rho * t_k * grad_norm - (L_k / 2) * t_k^2 * rho^2
                x_prev = copy(x_curr); x_curr = x_new
                push!(recent_backtracks, i)
                break
            else
                L_k *= beta
                t_k = grad_norm / (rho * L_k)
                i += 1
            end
        end

        if k % 10 == 0 && k > 0
            total_backtracks = sum(recent_backtracks)
            if total_backtracks == 0
                gamma =  gamma * 0.9  # Decrease gamma, with a lower bound
             #   println(">> k $k: No backtracking, >> reducing gamma to $(round(gamma, digits=4))")
            elseif total_backtracks > 10
                gamma =min(1,gamma * 1.1)  # Increase gamma, with an upper bound
            #    println("<< k $k: ($total_backtracks) backtracking, << increasing gamma to $(round(gamma, digits=4))")
            end
            #push!(gamma_history, gamma)
            recent_backtracks = Int[]  # Reset for the next 10 iterations
        elseif k % 10 == 0
            #push!(gamma_history, gamma)
            recent_backtracks = Int[]
        else
            push!(recent_backtracks, i)
        end
        k += 1
        push!(times, time() - start)
        current_f = f(x_curr); prev_grad = current_grad; prev_t_k = t_k
        push!(gaps, grad_norm); push!(values, current_f); push!(L_ks, L_k)
    end
    return (x=x_curr, values=values, times=times, gaps=gaps, L_ks=L_ks,
            gamma_history=gamma_history, k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_pedregosa(f, grad_f, lmo, x0, rho;
        epsilon=1e-5, max_iter_cap=200000, max_time=500.0, fstar=0.0)
    x_prev = copy(x0)
    values = [f(x_prev)]; times = [0.0]; gaps = Float64[]; L_ks = Float64[]
    k = 0; solved = false; tau_bt = 2.0; eta = 0.9
    t_start_total = time()

    Random.seed!(23)
    d0 = randn(length(x0))
    L_minus1 = norm(grad_f(x0) - grad_f(x0 + 1e-3 * d0)) / (1e-3 * norm(d0))
    M = L_minus1 * eta

    while true
        start = time()
        grad = grad_f(x_prev)
        v = lmo(grad)
        grad_norm = norm(grad)
        f_prev = f(x_prev)

        if grad_norm <= epsilon || (f_prev - fstar) <= epsilon
            push!(times, time() - start); push!(gaps, grad_norm)
            solved = true
            break
        end
        if k >= max_iter_cap # || (time() - t_start_total) > max_time
            solved = false
            break
        end

        t_k = grad_norm / (M * rho)
        x_new = x_prev
        while true
            x_new = x_prev + t_k * v
            f_new = f(x_new)
            if f_prev - f_new >= rho * t_k * grad_norm - (M / 2) * t_k^2 * rho^2
                break
            else
                M *= tau_bt
                t_k = grad_norm / (M * rho)
            end
        end

        x_prev = x_new
        k += 1
        push!(times, time() - start)
        push!(gaps, grad_norm); push!(values, f(x_new)); push!(L_ks, M)
        M = M * eta
    end
    return (x=x_prev, values=values, times=times, gaps=gaps, L_ks=L_ks,
            k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_auto_conditioned(f, grad_f, lmo, x0, rho;
        epsilon=1e-5, delta=1e-10, max_iter_cap=200000, max_time=500.0,
        fstar=0.0)
    x_curr = copy(x0)
    values = [f(x_curr)]; times = [0.0]; gaps = Float64[]; L_ks = Float64[]
    L_history = Float64[]
    k = 0; solved = false
    t_start_total = time()

    Random.seed!(23)
    d0 = randn(length(x0))
    L0 = (norm(grad_f(x0) - grad_f(x0 + 1e-3 * d0)) / (1e-3 * norm(d0))) + delta
    push!(L_history, L0)

    current_f = f(x_curr)
    while true
        start = time()
        current_grad = grad_f(x_curr)
        v = lmo(current_grad)
        grad_norm = norm(current_grad)

        if grad_norm <= epsilon || (current_f - fstar) <= epsilon
            push!(times, time() - start); push!(gaps, grad_norm)
            solved = true
            break
        end
        if k >= max_iter_cap # || (time() - t_start_total) > max_time
            solved = false
            break
        end

        hat_L_k = maximum(L_history)
        t_k = grad_norm / (rho * hat_L_k)
        x_new = x_curr + t_k * v
        new_f = f(x_new)

        # Unconstrained + unit-ball LMO (rho=1) => exact closed form,
        # no need to recompute norm(x_new - x_curr) numerically:
        #   ||x_new - x_curr|| = ||t_k * v|| = t_k * rho
        norms2 = t_k^2 * rho^2
        if norms2 > 0
            L_k_realized = 2 * (new_f - current_f - dot(current_grad, t_k * v)) / norms2
            push!(L_history, max(L_k_realized, delta))
        else
            push!(L_history, hat_L_k)
        end

        x_curr = x_new
        k += 1
        push!(times, time() - start)
        current_f = new_f
        push!(gaps, grad_norm); push!(values, current_f); push!(L_ks, hat_L_k)
    end
    return (x=x_curr, values=values, times=times, gaps=gaps, L_ks=L_ks,
            k=k, solved=solved, total_time=sum(times))
end

function adabb(f, grad_f, x0, alpha0;
        epsilon=1e-5, max_iter_cap=200000, max_time=500.0, fstar=0.0)
    x_prev = copy(x0)
    grad_prev = grad_f(x_prev)
    gap = norm(grad_prev)
    x = x_prev - alpha0 * grad_prev

    grad_curr = grad_f(x)
    s_1 = x - x_prev; y_1 = grad_curr - grad_prev
    lambda_1 = dot(y_1, s_1) / dot(y_1, y_1)
    theta0 = lambda_1 >= sqrt(2) * alpha0 ? lambda_1^2 / (2 * alpha0^2) - 1.0 : 0.0

    alpha_km1 = alpha0; theta_km1 = theta0
    values = [f(x_prev), f(x)]; times = [0.0]; gaps = Float64[]
    k = 1; solved = false
    t_start_total = time()

    while true
        start = time()
        if k > 1
            grad_curr = grad_f(x)
        end
        f_curr = f(x)
        gap = norm(grad_curr)
        if gap <= epsilon || (f_curr - fstar) <= epsilon
            push!(times, time() - start); push!(gaps, gap)
            solved = true
            break
        end
        if k >= max_iter_cap # || (time() - t_start_total) > max_time
            solved = false
            break
        end

        s_k = x - x_prev; y_k = grad_curr - grad_prev
        lambda_k = dot(y_k, s_k) / dot(y_k, y_k)

        if lambda_k >= alpha_km1
            alpha_k = sqrt(1 + theta_km1) * alpha_km1; theta_k = alpha_k / alpha_km1
        elseif alpha_km1 / 2 < lambda_k < alpha_km1
            alpha_k = lambda_k; theta_k = 2 * alpha_k / alpha_km1 - alpha_k / lambda_k
        else
            alpha_k = lambda_k / sqrt(2); theta_k = alpha_k / alpha_km1
        end

        x_prev = copy(x); grad_prev = copy(grad_curr)
        x = x - alpha_k * grad_curr
        k += 1
        alpha_km1 = alpha_k; theta_km1 = theta_k

        push!(times, time() - start)
        f_new = f(x); gap = norm(grad_prev)
        push!(gaps, gap); push!(values, f_new)
    end
    return (x=x, values=values, times=times, gaps=gaps,
            k=k, solved=solved, total_time=sum(times))
end

# --------------------------------------------------------------------
# 3. Run all 5 methods on each of the N seeded problems
# --------------------------------------------------------------------
seeds  = 1:50
gamma0 = 1/4
alpha0 = 1e-10

method_order = ["Adaptive constant", "Adaptive adjustable",
                "Pure backtracking", "Auto-conditioned", "AdaBB"]

all_times  = Dict(name => fill(Inf, length(seeds)) for name in method_order)
all_solved = Dict(name => falses(length(seeds)) for name in method_order)

for (idx, s) in enumerate(seeds)
    println("\n================ seed = $s ================")
    prob  = generate_problem(s)
    local fstar = prob.optimal_value   # true LS optimum (with noise), NOT 0

    runs = Dict(
         "Auto-conditioned"    => conditional_gradient_auto_conditioned(prob.f, prob.grad_f, prob.lmo, prob.x0, prob.rho; fstar=fstar),
        "Adaptive constant"   => conditional_gradient_adaptive(prob.f, prob.grad_f, prob.lmo, prob.x0, prob.rho, gamma0; fstar=fstar),
        "Adaptive adjustable" => conditional_gradient_adjustable_scaling(prob.f, prob.grad_f, prob.lmo, prob.x0, prob.rho, gamma0; fstar=fstar),
        "Pure backtracking"   => conditional_gradient_pedregosa(prob.f, prob.grad_f, prob.lmo, prob.x0, prob.rho; fstar=fstar),
        "AdaBB"               => adabb(prob.f, prob.grad_f, prob.x0, alpha0; fstar=fstar),
    )

    for name in method_order
        r = runs[name]
        all_solved[name][idx] = r.solved
        all_times[name][idx]  = r.solved ? r.total_time : Inf
        status = r.solved ? "solved" : "NOT solved"
        println(rpad(name, 22), status, "  iters=", r.k, "  time=", round(r.total_time, digits=4), "s")
    end
end

# --------------------------------------------------------------------
# 4. Summary table + save raw results
# --------------------------------------------------------------------
println("\n# Summary ($(length(seeds))-seed sweep) #")
println(rpad("Method", 22), rpad("Solved", 10), rpad("Mean time (s)", 16), "Median time (s)")
for name in method_order
    t = filter(isfinite, all_times[name])
    n_solved = length(t)
    mean_t = n_solved > 0 ? round(mean(t), digits=3) : NaN
    med_t  = n_solved > 0 ? round(median(t), digits=3) : NaN
    println(rpad(name, 22), rpad("$n_solved/$(length(seeds))", 10), rpad(string(mean_t), 16), string(med_t))
end

matvarname(s) = replace(s, r"[^A-Za-z0-9_]" => "_")

matwrite(joinpath(output_dir, "benchmark_all_methods_LS.mat"), Dict(
    "seeds"  => collect(seeds),
    "times"  => Dict(matvarname(name) => all_times[name]  for name in method_order),
    "solved" => Dict(matvarname(name) => all_solved[name] for name in method_order),
))

# --------------------------------------------------------------------
# 5. Dolan-Moré performance profile (ratio-based x-axis)
# See: https://tmigot.github.io/posts/2024/06/teaching/
# --------------------------------------------------------------------
T = hcat([all_times[name] for name in method_order]...)   # n_problems x n_solvers

style = (
    titlefont = (20, "serif"), guidefont = (20, "serif"),
    tickfont  = (20, "serif"), legendfont = (16, "serif"),
    grid = false, framestyle = :box, margin = 5mm,
    size = (900, 700), linewidth = 3,
)

p = performance_profile(PlotsBackend(), T, method_order;
        logscale = true,
        legend = :bottomright,
        xlabel = "Performance Ratio",
        ylabel = "Fraction of Problems Solved",
        xformatter = :plain,
        style...)

plot!(p,
    xtickfont = Plots.font(20, "serif"),
    ytickfont = Plots.font(20, "serif"),
    guidefont = Plots.font(20, "serif"),
    ylims  = (-0.03, 1.03),
    xrotation = 20,
)

custom_colors = [:blue, :green, :red, :black, :olive]
for (i, series) in enumerate(p.series_list)
    series[:linecolor] = custom_colors[mod1(i, length(custom_colors))]
end

Plots.savefig(p, joinpath(output_dir, "LeastSquares_performance_profile_all_methods.eps"))
println("✅ Saved: LeastSquares_performance_profile_all_methods.eps in ", output_dir)
display(p)