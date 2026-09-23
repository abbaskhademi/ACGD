# ==========================================================================
# Matrix Factorization Frank-Wolfe: all 6 methods x N seeds -> combined
# performance profile
# ==========================================================================
# "Solved" = EITHER of the method's own convergence criteria fires:
#     (a) FW gap <= epsilon
#     (b) objective within epsilon of the TRUE optimum
#         (current_f - fstar <= epsilon), where fstar = 0 because
#         Y = U_star * V_star' exactly and tau1 = ceil(opnorm(U_star)),
#         tau2 = ceil(opnorm(V_star)) make (U_star, V_star) FEASIBLE for
#         the spectral-norm balls, so f(U_star, V_star) = 0 is attained.
# This is NOT an artificial max_iter cutoff.
#
# NOTE (nonconvexity): f(U,V) = 0.5||U V' - Y||^2 is bilinear, hence
# nonconvex. The FW gap is therefore only a STATIONARITY measure and does
# NOT bound f - fstar from above (unlike the convex Lasso case, where
# gap >= f - fstar). Criteria (a) and (b) are genuinely unordered here: a
# run can stop at a stationary/boundary point with a tiny gap while f is
# still far from 0. `final_gap` and `final_value` are both logged below so
# these two exit modes can be told apart per seed.
#
# Every solver's inner loop is `while true` (no hard max_iter bound). A
# safety net (max_iter_cap / max_time) prevents a genuine infinite loop on
# a run that never satisfies epsilon -- hitting it does NOT count as
# solving (time = Inf, excluded from the profile at that point).
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
using BenchmarkProfiles

pyplot()
PyPlot.PyCall.pyimport("warnings").filterwarnings("ignore")

problem_name = "matrix_factorization"
output_dir   = joinpath(@__DIR__, problem_name)
mkpath(output_dir)


# MAT.jl struct field names must be valid identifiers -- sanitize just for
# the .mat export. method_order, console output, and the profile plot
# below still use the original names. (Defined early: now also needed by
# the per-iteration .mat writer in section 4b.)
matvarname(s) = replace(s, r"[^A-Za-z0-9_]" => "_")

# --------------------------------------------------------------------
# 1. Shared helper functions (do not depend on seed)
# --------------------------------------------------------------------
fro_inner(A, B) = dot(A, B)
fro_norm2_concat(A, B) = norm(A)^2 + norm(B)^2
fro_norm_concat(A, B) = sqrt(fro_norm2_concat(A, B))

function lmo_spectral(G, tau)
    F = svd(G)
    return -tau * F.U * F.Vt
end

function project_spectral(X, tau)
    F = svd(X)
    return F.U * Diagonal(min.(F.S, tau)) * F.Vt
end

# --------------------------------------------------------------------
# 2. Problem generator -- one seed -> one independent Matrix
#    Factorization instance
# --------------------------------------------------------------------
function generate_problem(seed; m=400, n=300, r=20, sigma_max=10.0, sigma_min=0.01)
    Random.seed!(seed)

    sigmas = exp.(range(log(sigma_max), log(sigma_min), length=r)) |> collect
    Sigma_half = Diagonal(sqrt.(sigmas))

    U_bar, _ = qr(randn(m, r)); U_bar = Matrix(U_bar)[:, 1:r]
    V_bar, _ = qr(randn(n, r)); V_bar = Matrix(V_bar)[:, 1:r]

    U_star = U_bar * Sigma_half
    V_star = V_bar * Sigma_half
    Y = U_star * V_star'

    # ceil() => tau_i >= opnorm of the planted factor, so (U_star, V_star)
    # is feasible and the global optimum is exactly 0 (see header note).
    tau1 = ceil(Int, opnorm(U_star))
    tau2 = ceil(Int, opnorm(V_star))

    f(U, V) = 0.5 * norm(U * V' - Y)^2
    function grad_f(U, V)
        R = U * V' - Y
        return R * V, R' * U
    end
    lmo(grad_U, grad_V) = (lmo_spectral(grad_U, tau1), lmo_spectral(grad_V, tau2))

    # Initial point: at constraint boundary, far from optimum.
    # Varies with `seed` (see note above the script about the double
    # Random.seed!(23) in the original single-run version).
    U0 = project_spectral(10.0 * randn(m, r), tau1)
    V0 = project_spectral(10.0 * randn(n, r), tau2)

    L_global = max(tau1^2, tau2^2) + opnorm(Y) + tau1 * tau2

    return (f=f, grad_f=grad_f, lmo=lmo, U0=U0, V0=V0,
            tau1=tau1, tau2=tau2, Y=Y, L_global=L_global,
            U_star=U_star, V_star=V_star, optimal_value=0.0, seed=seed)
end

# --------------------------------------------------------------------
# 3. Solvers -- all `while true`, all return (..., solved, total_time)
#    All accept `fstar` (true optimal objective value) and stop when
#    gap <= epsilon  OR  current_f - fstar <= epsilon.
# --------------------------------------------------------------------
function conditional_gradient_adaptive_matfac(f, grad_f, lmo, U0, V0, gamma;
        epsilon=1e-5, delta=1e-10, beta=2, max_iter_cap=100000, max_time=500.0,
        fstar=0.0)
    U_prev = copy(U0); V_prev = copy(V0)
    U_curr = copy(U0); V_curr = copy(V0)
    current_f = f(U_curr, V_curr)
    values = [current_f]; times = [0.0]; gaps = Float64[]; L_ks = Float64[]
    k = 0; solved = false
    prev_grad_U, prev_grad_V = grad_f(U_prev, V_prev)
    t_start_total = time()

    while true
        start = time()
        curr_grad_U, curr_grad_V = grad_f(U_curr, V_curr)
        P, Q = lmo(curr_grad_U, curr_grad_V)
        dU = P - U_curr; dV = Q - V_curr
        normd2 = fro_norm2_concat(dU, dV)
        gap = -(fro_inner(curr_grad_U, dU) + fro_inner(curr_grad_V, dV))

        

        if k == 0
            Random.seed!(23)
            dU0 = randn(size(U0)); dV0 = randn(size(V0))
            norm_d0 = fro_norm_concat(dU0, dV0)
            eps_fd = 1e-3
            g1_U, g1_V = grad_f(U0, V0)
            g2_U, g2_V = grad_f(U0 + eps_fd * dU0, V0 + eps_fd * dV0)
            grad_diff_norm = fro_norm_concat(g2_U - g1_U, g2_V - g1_V)
            L_k = gamma * (grad_diff_norm / (eps_fd * norm_d0) + delta)
        else
            grad_diff_norm = fro_norm_concat(curr_grad_U - prev_grad_U, curr_grad_V - prev_grad_V)
            x_diff_norm = fro_norm_concat(U_curr - U_prev, V_curr - V_prev)
            L_k = gamma * (grad_diff_norm / x_diff_norm + delta)
        end

        Lknormd2 = L_k * normd2
        t_k = min(gap / Lknormd2, 1.0)
        while true
            U_new = U_curr + t_k * dU
            V_new = V_curr + t_k * dV
            new_f = f(U_new, V_new)
            if current_f - new_f >= t_k * gap - (Lknormd2 / 2) * t_k^2
                U_prev = copy(U_curr); V_prev = copy(V_curr)
                U_curr = U_new; V_curr = V_new
                break
            else
                L_k *= beta
                Lknormd2 = L_k * normd2
                t_k = min(gap / Lknormd2, 1.0)
            end
        end

        k += 1
        push!(times, time() - start);
        current_f = f(U_curr, V_curr)
        prev_grad_U = curr_grad_U; prev_grad_V = curr_grad_V
        push!(gaps, gap); push!(values, current_f); push!(L_ks, L_k)
        if gap <= epsilon || (current_f - fstar) <= epsilon
            solved = true
            break
        end
        if k >= max_iter_cap # || (time() - t_start_total) > max_time
            solved = false
            break
        end
    end
    return (U=U_curr, V=V_curr, values=values, times=times, gaps=gaps, L_ks=L_ks,
            k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_adjustable_scaling_matfac(f, grad_f, lmo, U0, V0, gamma;
        epsilon=1e-5, delta=1e-10, beta=2, max_iter_cap=100000, max_time=500.0,
        fstar=0.0)
    U_prev = copy(U0); V_prev = copy(V0)
    U_curr = copy(U0); V_curr = copy(V0)
    current_f = f(U_curr, V_curr)
    values = [current_f]; times = [0.0]; gaps = Float64[]; L_ks = Float64[]
    gamma_history = [gamma]
    k = 0; solved = false
    prev_grad_U, prev_grad_V = grad_f(U_prev, V_prev)
    recent_backtracks = Int[]
    t_start_total = time()

    while true
        start = time()
        curr_grad_U, curr_grad_V = grad_f(U_curr, V_curr)
        P, Q = lmo(curr_grad_U, curr_grad_V)
        dU = P - U_curr; dV = Q - V_curr
        normd2 = fro_norm2_concat(dU, dV)
        gap = -(fro_inner(curr_grad_U, dU) + fro_inner(curr_grad_V, dV))

        if k == 0
            Random.seed!(23)
            dU0 = randn(size(U0)); dV0 = randn(size(V0))
            norm_d0 = fro_norm_concat(dU0, dV0)
            eps_fd = 1e-3
            g1_U, g1_V = grad_f(U0, V0)
            g2_U, g2_V = grad_f(U0 + eps_fd * dU0, V0 + eps_fd * dV0)
            grad_diff_norm = fro_norm_concat(g2_U - g1_U, g2_V - g1_V)
            L_k = gamma * (grad_diff_norm / (eps_fd * norm_d0) + delta)
        else
            grad_diff_norm = fro_norm_concat(curr_grad_U - prev_grad_U, curr_grad_V - prev_grad_V)
            x_diff_norm = fro_norm_concat(U_curr - U_prev, V_curr - V_prev)
            L_k = gamma * (grad_diff_norm / x_diff_norm + delta)
        end

        Lknormd2 = L_k * normd2
        t_k = min(gap / Lknormd2, 1.0)
        i = 0
        while true
            U_new = U_curr + t_k * dU
            V_new = V_curr + t_k * dV
            new_f = f(U_new, V_new)
            if current_f - new_f >= t_k * gap - (Lknormd2 / 2) * t_k^2
                U_prev = copy(U_curr); V_prev = copy(V_curr)
                U_curr = U_new; V_curr = V_new
                push!(recent_backtracks, i)
                break
            else
                L_k *= beta
                Lknormd2 = L_k * normd2
                t_k = min(gap / Lknormd2, 1.0)
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
        push!(times, time() - start);
        current_f = f(U_curr, V_curr)
        prev_grad_U = curr_grad_U; prev_grad_V = curr_grad_V
        push!(gaps, gap); push!(values, current_f); push!(L_ks, L_k)
        if gap <= epsilon || (current_f - fstar) <= epsilon
            solved = true
            break
        end
        if k >= max_iter_cap # || (time() - t_start_total) > max_time
            solved = false
            break
        end
    end
    return (U=U_curr, V=V_curr, values=values, times=times, gaps=gaps, L_ks=L_ks,
            gamma_history=gamma_history, k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_auto_conditioned_matfac(f, grad_f, lmo, U0, V0;
        epsilon=1e-5, delta=1e-10, max_iter_cap=100000, max_time=500.0,
        fstar=0.0)
    U_curr = copy(U0); V_curr = copy(V0)
    current_f = f(U_curr, V_curr)
    values = [current_f]; times = [0.0]; gaps = Float64[]; L_ks = Float64[]
    L_history = Float64[]
    k = 0; solved = false
    t_start_total = time()

    Random.seed!(23)
    dU0 = randn(size(U0)); dV0 = randn(size(V0))
    norm_d0 = fro_norm_concat(dU0, dV0)
    eps_fd = 1e-3
    g1_U, g1_V = grad_f(U0, V0)
    g2_U, g2_V = grad_f(U0 + eps_fd * dU0, V0 + eps_fd * dV0)
    grad_diff_norm = fro_norm_concat(g2_U - g1_U, g2_V - g1_V)
    L0 = grad_diff_norm / (eps_fd * norm_d0) + delta
    push!(L_history, L0)

    while true
        start = time()
        curr_grad_U, curr_grad_V = grad_f(U_curr, V_curr)
        P, Q = lmo(curr_grad_U, curr_grad_V)
        dU = P - U_curr; dV = Q - V_curr
        normd2 = fro_norm2_concat(dU, dV)
        gap = -(fro_inner(curr_grad_U, dU) + fro_inner(curr_grad_V, dV))

        # Running MAXIMUM over all realized curvature so far -- a single
        hat_L_k = maximum(L_history)
        Lknormd2 = hat_L_k * normd2
        t_k = min(gap / Lknormd2, 1.0)
        U_new = U_curr + t_k * dU
        V_new = V_curr + t_k * dV
        new_f = f(U_new, V_new)

        sU = U_new - U_curr; sV = V_new - V_curr
        norms2 = fro_norm2_concat(sU, sV)
        if norms2 > 0
            L_k_realized = 2 * (new_f - current_f -
                (fro_inner(curr_grad_U, sU) + fro_inner(curr_grad_V, sV))) / norms2
            push!(L_history, L_k_realized)
        else
            push!(L_history, hat_L_k)
        end

        U_curr = U_new; V_curr = V_new
        k += 1
        push!(times, time() - start);
        current_f = new_f
        push!(gaps, gap); push!(values, current_f);  push!(L_ks, hat_L_k)
         if gap <= epsilon || (current_f - fstar) <= epsilon
            solved = true
            break
        end
        if k >= max_iter_cap # || (time() - t_start_total) > max_time
            solved = false
            break
        end
    end
    return (U=U_curr, V=V_curr, values=values, times=times, gaps=gaps, L_ks=L_ks,
            k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_pure_backtracking_matfac(f, grad_f, lmo, U0, V0;
        epsilon=1e-5, max_iter_cap=100000, max_time=500.0, fstar=0.0)
    U_prev = copy(U0); V_prev = copy(V0)
    current_f = f(U_prev, V_prev)
    values = [current_f]; times = [0.0]; gaps = Float64[]; L_ks = Float64[]
    k = 0; solved = false
    tau_bt = 2.0; eta = 0.9
    t_start_total = time()

    Random.seed!(23)
    dU0 = randn(size(U0)); dV0 = randn(size(V0))
    norm_d0 = fro_norm_concat(dU0, dV0)
    eps_fd = 1e-3
    g1_U, g1_V = grad_f(U0, V0)
    g2_U, g2_V = grad_f(U0 + eps_fd * dU0, V0 + eps_fd * dV0)
    grad_diff_norm = fro_norm_concat(g2_U - g1_U, g2_V - g1_V)
    L_minus1 = grad_diff_norm / (eps_fd * norm_d0)
    M = L_minus1 * eta

    while true
        start = time()
        grad_U, grad_V = grad_f(U_prev, V_prev)
        P, Q = lmo(grad_U, grad_V)
        dU = P - U_prev; dV = Q - V_prev
        normd2 = fro_norm2_concat(dU, dV)
        gap = -(fro_inner(grad_U, dU) + fro_inner(grad_V, dV))
        f_prev = current_f


        t_k = min(gap / (M * normd2), 1.0)
        U_new = U_prev; V_new = V_prev
        while true
            U_new = U_prev + t_k * dU
            V_new = V_prev + t_k * dV
            f_new = f(U_new, V_new)
            if f_prev - f_new >= t_k * gap - (M / 2) * t_k^2 * normd2
                break
            else
                M *= tau_bt
                t_k = min(gap / (M * normd2), 1.0)
            end
        end

        U_prev = U_new; V_prev = V_new
        k += 1
        M = M * eta
        push!(times, time() - start);
        current_f = f(U_prev, V_prev)
        push!(gaps, gap); push!(values, current_f); push!(L_ks, M)
        
        if gap <= epsilon || (f_prev - fstar) <= epsilon
            solved = true
            break
        end
        if k >= max_iter_cap # || (time() - t_start_total) > max_time
            solved = false
            break
        end
    end
    return (U=U_prev, V=V_prev, values=values, times=times, gaps=gaps, L_ks=L_ks,
            k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_open_matfac(f, grad_f, lmo, U0, V0;
        epsilon=1e-5, max_iter_cap=100000, max_time=500.0, fstar=0.0)
    U = copy(U0); V = copy(V0)
    values = [f(U, V)]; times = [0.0]; gaps = Float64[]
    k = 0; solved = false
    t_start_total = time()

    while true
        start = time()
        grad_U, grad_V = grad_f(U, V)
        P, Q = lmo(grad_U, grad_V)
        dU = P - U; dV = Q - V
        t_k = 1 / sqrt(1 + k)
        U = U + t_k * dU; V = V + t_k * dV
        k += 1
        push!(times, time() - start)
        gap = -(fro_inner(grad_U, dU) + fro_inner(grad_V, dV))
        f_prev = f(U, V)
        push!(gaps, gap); push!(values, f_prev);
        if gap <= epsilon || (f_prev - fstar) <= epsilon
            solved = true
            break
        end
        if k >= max_iter_cap # || (time() - t_start_total) > max_time
            solved = false
            break
        end
    end
    return (U=U, V=V, values=values, times=times, gaps=gaps, k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_short_step_matfac(f, grad_f, lmo, U0, V0, L;
        epsilon=1e-5, max_iter_cap=100000, max_time=500.0, fstar=0.0)
    U = copy(U0); V = copy(V0)
    values = [f(U, V)]; times = [0.0]; gaps = Float64[]
    k = 0; solved = false
    t_start_total = time()

    while true
        start = time()
        grad_U, grad_V = grad_f(U, V)
        P, Q = lmo(grad_U, grad_V)
        dU = P - U; dV = Q - V
        normd2 = fro_norm2_concat(dU, dV)
        gap = -(fro_inner(grad_U, dU) + fro_inner(grad_V, dV))
        t_k = min(gap / (L * normd2), 1.0)
        U = U + t_k * dU; V = V + t_k * dV
        k += 1
        push!(times, time() - start)
        f_prev = f(U, V)
        push!(gaps, gap); push!(values, f_prev);
        if gap <= epsilon || (f_prev - fstar) <= epsilon
            solved = true
            break
        end
        if k >= max_iter_cap # || (time() - t_start_total) > max_time
            solved = false
            break
        end
    end
    return (U=U, V=V, values=values, times=times, gaps=gaps, k=k, solved=solved, total_time=sum(times))
end

# --------------------------------------------------------------------
# 4. Run all 6 methods on each of the N seeded problems
# --------------------------------------------------------------------
println("\n================ Matrix Factorization Problem ================")
seeds  = 1:100   # consider trying seeds = 1:5 first -- SVDs each iteration
                # (400x20, 300x20) make this considerably slower than Lasso
gamma0 = 1/4

method_order = ["Adaptive constant", "Adaptive adjustable", "Pure backtracking", "Auto-conditioned",
                 "Short-step", "Open-loop"]

all_times  = Dict(name => fill(Inf, length(seeds)) for name in method_order)
all_solved = Dict(name => falses(length(seeds)) for name in method_order)

# per-method, per-seed iteration histories (values / gaps / times / L_k)
per_iter     = Dict(name => Dict{String,Any}() for name in method_order)
report_lines = String[]                      # text report accumulator

for (idx, s) in enumerate(seeds)
    println("\n================ instance = $idx | seed = $s ================")
    prob = generate_problem(s)
    fstar = prob.optimal_value   # = 0.0 (attained at the feasible planted factors)

    runs = Dict(
        "Short-step"          => conditional_gradient_short_step_matfac(prob.f, prob.grad_f, prob.lmo, prob.U0, prob.V0, prob.L_global; fstar=fstar),
        "Open-loop"           => conditional_gradient_open_matfac(prob.f, prob.grad_f, prob.lmo, prob.U0, prob.V0; fstar=fstar),
        "Adaptive constant"   => conditional_gradient_adaptive_matfac(prob.f, prob.grad_f, prob.lmo, prob.U0, prob.V0, gamma0; fstar=fstar),
        "Adaptive adjustable" => conditional_gradient_adjustable_scaling_matfac(prob.f, prob.grad_f, prob.lmo, prob.U0, prob.V0, gamma0; fstar=fstar),
        "Auto-conditioned"    => conditional_gradient_auto_conditioned_matfac(prob.f, prob.grad_f, prob.lmo, prob.U0, prob.V0; fstar=fstar),
        "Pure backtracking"   => conditional_gradient_pure_backtracking_matfac(prob.f, prob.grad_f, prob.lmo, prob.U0, prob.V0; fstar=fstar),
    )

    for name in method_order
        r = runs[name]
        all_solved[name][idx] = r.solved
        all_times[name][idx]  = r.solved ? r.total_time : Inf

        # Open-loop / Short-step have no L_k sequence -> store empty vector
        Lk = hasproperty(r, :L_ks) ? collect(Float64, r.L_ks) : Float64[]

        f_last   = isempty(r.values) ? NaN : Float64(last(r.values))
        gap_last = isempty(r.gaps)   ? NaN : Float64(last(r.gaps))

        per_iter[name]["seed_$(s)"] = Dict(
            "instance"     => idx,
            "seed"         => s,
            "values"       => collect(Float64, r.values),   # f per iteration
            "gaps"         => collect(Float64, r.gaps),     # FW gap per iteration
            "times"        => collect(Float64, r.times),    # time per iteration
            "L_ks"         => Lk,                           # L_k per iteration
            "iterations"   => r.k,
            "solved"       => r.solved ? 1 : 0,
            "total_time"   => r.total_time,
            "fstar"        => fstar,
            "final_value"  => f_last,
            "final_gap"    => gap_last,
            "final_subopt" => f_last - fstar,
            "tau1"         => prob.tau1,                    # seed-dependent radii
            "tau2"         => prob.tau2,
        )

        status = r.solved ? "solved" : "NOT solved"
        line = string(rpad(name, 22), status,
                      "  iters=", r.k,
                      "  time=", round(r.total_time, digits=2), "s",
                      "  f_last=", round(f_last, digits=4),
                      "  gap_last=", round(gap_last, digits=4))
        println(line)
        push!(report_lines, "instance=$(idx)  seed=$(s)  " * line)
    end
end

# --------------------------------------------------------------------
# 4b. Save per-iteration histories: one .mat per method
#     File: <output_dir>/<problem>_iterations_<Method>.mat
#     Struct: runs.seed_<s>.{values, gaps, times, L_ks, final_value, final_gap, ...}
# --------------------------------------------------------------------
for name in method_order
    fname = joinpath(output_dir,
                     "$(problem_name)_iterations_$(matvarname(name)).mat")
    matwrite(fname, Dict(
        "problem" => problem_name,
        "method"  => name,
        "seeds"   => collect(seeds),
        "runs"    => per_iter[name],
    ))
    println("✅ Saved per-iteration history: ", basename(fname))
end

# plain-text report (per instance/seed: status, iters, time, last f, last gap)
open(joinpath(output_dir, "$(problem_name)_report.txt"), "w") do io
    println(io, "Problem: ", problem_name, "   seeds: ", collect(seeds))
    for l in report_lines
        println(io, l)
    end
end

# --------------------------------------------------------------------
# 5. Summary table + save raw results
# --------------------------------------------------------------------
println("\n# Summary ($(length(seeds))-seed sweep) #")
println(rpad("Method", 22), rpad("Solved", 10), rpad("Mean time (s)", 16), "Median time (s)")
for name in method_order
    t = filter(isfinite, all_times[name])
    n_solved = length(t)
    mean_t = n_solved > 0 ? round(mean(t), digits=3) : NaN
    med_t  = n_solved > 0 ? round(median(t), digits=3) : NaN
    println(rpad(name, 22), rpad("$n_solved/$(length(seeds))", 10), rpad(string(mean_t), 16), string(med_t))

    fv  = [per_iter[name]["seed_$(s)"]["final_value"] for s in seeds]
    gpv = [per_iter[name]["seed_$(s)"]["final_gap"]   for s in seeds]
    println("    last f: mean=", round(mean(fv), sigdigits=6),
            "   last gap: mean=", round(mean(gpv), sigdigits=4))
end

matwrite(joinpath(output_dir, "matrixfactorization_benchmark_all_methods.mat"), Dict(
    "seeds"  => collect(seeds),
    "times"  => Dict(matvarname(name) => all_times[name]  for name in method_order),
    "solved" => Dict(matvarname(name) => all_solved[name] for name in method_order),
))

# --------------------------------------------------------------------
# 6. Dolan-Moré performance profile (ratio-based x-axis)
# See: https://tmigot.github.io/posts/2024/06/teaching/
#
# T[p, s] = metric (total_time) for solver s on problem p; Inf means
# solver s did NOT solve problem p (hit max_iter_cap / max_time) -- the
# +Inf failure penalty, so each curve's plateau height is the fraction of
# problems that solver actually solved.
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

# Fixed name -> color mapping (matches the Lasso figure's assignment)
# rather than a positional list, so each method keeps the same color
# across every subsection's figure even when a given problem only plots
# a subset of the six methods.
method_colors = Dict(
    "Adaptive constant"   => :blue,
    "Adaptive adjustable" => :green,
    "Pure backtracking"   => :red,
    "Auto-conditioned"    => :black,
    "Short-step"          => :orange,
    "Open-loop"           => :purple,
)
for (i, series) in enumerate(p.series_list)
    series[:linecolor] = method_colors[method_order[i]]
end

Plots.savefig(p, joinpath(output_dir, "matrix_factorization_performance_profile_all_methods.eps"))
println("✅ Saved: matrix_factorization_performance_profile_all_methods.eps in ", output_dir)
display(p)
