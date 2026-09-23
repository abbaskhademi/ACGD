# ==========================================================================
# matrix_balancing  Frank-Wolfe: all 6 methods x N seeds -> combined performance profile
# ==========================================================================
# "Solved" = EITHER of the method's own convergence criteria fires:
#     (a) FW gap <= epsilon                    (gap <= epsilon)
#     (b) objective is within epsilon of the TRUE optimum
#         (current_f - fstar <= epsilon), where fstar = prob.optimal_value
#         (NOT 0 — the unconstrained/attained optimum of this problem is
#         sum(A), achieved when x is constant, since A is symmetric).
# This is NOT an artificial max_iter cutoff.
#
# Every solver's inner loop is `while true` (no max_iter bound). A safety
# net (max_iter_cap / max_time) prevents a genuine infinite loop on a run
# that never satisfies epsilon — hitting it does NOT count as solving
# (time = Inf, excluded from the profile at that point).
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
using Zygote
using SparseArrays
using BenchmarkProfiles

pyplot()
PyPlot.PyCall.pyimport("warnings").filterwarnings("ignore")

# --- output goes into <run dir>/matrix_balancing/ -------------------
problem_name = "matrix_balancing"
output_dir   = joinpath(@__DIR__, problem_name)
mkpath(output_dir)

# sanitize method names into valid .mat struct field names
matvarname(s) = replace(s, r"[^A-Za-z0-9_]" => "_")

# --------------------------------------------------------------------
# 1. Problem generator — one seed -> one independent matrix_balancing  instance
# --------------------------------------------------------------------
function generate_problem(seed; n=100, a=5, l=1.0, u=10.0)
    Random.seed!(seed)
    A_sparse = sprand(n, n, a/n)
    A = abs.(Matrix(A_sparse) + Matrix(A_sparse)') + 0.05 * I

    f(x) = sum(A .* exp.(x .- x'))          # vectorized, see point 5
    grad_f(x) = Zygote.gradient(f, x)[1]

    function lmo(g)
        return [g[i] > 0 ? l : u for i in 1:n]
    end

    x0 = l .+ (u - l) .* rand(n)

    return (f=f, grad_f=grad_f, lmo=lmo, x0=x0, A=A, seed=seed, optimal_value=sum(A))
end

function estimate_L(A; τ = 9.0)
    off = A - Diagonal(diag(A))
    Lap = Diagonal(vec(sum(off, dims=2))) - off
    M = (exp(τ) + exp(-τ)) * Lap + Diagonal(2 .* diag(A))
    return eigmax(Symmetric(M))
end

# --------------------------------------------------------------------
# 2. Solvers — all `while true`, all return (..., solved, total_time)
#    All now accept `fstar` (true optimal objective value) and stop when
#    gap <= epsilon  OR  current_f - fstar <= epsilon.
# --------------------------------------------------------------------
function conditional_gradient_adaptive(f, grad_f, lmo, x0, gamma;
        epsilon=1e-5, delta=1e-10, beta=2, max_iter_cap=100000, max_time=500.0,
        fstar=0.0)
    x_prev = copy(x0); x_curr = copy(x0)
    f0 = f(x0)
    values = [f(x_curr)]; times = [0.0]; gaps = Float64[]; L_ks = Float64[]
    k = 0; solved = false
    prev_grad = grad_f(x_prev); current_f = f(x_curr)
    t_start_total = time()
    while true
        start = time()
        current_grad = grad_f(x_curr)
        v = lmo(current_grad)
        d = v - x_curr
        normd2 = dot(d, d)
        gap = -dot(current_grad, d)
        if k == 0
            Random.seed!(23)
            d0 = randn(length(x0))
            L_k = gamma * ((norm(grad_f(x0) - grad_f(x0 + 1e-3 * d0)) / (1e-3 * norm(d0))) + delta)
        else
            L_k = gamma * ((norm(current_grad - prev_grad) / norm(x_curr - x_prev)) + delta)
        end
        Lknormd2 = L_k * normd2
        t_k = min(gap / Lknormd2, 1.0)
        while true
            x_new = x_curr + t_k * d
            new_f = f(x_new)
            if current_f - new_f >= t_k * gap - (Lknormd2 / 2) * t_k^2
                x_prev = copy(x_curr); x_curr = x_new
                break
            else
                L_k *= beta
                Lknormd2 = L_k * normd2
                t_k = min(gap / Lknormd2, 1.0)
            end
        end
        k += 1
        push!(times, time() - start);
        current_f = f(x_curr); prev_grad = current_grad
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
    return (x=x_curr, values=values, times=times, gaps=gaps, L_ks=L_ks,
            k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_adjustable_scaling(f, grad_f, lmo, x0, gamma;
        epsilon=1e-5, delta=1e-10, beta=2, max_iter_cap=100000, max_time=500.0,
        fstar=0.0)
    x_prev = copy(x0); x_curr = copy(x0)
    f0 = f(x0)
    values = [f(x_curr)]; times = [0.0]; gaps = Float64[]; L_ks = Float64[]
    gamma_history = [gamma]
    k = 0; solved = false
    prev_grad = grad_f(x_prev); current_f = f(x_curr)
    recent_backtracks = Int[]
    t_start_total = time()
    while true
        start = time()
        current_grad = grad_f(x_curr)
        v = lmo(current_grad)
        d = v - x_curr
        normd2 = dot(d, d)
        gap = -dot(current_grad, d)
        if k == 0
            Random.seed!(23)
            d0 = randn(length(x0))
            L_k = gamma * ((norm(grad_f(x0) - grad_f(x0 + 1e-3 * d0)) / (1e-3 * norm(d0))) + delta)
        else
            L_k = gamma * ((norm(current_grad - prev_grad) / norm(x_curr - x_prev)) + delta)
        end
        Lknormd2 = L_k * normd2
        t_k = min(gap / Lknormd2, 1.0)
        i = 0
        while true
            x_new = x_curr + t_k * d
            new_f = f(x_new)
            if current_f - new_f >= t_k * gap - (Lknormd2 / 2) * t_k^2
                x_prev = copy(x_curr); x_curr = x_new
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
        current_f = f(x_curr); prev_grad = current_grad
        push!(gaps, gap); push!(values, current_f);  push!(L_ks, L_k)
        if gap <= epsilon || (current_f - fstar) <= epsilon
            solved = true
            break
        end
        if k >= max_iter_cap # || (time() - t_start_total) > max_time
            solved = false
            break
        end
    end
    return (x=x_curr, values=values, times=times, gaps=gaps, L_ks=L_ks,
            gamma_history=gamma_history, k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_auto_conditioned(f, grad_f, lmo, x0;
        epsilon=1e-5, delta=1e-10, max_iter_cap=100000, max_time=500.0,
        fstar=0.0)
    x_curr = copy(x0)
    f0 = f(x0)
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
        d = v - x_curr
        normd2 = dot(d, d)
        gap = -dot(current_grad, d)
        hat_L_k = maximum(L_history)
        Lknormd2 = hat_L_k * normd2
        t_k = min(gap / Lknormd2, 1.0)
        x_new = x_curr + t_k * d
        new_f = f(x_new)
        s = x_new - x_curr
        norms2 = dot(s, s)
        if norms2 > 0
            L_k_realized = 2 * (new_f - current_f - dot(current_grad, s)) / norms2
            push!(L_history, L_k_realized)
        else
            push!(L_history, hat_L_k)
        end
        x_curr = x_new
        k += 1
        push!(times, time() - start);
        current_f = new_f
        push!(gaps, gap); push!(values, current_f); push!(L_ks, hat_L_k)
        if gap <= epsilon || (current_f - fstar) <= epsilon
            solved = true
            break
        end
        if k >= max_iter_cap # || (time() - t_start_total) > max_time
            solved = false
            break
        end
    end
    return (x=x_curr, values=values, times=times, gaps=gaps, L_ks=L_ks,
            k=k, solved=solved, total_time=sum(times))
end

# --------------------------------------------------------------------
# FIX #2 applied here: `f_new` is now pre-declared *before* the inner
# backtracking `while` loop (mirroring the Lasso script), so it stays in
# scope afterward. We then set `current_f = f_new` and check
# `current_f - fstar <= epsilon` against the POST-STEP objective, not the
# stale pre-step `f_prev` value that the previous version accidentally
# used. `values`/`L_ks` now push `current_f` directly instead of
# recomputing `f(x_new)` a second time.
# --------------------------------------------------------------------
function conditional_gradient_pure_backtracking(f, grad_f, lmo, x0;
        epsilon=1e-5, max_iter_cap=100000, max_time=500.0,
        fstar=0.0)
    x_prev = copy(x0)
    f0 = f(x0)
    values = [f(x_prev)]; times = [0.0]; gaps = Float64[]; L_ks = Float64[]
    k = 0; solved = false
    tau_bt = 2.0; eta = 0.9
    t_start_total = time()

    Random.seed!(23)
    d0 = randn(length(x0))
    L_minus1 = norm(grad_f(x0) - grad_f(x0 + 1e-3 * d0)) / (1e-3 * norm(d0))
    M = L_minus1 * eta
    while true
        start = time()
        grad = grad_f(x_prev)
        v = lmo(grad)
        d = v - x_prev
        normd2 = norm(d)^2
        gap = -dot(grad, d)
        f_prev = f(x_prev)
        t_k = min(gap / (M * normd2), 1)
        x_new = x_prev
        f_new = f_prev   # pre-declared so it survives past the inner loop
        while true
            x_new = x_prev + t_k * d
            f_new = f(x_new)
            if f_prev - f_new >= t_k * gap - (M / 2) * t_k^2 * normd2
                break
            else
                M *= tau_bt
                t_k = min(gap / (M * normd2), 1)
            end
        end
        x_prev = x_new
        current_f = f_new
        k += 1
        push!(times, time() - start);
        push!(gaps, gap); push!(values, current_f); push!(L_ks, M)
        M = M * eta
        if gap <= epsilon || (current_f - fstar) <= epsilon
            solved = true
            break
        end
        if k >= max_iter_cap # || (time() - t_start_total) > max_time
            solved = false
            break
        end
    end
    return (x=x_prev, values=values, times=times, gaps=gaps, L_ks=L_ks,
            k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_open(f, grad_f, lmo, x0;
        epsilon=1e-5, max_iter_cap=100000, max_time=500.0,
        fstar=0.0)
    x = copy(x0)
    f0 = f(x0)
    values = [f(x)]; times = [0.0]; gaps = Float64[]
    k = 0; solved = false
    t_start_total = time()

    while true
        start = time()
        grad = grad_f(x)
        v = lmo(grad)
        d = v - x
        t_k = 2 / (2 + k)
        x = x + t_k * d
        k += 1
        push!(times, time() - start);
        gap = -dot(grad, d)
        f_prev = f(x)
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
    return (x=x, values=values, times=times, gaps=gaps, k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_short_step(f, grad_f, lmo, x0, L;
        epsilon=1e-5, max_iter_cap=100000, max_time=500.0,
        fstar=0.0)
    x = copy(x0)
    f0 = f(x0)
    values = [f(x)]; times = [0.0]; gaps = Float64[]
    k = 0; solved = false
    t_start_total = time()

    while true
        start = time()
        grad = grad_f(x)
        v = lmo(grad)
        d = v - x
        gap = -dot(grad, d)
        f_prev = f(x)

        t_k = min(gap / (L * norm(d)^2), 1.0)
        x = x + t_k * d
        k += 1
        push!(times, time() - start)
        push!(gaps, gap); push!(values, f_prev);

        if gap <= epsilon  || (f_prev - fstar) <= epsilon
            solved = true
            break
        end
        if k >= max_iter_cap # || (time() - t_start_total) > max_time
            solved = false
            break
        end
    end
    return (x=x, values=values, times=times, gaps=gaps, k=k, solved=solved, total_time=sum(times))
end

# --------------------------------------------------------------------
# 3. Run all methods on each of the N seeded problems
# --------------------------------------------------------------------
seeds  = 1:100

gamma0 = 1/4

method_order = ["Adaptive constant", "Adaptive adjustable", "Pure backtracking", "Auto-conditioned",
                "Open-loop",
                #"Short-step"
                ]

all_times  = Dict(name => fill(Inf, length(seeds)) for name in method_order)
all_solved = Dict(name => falses(length(seeds)) for name in method_order)

# FIX #1: use Dict{String,Any} (base Julia), NOT Dictionaries.jl's
# `Dictionary`, which was never `using`-imported and would throw
# UndefVarError at runtime.
per_iter     = Dict(name => Dict{String,Any}() for name in method_order)
report_lines = String[]                      # text report accumulator

for (idx, s) in enumerate(seeds)
    println("\n================ instance = $idx | seed = $s ================")
    prob  = generate_problem(s)
    # L_est = estimate_L(prob.A)   # only needed if Short-step is re-enabled
    fstar = prob.optimal_value   # true optimal objective value (sum(A)), NOT 0

    runs = Dict(
        #"Short-step"          => conditional_gradient_short_step(prob.f, prob.grad_f, prob.lmo, prob.x0, L_est; fstar=fstar),
        "Open-loop"           => conditional_gradient_open(prob.f, prob.grad_f, prob.lmo, prob.x0; fstar=fstar),
        "Adaptive constant"   => conditional_gradient_adaptive(prob.f, prob.grad_f, prob.lmo, prob.x0, gamma0; fstar=fstar),
        "Adaptive adjustable" => conditional_gradient_adjustable_scaling(prob.f, prob.grad_f, prob.lmo, prob.x0, gamma0; fstar=fstar),
        "Auto-conditioned"    => conditional_gradient_auto_conditioned(prob.f, prob.grad_f, prob.lmo, prob.x0; fstar=fstar),
        "Pure backtracking"   => conditional_gradient_pure_backtracking(prob.f, prob.grad_f, prob.lmo, prob.x0; fstar=fstar),
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
        )

        status = r.solved ? "solved" : "NOT solved"
        line = string(rpad(name, 22), status,
                      "  iters=", r.k,
                      "  time=", round(r.total_time, digits=4), "s",
                      "  f_last=", round(f_last, digits=8),
                      "  gap_last=", round(gap_last, sigdigits=4))
        println(line)
        push!(report_lines, "instance=$(idx)  seed=$(s)  " * line)
    end
end

# --------------------------------------------------------------------
# 3b. Save per-iteration histories: one .mat per method
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
    println("Saved per-iteration history: ", basename(fname))
end

# plain-text report (per instance/seed: status, iters, time, last f, last gap)
open(joinpath(output_dir, "$(problem_name)_report.txt"), "w") do io
    println(io, "Problem: ", problem_name, "   seeds: ", collect(seeds))
    for l in report_lines
        println(io, l)
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

    fv  = [per_iter[name]["seed_$(s)"]["final_value"] for s in seeds]
    gpv = [per_iter[name]["seed_$(s)"]["final_gap"]   for s in seeds]
    println("    last f: mean=", round(mean(fv), digits=6),
            "   last gap: mean=", round(mean(gpv), sigdigits=4))
end

matwrite(joinpath(output_dir, "benchmark_all_methods.mat"), Dict(
    "seeds"  => collect(seeds),
    "times"  => Dict(matvarname(name) => all_times[name]  for name in method_order),
    "solved" => Dict(matvarname(name) => all_solved[name] for name in method_order),
))

# --------------------------------------------------------------------
# 5. Dolan-Moré performance profile
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

# --------------------------------------------------------------------
# FIX #3: color mapping is now keyed by METHOD NAME, not by position in
# `method_order`. Since "Short-step" is dropped from this benchmark, a
# purely positional color list would silently shift every color after
# "Adaptive adjustable" — e.g. "Pure backtracking" would render black
# instead of red, and "Auto-conditioned" red instead of black — breaking
# visual consistency with companion figures (e.g. the Lasso benchmark)
# that use the full 6-method roster. This dict is stable regardless of
# which subset of methods is included or their order.
# --------------------------------------------------------------------
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

Plots.savefig(p, joinpath(output_dir,
    "Matrix_Balancing_performance_profile_all_methods.eps"))
println("\n✅ Saved performance profile")
display(p)