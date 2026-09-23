# ==========================================================================
# Traffic Assignment (Beckmann formulation) Frank-Wolfe benchmark:
#   6 step-size strategies x N seeds -> combined Dolan-Moré performance profile
# ==========================================================================
# Formulation (Beckmann):
#   min_x  sum_a int_0^{x_a} t_a(xi) dxi
#   s.t.   x_a = sum_r delta_ar y_r ,  a in A
#          sum_{r in R_ij} y_r = d_ij ,  (i,j) in S
#          y_r >= 0
#
#   BPR travel-time:  t_a(x_a) = tau_a * (1 + 0.15*(x_a/c_a)^4),  tau_a = 1
#
#   A linear minimization over the route-flow polytope (equivalently over
#   the feasible link-flow set induced by it) is exactly an all-pairs
#   shortest-route computation -> Dijkstra's algorithm from every source
#   that appears in an OD pair, using the current link "cost" (gradient) as
#   edge weight. This is the LMO used by every Frank-Wolfe variant below.
#
# Instance generation (per seed):
#   - DAG with n=20 layers of m=25 nodes each (mn=500 nodes), edges only
#     between consecutive layers, each with probability p=0.5
#     (|A| ~ 6000 as in the problem statement).
#   - OD pairs S = all reachable node pairs in the DAG (|S| ~ 113000).
#   - Demand d_ij ~ Uniform(0,1) for every (i,j) in S, as in
#     Combettes & Pokutta, "Boosting Frank-Wolfe by chasing gradients", 2020.
#
# "Solved" criterion (identical philosophy as in the Lasso benchmark):
#     (a) FW gap <= epsilon                      (gap <= epsilon)
#  OR (b) f(x^k) - fstar <= epsilon
# Here fstar is NOT known exactly (unlike the Lasso instance). Since every
# term x_a + 0.03*x_a^5/c^4 is nonnegative for x_a >= 0, f(x) >= 0 for every
# feasible x, so fstar = 0.0 is a valid (if generally loose) lower bound.
# Criterion (a) is therefore the one that effectively drives convergence
# in practice; criterion (b) is kept only for consistency with the same
# generic solvers used in the Lasso benchmark.
#
# All six solvers below are the exact generic Frank-Wolfe step-size
# strategies from the Lasso benchmark (they only use f, grad_f, lmo, x0),
# reused unchanged. A safety net (max_iter_cap / max_time) prevents a
# genuine infinite loop; hitting it does NOT count as solving
# (time = Inf, excluded from the profile at that instance).
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
using Graphs
using Zygote

pyplot()
PyPlot.PyCall.pyimport("warnings").filterwarnings("ignore")

problem_name = "traffic_assignment"
output_dir = joinpath(@__DIR__, problem_name)
mkpath(output_dir)

# sanitize method names into valid .mat struct field names
matvarname(s) = replace(s, r"[^A-Za-z0-9_]" => "_")

# --------------------------------------------------------------------
# 1. Problem generator — one seed -> one independent traffic-assignment
#    instance (DAG topology, OD demands, LMO, Beckmann objective, L est.)
# --------------------------------------------------------------------
function generate_problem(seed; m=15, n=10, p=0.4)
    Random.seed!(seed)
    mn = m * n

    # ---- Directed acyclic layered graph -----------------------------
    DG = SimpleDiGraph(mn)
    for i in 1:(mn - m)
        l = (i - 1) ÷ m
        for h in 0:(m - 1)
            if rand() < p
                add_edge!(DG, i, (l + 1) * m + h + 1)
            end
        end
    end

    out_neighbors = Dict{Int, Vector{Int}}()
    for e in edges(DG)
        push!(get!(out_neighbors, src(e), Int[]), dst(e))
    end

    # ---- All reachable OD pairs via BFS from every source ------------
    OD = Tuple{Int, Int}[]
    for s in 1:mn
        haskey(out_neighbors, s) || continue
        append!(OD, [(s, j) for j in out_neighbors[s]])
        queue = copy(out_neighbors[s])
        dequeued = Int[]
        while !isempty(queue)
            i = popfirst!(queue)
            push!(dequeued, i)
            haskey(out_neighbors, i) || continue
            for j in out_neighbors[i]
                if j ∉ dequeued && j ∉ queue
                    push!(OD, (s, j))
                    push!(queue, j)
                end
            end
        end
    end

    # ---- Demands ~ Uniform(0,1) for every OD pair ---------------------
    demands = Dict((i, j) => rand() for (i, j) in OD)

    edge_indices = [(src(e), dst(e)) for e in edges(DG)]
    c = length(OD) / (m * n)   # aggregate practical-capacity scalar

    # Precompute, per source, the list of targets (for LMO efficiency)
    targets_by_source = Dict{Int, Vector{Int}}()
    for (s, t) in OD
        push!(get!(targets_by_source, s, Int[]), t)
    end
    unique_sources = unique(first.(OD))

    # ---- Linear Minimization Oracle: all-pairs shortest routes --------
    function lmo(g)
        v = zeros(mn, mn)
        weights = reshape(g, mn, mn)

        W = fill(Inf, mn, mn)
        for e in edges(DG)
            i, j = src(e), dst(e)
            W[i, j] = weights[i, j]
        end

        for s in unique_sources
            ds = dijkstra_shortest_paths(DG, s, W)
            for t in targets_by_source[s]
                path = enumerate_paths(ds, t)
                if !isempty(path)
                    for idx in 1:(length(path) - 1)
                        v[path[idx], path[idx + 1]] += demands[(s, t)]
                    end
                end
            end
        end
        return vec(v)
    end

    # ---- Beckmann objective with BPR travel-time (tau_a = 1) ---------
    #   int_0^{x_a} 1 + 0.15*(xi/c)^4 dxi  =  x_a + 0.03*x_a^5 / c^4
    function f(x)
        X = reshape(x, mn, mn)
        return sum(X[i, j] + 0.03 * X[i, j]^5 / c^4 for (i, j) in edge_indices) / mn^2
    end

    grad_f(x) = Zygote.gradient(f, x)[1]

    x0 = lmo(rand(mn^2))

    total_demand = sum(values(demands))
    # global Lipschitz-constant estimate for the (separable, quartic-growth)
    # BPR-based objective on the feasible set (used by short-step & as the
    # curvature-estimate seed for the adaptive/backtracking variants)
    L = 0.6 * total_demand^3 / (c^4 * mn^2)

    return (f=f, grad_f=grad_f, lmo=lmo, x0=x0, L=L, mn=mn, seed=seed,
            n_links=length(edge_indices), n_od=length(OD),
            optimal_value=0.0)   # fstar=0.0 is a valid lower bound (f>=0)
end

# --------------------------------------------------------------------
# 2. Solvers — all `while true`, all return (..., solved, total_time)
#    (identical generic Frank-Wolfe step-size strategies as in the
#    Lasso benchmark; they only depend on f, grad_f, lmo, x0)
# --------------------------------------------------------------------
function conditional_gradient_adaptive(f, grad_f, lmo, x0, gamma;
        epsilon=1e-3, delta=1e-10, beta=2, max_iter_cap=3000, max_time=1000.0,
        fstar=0.0)
    x_prev = copy(x0); x_curr = copy(x0)
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
        push!(times, time() - start)
        current_f = f(x_curr); prev_grad = current_grad
        push!(gaps, gap); push!(values, current_f); push!(L_ks, L_k)
        if gap <= epsilon || (current_f - fstar) <= epsilon
            solved = true
            break
        end
        if k >= max_iter_cap #|| (time() - t_start_total) > max_time
            solved = false
            break
        end
    end
    return (x=x_curr, values=values, times=times, gaps=gaps, L_ks=L_ks,
            k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_adjustable_scaling(f, grad_f, lmo, x0, gamma;
        epsilon=1e-3, delta=1e-10, beta=2, max_iter_cap=3000, max_time=1000.0,
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
        push!(times, time() - start)
        current_f = f(x_curr); prev_grad = current_grad
        push!(gaps, gap); push!(values, current_f); push!(L_ks, L_k)
        if gap <= epsilon || (current_f - fstar) <= epsilon
            solved = true
            break
        end
        if k >= max_iter_cap #|| (time() - t_start_total) > max_time
            solved = false
            break
        end
    end
    return (x=x_curr, values=values, times=times, gaps=gaps, L_ks=L_ks,
            gamma_history=gamma_history, k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_auto_conditioned(f, grad_f, lmo, x0;
        epsilon=1e-3, delta=1e-10, max_iter_cap=3000, max_time=1000.0,
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
        push!(times, time() - start)
        current_f = new_f
        push!(gaps, gap); push!(values, current_f); push!(L_ks, hat_L_k)
        if gap <= epsilon || (current_f - fstar) <= epsilon
            solved = true
            break
        end
        if k >= max_iter_cap #|| (time() - t_start_total) > max_time
            solved = false
            break
        end
    end
    return (x=x_curr, values=values, times=times, gaps=gaps, L_ks=L_ks,
            k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_pure_backtracking(f, grad_f, lmo, x0;
        epsilon=1e-3, max_iter_cap=3000, max_time=1000.0, fstar=0.0)
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
        f_new = f_prev
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
        M = M * eta
        k += 1
        push!(times, time() - start)
        current_f = f_new
        push!(gaps, gap); push!(values, current_f); push!(L_ks, M)
        if gap <= epsilon || (current_f - fstar) <= epsilon
            solved = true
            break
        end
        if k >= max_iter_cap #|| (time() - t_start_total) > max_time
            solved = false
            break
        end
    end
    return (x=x_prev, values=values, times=times, gaps=gaps, L_ks=L_ks,
            k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_open(f, grad_f, lmo, x0;
        epsilon=1e-3, max_iter_cap=3000, max_time=1000.0, fstar=0.0)
    x = copy(x0)
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
        push!(times, time() - start)
        f_prev = f(x)
        gap = -dot(grad, d)
        push!(gaps, gap); push!(values, f_prev)
        if gap <= epsilon || (f_prev - fstar) <= epsilon
            solved = true
            break
        end
        if k >= max_iter_cap #|| (time() - t_start_total) > max_time
            solved = false
            break
        end
    end
    return (x=x, values=values, times=times, gaps=gaps, k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_short_step(f, grad_f, lmo, x0, L;
        epsilon=1e-3, max_iter_cap=3000, max_time=1000.0, fstar=0.0)
    x = copy(x0)
    values = [f(x)]; times = [0.0]; gaps = Float64[]
    k = 0; solved = false
    t_start_total = time()

    while true
        start = time()
        grad = grad_f(x)
        v = lmo(grad)
        d = v - x
        gap = -dot(grad, d)

        normd2 = dot(d, d)
        t_k = min(gap / (L * normd2), 1.0)
        x = x + t_k * d
        k += 1
        push!(times, time() - start)
        current_f = f(x)
        push!(gaps, gap); push!(values, current_f)

        if gap <= epsilon || (current_f - fstar) <= epsilon
            solved = true
            break
        end
        if k >= max_iter_cap #|| (time() - t_start_total) > max_time
            solved = false
            break
        end
    end
    return (x=x, values=values, times=times, gaps=gaps, k=k, solved=solved, total_time=sum(times))
end

# --------------------------------------------------------------------
# 3. Run all 6 methods on each of the N seeded traffic-assignment instances
# --------------------------------------------------------------------
println("\n================ Traffic Assignment Problem ================")
seeds  = 1:100
gamma0 = 1/4
EPSILON      = 1e-3
MAX_ITER_CAP = 100000
MAX_TIME     = 1000.0

method_order = ["Adaptive constant", "Adaptive adjustable", "Pure backtracking", "Auto-conditioned",
               # "Short-step", 
                "Open-loop"]

all_times  = Dict(name => fill(Inf, length(seeds)) for name in method_order)
all_solved = Dict(name => falses(length(seeds)) for name in method_order)

per_iter     = Dict(name => Dict{String,Any}() for name in method_order)
report_lines = String[]

for (idx, s) in enumerate(seeds)
    println("\n================ instance = $idx | seed = $s ================")
    prob  = generate_problem(s)
    L_est = prob.L
    fstar = prob.optimal_value   # 0.0, a valid lower bound (f(x) >= 0 for all feasible x)

    println("  |A| ≈ ", prob.n_links, "   |S| ≈ ", prob.n_od,
            "   dim(x) = ", prob.mn^2, "   L_est = ", round(L_est, sigdigits=4))

    runs = Dict(
        #"Short-step"          => conditional_gradient_short_step(prob.f, prob.grad_f, prob.lmo, prob.x0, L_est; epsilon=EPSILON, max_iter_cap=MAX_ITER_CAP, max_time=MAX_TIME, fstar=fstar),
        "Open-loop"           => conditional_gradient_open(prob.f, prob.grad_f, prob.lmo, prob.x0;
                                        epsilon=EPSILON, max_iter_cap=MAX_ITER_CAP, max_time=MAX_TIME, fstar=fstar),
        "Adaptive constant"   => conditional_gradient_adaptive(prob.f, prob.grad_f, prob.lmo, prob.x0, gamma0;
                                        epsilon=EPSILON, max_iter_cap=MAX_ITER_CAP, max_time=MAX_TIME, fstar=fstar),
        "Adaptive adjustable" => conditional_gradient_adjustable_scaling(prob.f, prob.grad_f, prob.lmo, prob.x0, gamma0;
                                        epsilon=EPSILON, max_iter_cap=MAX_ITER_CAP, max_time=MAX_TIME, fstar=fstar),
        "Auto-conditioned"    => conditional_gradient_auto_conditioned(prob.f, prob.grad_f, prob.lmo, prob.x0;
                                        epsilon=EPSILON, max_iter_cap=MAX_ITER_CAP, max_time=MAX_TIME, fstar=fstar),
        "Pure backtracking"   => conditional_gradient_pure_backtracking(prob.f, prob.grad_f, prob.lmo, prob.x0;
                                        epsilon=EPSILON, max_iter_cap=MAX_ITER_CAP, max_time=MAX_TIME, fstar=fstar),
    )

    for name in method_order
        r = runs[name]
        all_solved[name][idx] = r.solved
        all_times[name][idx]  = r.solved ? r.total_time : Inf

        Lk = hasproperty(r, :L_ks) ? collect(Float64, r.L_ks) : Float64[]

        f_last   = isempty(r.values) ? NaN : Float64(last(r.values))
        gap_last = isempty(r.gaps)   ? NaN : Float64(last(r.gaps))

        per_iter[name]["seed_$(s)"] = Dict(
            "instance"     => idx,
            "seed"         => s,
            "values"       => collect(Float64, r.values),
            "gaps"         => collect(Float64, r.gaps),
            "times"        => collect(Float64, r.times),
            "L_ks"         => Lk,
            "iterations"   => r.k,
            "solved"       => r.solved ? 1 : 0,
            "total_time"   => r.total_time,
            "fstar"        => fstar,
            "final_value"  => f_last,
            "final_gap"    => gap_last,
            "final_subopt" => f_last - fstar,
            "n_links"      => prob.n_links,
            "n_od"         => prob.n_od,
        )

        status = r.solved ? "solved" : "NOT solved"
        line = string(rpad(name, 22), status,
                      "  iters=", r.k,
                      "  time=", round(r.total_time, digits=2), "s",
                      "  f_last=", round(f_last, digits=6),
                      "  gap_last=", round(gap_last, digits=6))
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
    println("✅ Saved per-iteration history: ", basename(fname))
end

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
# 5. Dolan-Moré performance profile (ratio-based x-axis)
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

method_colors = Dict(
    "Adaptive constant"   => :blue,
    "Adaptive adjustable" => :green,
    "Pure backtracking"   => :red,
    "Auto-conditioned"    => :black,
  #  "Short-step"          => :orange,
    "Open-loop"           => :purple,
)
for (i, series) in enumerate(p.series_list)
    series[:linecolor] = method_colors[method_order[i]]
end

Plots.savefig(p, joinpath(output_dir, "traffic_assignment_performance_profile_all_methods.eps"))
println("✅ Saved: traffic_assignment_performance_profile_all_methods.eps in ", output_dir)
display(p)