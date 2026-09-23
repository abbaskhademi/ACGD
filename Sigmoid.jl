# ==========================================================================
# Non-Convex Frank-Wolfe: w8a Dataset (Squared Sigmoid Loss)
# All 6 methods x N seeds -> combined performance profile
# ==========================================================================
# "Solved" = Frank-Wolfe gap <= epsilon. 
# NOTE: The (current_f - fstar) <= epsilon check is DISABLED (commented out) 
# because the true global optimum is unknown for non-convex problems.
#
# Every solver's inner loop is `while true` (no max_iter bound in the condition). 
# A safety net (max_iter_cap) prevents genuine infinite loops.
# ==========================================================================

using Random, LinearAlgebra, Statistics, SparseArrays
using CSV, Tables          # For loading local CSV files (matches pandas read_csv)
using Zygote
using BenchmarkProfiles, Plots
using MAT, Measures 
using LIBSVMdata

pyplot()
PyPlot.PyCall.pyimport("warnings").filterwarnings("ignore")

problem_name = "Sigmoid_w8a"
output_dir = joinpath(@__DIR__, "Sigmoid_w8a")
mkpath(output_dir)

# sanitize method names into valid .mat struct field names
matvarname(s) = replace(s, r"[^A-Za-z0-9_]" => "_")

# --------------------------------------------------------------------
# 1. Load the FULL w8a dataset ONCE (outside the seed loop)
# --------------------------------------------------------------------
Random.seed!(23)

println("Loading w8a dataset using LIBSVMdata...")
A_sparse, y_raw = load_dataset("w8a",
    dense   = false,
    replace = false,
    verbose = true,
)

A_full = A_sparse
y_full = (y_raw .+ 1) ./ 2    # CRITICAL: convert {-1,1} → {0,1}

m_total = size(A_full, 1)
n_total = size(A_full, 2)

println("✅ w8a loaded: m_total=$m_total, n_total=$n_total")
println("   Unique labels: ", unique(y_full))

# Alternative: If you ever want to use LIBSVMdata instead, uncomment below:
# using LIBSVMdata
# A_full, y_full = load_dataset("w8a", dense=false, replace=false, verbose=true)
# y_full = (y_full .+ 1) ./ 2

#sigmoid(z) = 1 / (1 + exp(-z))
# sigmoid(z) = 1.0 / (1.0 + exp(-z))
sigmoid(z) = z >= 0 ? 1/(1+exp(-z)) : exp(z)/(1+exp(z))  # stable sigmoid
# --------------------------------------------------------------------
# 2. Problem generator — subsamples rows/cols to build one benchmark instance
# --------------------------------------------------------------------
function generate_problem(seed::Int; m_sub::Int=2000, n_sub::Int=4000, tau::Float64=50.0)
    @assert m_total >= m_sub "m_total=$m_total < requested m_sub=$m_sub"
    @assert n_total >= n_sub "n_total=$n_total < requested n_sub=$n_sub"

    Random.seed!(seed)

    row_indices = randperm(m_total)[1:m_sub]
    col_indices = randperm(n_total)[1:n_sub]

    A_k = A_full[row_indices, col_indices]
    y_k = y_full[row_indices]

    m, n = size(A_k)

    # Non-convex objective: Squared-error loss on sigmoid predictions
    function f(x)
        predictions = sigmoid.(A_k * x)
        errors = y_k .- predictions
        return mean(errors .^ 2)
    end

    # Automatic differentiation for the gradient
    grad_f(x) = Zygote.gradient(f, x)[1]

    # Linear Minimization Oracle (LMO) over the L1 ball of radius tau
    function lmo(g)
        i = argmax(abs.(g))
        v = zeros(n)
        v[i] = -tau * sign(g[i])
        return v
    end

    x0 = zeros(n)

    # Lipschitz constant estimate for Short-step (safe overestimate for sigmoid squared loss)
    # Note: For large sparse matrices, opnorm(Array(A_k)) converts to dense. 
    # If memory is tight, use: L_est = (norm(A_k, 1) * norm(A_k, Inf)) / (4 * m)
    L_est = (opnorm(Array(A_k))^2) / (4 * m)
    println("Lipschitz constant estimate (L): ", round(L_est, digits=4))
    
    # For non-convex problems, true fstar is unknown. We set it to NaN.
    optimal_value = NaN

    return (f=f, grad_f=grad_f, lmo=lmo, x0=x0, L=L_est,
            optimal_value=optimal_value, A=A_k, y=y_k, m=m, n=n,
            m_sub=m_sub, n_sub=n_sub, tau=tau, seed=seed)
end

# --------------------------------------------------------------------
# 3. Solvers — all `while true`, all return (..., solved, total_time)
#    Stopping criterion relies ONLY on FW gap <= epsilon.
# --------------------------------------------------------------------
function conditional_gradient_adaptive(f, grad_f, lmo, x0, gamma;
        epsilon=1e-3, delta=1e-10, beta=2, max_iter_cap=100000, max_time=500.0,
        fstar=NaN)
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
        push!(times, time() - start);
        current_f = f(x_curr); prev_grad = current_grad
        push!(gaps, gap); push!(values, current_f); push!(L_ks, L_k)
        
        # ONLY FW gap check for non-convex problems
        if gap <= epsilon # || (current_f - fstar) <= epsilon
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
        epsilon=1e-3, delta=1e-10, beta=2, max_iter_cap=100000, max_time=500.0,
        fstar=NaN)
    x_prev = copy(x0); x_curr = copy(x0)
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
        
        if gap <= epsilon # || (current_f - fstar) <= epsilon
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
        epsilon=1e-3, delta=1e-10, max_iter_cap=100000, max_time=500.0,
        fstar=NaN)
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
        
        if gap <= epsilon # || (current_f - fstar) <= epsilon
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

function conditional_gradient_pure_backtracking(f, grad_f, lmo, x0;
        epsilon=1e-3, max_iter_cap=100000, max_time=500.0, fstar=NaN)
    x_prev = copy(x0)
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
        
        if gap <= epsilon # || (current_f - fstar) <= epsilon
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
        epsilon=1e-3, max_iter_cap=100000, max_time=500.0, fstar=NaN)
    x = copy(x0)
    values = [f(x)]; times = [0.0]; gaps = Float64[]
    k = 0; solved = false
    t_start_total = time()

    while true
        start = time()
        grad = grad_f(x)
        v = lmo(grad)
        d = v - x

        t_k = 1.0 / sqrt(1.0 + k)
        x = x + t_k * d
        k += 1
        push!(times, time() - start)
        f_prev = f(x)
        gap = -dot(grad, d)
        push!(gaps, gap); push!(values, f_prev)
        
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

function conditional_gradient_short_step(f, grad_f, lmo, x0, L;
        epsilon=1e-3, max_iter_cap=100000, max_time=500.0, fstar=NaN)
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
        
        if gap <= epsilon # || (current_f - fstar) <= epsilon
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
# 4. Run all 6 methods on each of the N seeded w8a sub-problems
# --------------------------------------------------------------------
println("\n========== Sparse Least Squares Sigmoid Regression Problem ===========")
seeds  = 1:100
gamma0 = 1/4

m_sub = 2000
n_sub = 300  
tau_val = 50.0

method_order = ["Adaptive constant", "Adaptive adjustable", "Pure backtracking", "Auto-conditioned",
                 "Short-step", "Open-loop"]

all_times  = Dict(name => fill(Inf, length(seeds)) for name in method_order)
all_solved = Dict(name => falses(length(seeds)) for name in method_order)

per_iter     = Dict(name => Dict{String,Any}() for name in method_order)
report_lines = String[]

for (idx, s) in enumerate(seeds)
    println("\n================ instance = $idx | seed = $s ================")
    prob  = generate_problem(s; m_sub=m_sub, n_sub=n_sub, tau=tau_val)
    fstar = prob.optimal_value   # NaN for non-convex

    runs = Dict(
        "Short-step"          => conditional_gradient_short_step(prob.f, prob.grad_f, prob.lmo, prob.x0, prob.L; fstar=fstar),
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
            "final_subopt" => isnan(fstar) ? f_last : (f_last - fstar),
        )

        status = r.solved ? "solved" : "NOT solved"
        line = string(rpad(name, 22), status,
                      "  iters=", r.k,
                      "  time=", round(r.total_time, digits=4), "s",
                      "  f_last=", round(f_last, digits=4),
                      "  gap_last=", round(gap_last, digits=4))
        println(line)
        push!(report_lines, "instance=$(idx)  seed=$(s)  " * line)
    end
end

# --------------------------------------------------------------------
# 4b. Save per-iteration histories: one .mat per method
# --------------------------------------------------------------------
for name in method_order
    fname = joinpath(output_dir, "$(problem_name)_iterations_$(matvarname(name)).mat")
    matwrite(fname, Dict(
        "problem" => problem_name,
        "method"  => name,
        "seeds"   => collect(seeds),
        "runs"    => per_iter[name],
    ))
    println("✅ Saved per-iteration history: ", basename(fname))
end

# plain-text report
open(joinpath(output_dir, "$(problem_name)_report.txt"), "w") do io
    println(io, "Problem: ", problem_name, "   seeds: ", collect(seeds))
    println(io, "m_sub: $m_sub, n_sub: $n_sub, tau: $tau_val")
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
    println("    last f: mean=", round(mean(fv), digits=6),
            "   last gap: mean=", round(mean(gpv), sigdigits=4))
end

matwrite(joinpath(output_dir, "benchmark_all_methods.mat"), Dict(
    "seeds"  => collect(seeds),
    "m_sub"  => m_sub,
    "n_sub"  => n_sub,
    "tau"    => tau_val,
    "times"  => Dict(matvarname(name) => all_times[name]  for name in method_order),
    "solved" => Dict(matvarname(name) => all_solved[name] for name in method_order),
))

# --------------------------------------------------------------------
# 6. Dolan-Moré performance profile (ratio-based x-axis)
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
        #legend = :right, 
        #legend = :bottom, 
        legend = :bottomright,
        xlabel = "Performance Ratio",
        ylabel = "Fraction of Problems Solved",
        xformatter = :plain,
        #background_color_legend = RGBA(1, 1, 1, 0.6),   # semi-transparent white (icy look)
        #foreground_color_legend = RGBA(0.6, 0.6, 0.6, 0.8),  # subtle light-gray border
        style...)

plot!(p,
    xtickfont = Plots.font(20, "serif"),
    ytickfont = Plots.font(20, "serif"),
    guidefont = Plots.font(20, "serif"),
    ylims  = (-0.03, 1.03),
    xrotation = 20,
)

# Fixed name -> color mapping (matches the Lasso figure's assignment)
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

Plots.savefig(p, joinpath(output_dir, "Sigmoid_w8a_nonconvex_performance_profile_all_methods.eps"))
println("✅ Saved: Sigmoid_w8a_nonconvex_performance_profile_all_methods.eps in ", output_dir)
display(p)