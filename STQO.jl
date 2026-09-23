# ==========================================================================
# Non-Convex StQP over the Simplex: Frank-Wolfe (6 methods) x N seeds
# -> combined performance profile + per-iteration histories
# ==========================================================================
# "Solved" = Frank-Wolfe gap <= epsilon
#     (gap = -dot(grad_f(x), v - x) <= epsilon)
# Standard, fstar-free stopping criterion for non-convex FW (StQP is
# NP-hard in general; the true global optimum is not tracked here).
# ==========================================================================

using LinearAlgebra, Statistics, Random
using Plots, Measures, PyPlot, MAT
using BenchmarkProfiles

pyplot()
PyPlot.PyCall.pyimport("warnings").filterwarnings("ignore")

problem_name = "NonconvexQP_Simplex"
output_dir = joinpath(@__DIR__, problem_name)   # change to "/content/output" if in Colab
mkpath(output_dir)

matvarname(s) = replace(s, r"[^A-Za-z0-9_]" => "_")

# --------------------------------------------------------------------
# 1. Problem generator — StQP instance k (deterministic myrandom stream)
# --------------------------------------------------------------------
function myrandom(r, a, b)
    r = mod(r * 41475557, 1.0)
    num = a + r * (b - a)
    return r, num
end

function generate_problem(igen; n=899, dens=0.75, dvert=2.0, x0_mode=:myrandom)
    N = n + 1
    myseed = Float64(igen)
    r = (4 * myseed + 1) / 16384 / 16384

    Fc = zeros(Float64, N, N)
    Fl = zeros(Float64, N, N)

    for i in 1:n
        for j in (i+1):N
            r, num = myrandom(r, 0.0, 1.0)
            if num < dens
                r, num = myrandom(r, 0.0, 10.0)
            else
                r, num = myrandom(r, -10.0, 0.0)
            end
            Fc[i, j] = num
            Fc[j, i] = num
        end
    end

    for i in 1:N
        r, num = myrandom(r, 0.0, dvert)
        Fl[i, i] = num
    end
    for i in 1:n
        for j in (i+1):N
            Fl[i, j] = 0.5 * (Fl[i, i] + Fl[j, j])
            Fl[j, i] = Fl[i, j]
        end
    end

    Q = Fl - Fc

    f(x)      = dot(x, Q * x)
    grad_f(x) = 2.0 * (Q * x)
    function lmo(g)
        _, i = findmin(g)
        v = zeros(N); v[i] = 1.0
        return v
    end

    z = zeros(N)
    if x0_mode == :myrandom
        for i in 1:N
            r, num = myrandom(r, 0.0, 1.0)
            z[i] = num
        end
    else
        Random.seed!(igen)
        z .= rand(N)
    end
    #x0 = z / sum(z)
    x0 = zeros(N); x0[1] = 1.0

    # StQP global optimum is unknown in general (NP-hard) -> NaN, same as Sigmoid template
    optimal_value = NaN

    return (f=f, grad_f=grad_f, lmo=lmo, x0=x0, Q=Q, n=n, N=N,
            dens=dens, dvert=dvert, optimal_value=optimal_value, seed=igen)
end

function estimate_L(Q)
    evals = eigvals(Symmetric(Q))
    return 2.0 * maximum(abs.(evals))
end

# --------------------------------------------------------------------
# 2. Solvers (identical logic to your StQP draft; fstar kept for API parity)
# --------------------------------------------------------------------
function conditional_gradient_adaptive(f, grad_f, lmo, x0, gamma;
        epsilon=1e-3, delta=1e-10, beta=2, max_iter_cap=100000, max_time=500.0,
        fstar=NaN)
    x_prev = copy(x0); x_curr = copy(x0)
    values = [f(x_curr)]; times = [0.0]; gaps = Float64[]; L_ks = Float64[]
    k = 0; solved = false
    prev_grad = grad_f(x_prev); current_f = f(x_curr)

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
        
        if gap <= epsilon
           # println("f_best AC= ", minimum(values))
            solved = true
            break
        end
        if k >= max_iter_cap
           # println("f_best AC= ", minimum(values))
            solved = false
            break
        end
        current_f = f(x_curr); prev_grad = current_grad
        push!(gaps, gap); push!(values, current_f); push!(L_ks, L_k)
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
        if gap <= epsilon
           # println("f_best AA= ", minimum(values))
            solved = true
            break
        end
        if k >= max_iter_cap
           # println("f_best AA= ", minimum(values))
            solved = false
            break
        end
        current_f = f(x_curr); prev_grad = current_grad
        push!(gaps, gap); push!(values, current_f); push!(L_ks, L_k)
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
        if gap <= epsilon
           # println("f_best At= ", minimum(values))
            solved = true
            break
        end
        if k >= max_iter_cap
           # println("f_best At= ", minimum(values))
            solved = false
            break
        end
        current_f = new_f
        push!(gaps, gap); push!(values, current_f); push!(L_ks, hat_L_k)
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

        

        t_k = min(gap / (M * normd2), 1.0)
        x_new = x_prev
        while true
            x_new = x_prev + t_k * d
            f_new = f(x_new)
            if f_prev - f_new >= t_k * gap - (M / 2) * t_k^2 * normd2
                break
            else
                M *= tau_bt
                t_k = min(gap / (M * normd2), 1.0)
            end
        end

        x_prev = x_new
        k += 1
        M = M * eta
        push!(times, time() - start)
        if gap <= epsilon
           # println("f_best PB= ", minimum(values))
            solved = true
            break
        end
        if k >= max_iter_cap
           # println("f_best PB= ", minimum(values))
            solved = false
            break
        end
        push!(gaps, gap); push!(values, f(x_new)); push!(L_ks, M/eta)
        
    end
    return (x=x_prev, values=values, times=times, gaps=gaps, L_ks=L_ks,
            k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_open(f, grad_f, lmo, x0;
        epsilon=1e-3, max_iter_cap=100000, max_time=500.0, fstar=NaN)
    x = copy(x0)
    values = [f(x)]; times = [0.0]; gaps = Float64[]
    k = 0; solved = false

    while true
        start = time()
        grad = grad_f(x)
        v = lmo(grad)
        d = v - x

        t_k = 1 / sqrt(1 + k)
        x = x + t_k * d
        k += 1

        push!(times, time() - start);
        gap = -dot(grad, d)
        f_prev = f(x)
        
        if gap <= epsilon
           # println("f_best OL= ", minimum(values))
            solved = true
            break
        end
        if k >= max_iter_cap
           # println("f_best OL= ", minimum(values))
            solved = false
            break
        end
        push!(gaps, gap); push!(values, f_prev)
    end
    return (x=x, values=values, times=times, gaps=gaps, k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_short_step(f, grad_f, lmo, x0, L;
        epsilon=1e-3, max_iter_cap=100000, max_time=500.0, fstar=NaN)
    x = copy(x0)
    values = [f(x)]; times = [0.0]; gaps = Float64[]
    k = 0; solved = false

    while true
        start = time()
        grad = grad_f(x)
        v = lmo(grad)
        d = v - x
        gap = -dot(grad, d)
        

        

        t_k = min(gap / (L * norm(d)^2), 1.0)
        x = x + t_k * d
        k += 1
        push!(times, time() - start)
        f_prev = f(x)
        if gap <= epsilon
           # println("f_best SS= ", minimum(values))
            solved = true
            break
        end
        if k >= max_iter_cap
           # println("f_best SS= ", minimum(values))
            solved = false
            break
        end
        push!(gaps, gap); push!(values, f_prev)
    end
    return (x=x, values=values, times=times, gaps=gaps, k=k, solved=solved, total_time=sum(times))
end

# --------------------------------------------------------------------
# 3. Run all 6 methods on each of the N seeded StQP instances
#    (SAME output structure as the Sigmoid/Gisette template)
# --------------------------------------------------------------------
println("\n================ Simplex Quadratic Optimization Problem ================")
seeds  = 1:100
gamma0 = 1/4
n_dim  = 699          # -> N = 700
dens_val  = 0.75
dvert_val = 2.0

method_order = ["Adaptive constant", "Adaptive adjustable", "Pure backtracking",
                 "Auto-conditioned", "Short-step", "Open-loop"]

all_times  = Dict(name => fill(Inf, length(seeds)) for name in method_order)
all_solved = Dict(name => falses(length(seeds)) for name in method_order)

per_iter     = Dict(name => Dict{String,Any}() for name in method_order)
report_lines = String[]

for (idx, s) in enumerate(seeds)
    println("\n================ instance = $idx | seed = $s ================")
    prob  = generate_problem(s; n=n_dim, dens=dens_val, dvert=dvert_val)
    L_est = estimate_L(prob.Q)
    fstar = prob.optimal_value   # NaN — global StQP optimum unknown

    runs = Dict(
        "Open-loop"           => conditional_gradient_open(prob.f, prob.grad_f, prob.lmo, prob.x0; fstar=fstar),
        "Short-step"          => conditional_gradient_short_step(prob.f, prob.grad_f, prob.lmo, prob.x0, L_est; fstar=fstar),
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
    println(io, "n_dim: $n_dim (N=$(n_dim+1)), dens: $dens_val, dvert: $dvert_val")
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
    "seeds"    => collect(seeds),
    "n_dim"    => n_dim,
    "dens"     => dens_val,
    "dvert"    => dvert_val,
    "times"    => Dict(matvarname(name) => all_times[name]  for name in method_order),
    "solved"   => Dict(matvarname(name) => all_solved[name] for name in method_order),
))

# --------------------------------------------------------------------
# 5. Dolan-Moré performance profile (ratio-based x-axis)
# --------------------------------------------------------------------
T = hcat([all_times[name] for name in method_order]...)

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

# Fixed name -> color mapping (SAME palette as Sigmoid template, keyed by name
# so it's robust to any reordering of method_order)
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

Plots.savefig(p, joinpath(output_dir, "$(problem_name)_performance_profile_all_methods.eps"))
println("✅ Saved: $(problem_name)_performance_profile_all_methods.eps in ", output_dir)
display(p)