# ==========================================================================
# Logistic_Regression_with_Regularization Frank-Wolfe: all 5 methods x N seeds -> combined performance profile
# ==========================================================================
# "Solved" = EITHER of the method's own convergence criteria fires:
#     (a) gradient norm <= epsilon             (grad_norm <= epsilon)
# ==========================================================================

using LinearAlgebra      # norm, dot, opnorm
using Statistics         # mean, median
using Random             # seed!, randperm, randn
using Plots              # plot!, savefig, display
using Measures           # mm (margin)
using PyPlot             # pyplot() backend + PyCall access for warnings filter
using MAT                # matwrite
using BenchmarkProfiles  # performance_profile
using LIBSVMdata         # load_dataset
using Zygote             # gradient

pyplot()
PyPlot.PyCall.pyimport("warnings").filterwarnings("ignore")

# --- output goes into <run dir>/Logistic_Regression_with_Regularization/ -----------------------
problem_name = "Logistic_Regression_with_Regularization"
output_dir   = joinpath(@__DIR__, problem_name)
mkpath(output_dir)

# sanitize method names into valid .mat struct field names
matvarname(s) = replace(s, r"[^A-Za-z0-9_]" => "_")

# --------------------------------------------------------------------
# 1. Problem generator — one seed -> one independent Logistic_Regression_with_Regularization instance
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# 1. Load the FULL a9a dataset ONCE (outside the seed loop)
# --------------------------------------------------------------------
Random.seed!(23)

println("Loading a9a dataset using LIBSVMdata...")
A_full, y_full = load_dataset("a9a",
#A_full, y_full = load_dataset("rcv1_train.binary",
    dense   = false,
    replace = false,
    verbose = true,
)


m_total = size(A_full, 1)
n_total = size(A_full, 2)
# --------------------------------------------------------------------
# 2. Problem generator — subsamples rows/cols to build one benchmark instance
# --------------------------------------------------------------------
function generate_problem(seed::Int; m_sub::Int=2000, n_sub::Int=123, tau::Float64=50.0)
    @assert m_total >= m_sub "m_total=$m_total < requested m_sub=$m_sub"
    @assert n_total >= n_sub "n_total=$n_total < requested n_sub=$n_sub"

    Random.seed!(seed)

    row_indices = randperm(m_total)[1:m_sub]
    col_indices = randperm(n_total)[1:n_sub]

    A = A_full[row_indices, col_indices]
    y = y_full[row_indices]

    m, n = size(A)  # This already captures m = size(A, 1)
 
    # Regularization parameter
    omega = 1/(n*m)  # Common choice; adjust as needed

    # Define regularized logistic loss function:
    # f(x) = (1/m) * sum(log(1 + exp(-y_i * a_i^T x))) + omega * ||x||_2^2
    f(x) = mean(log.(1 .+ exp.(-y .* (A * x)))) + omega * norm(x, 2)^2
    
    # Gradient via automatic differentiation
    grad_f(x) = Zygote.gradient(f, x)[1]

    optimal_value = NaN
    rho = 1.0
    function lmo(g)
        ng = norm(g)
        return ng > 0 ? -rho * g / ng : zeros(length(g))
    end

    x0 = zeros(n)

    # FIX: Added `m` and `omega` to the returned NamedTuple
    return (f=f, grad_f=grad_f, lmo=lmo, x0=x0, rho=rho, A=A, y=y,
            seed=seed, optimal_value=optimal_value, m=m, omega=omega)
end

# Lipschitz constant of x -> ||Ax-b||^2 is 2*sigma_max(A)^2
# (only needed if a fixed-step / short-step variant is re-enabled)
# estimate_L(A) =  opnorm(Matrix(A))^2 / (4 * m) + 2 * omega
# ∇²f = (1/m) Aᵀ diag(σ(1-σ)) A + 2ωI,  σ(1-σ) ≤ 1/4
estimate_L(A, m, omega) = opnorm(Matrix(A))^2 / (4m) + 2*omega
# --------------------------------------------------------------------
# 2. Solvers — all `while true`, all return (..., solved, total_time)
#    All accept `fstar` (true optimal objective value) and stop when
#    grad_norm <= epsilon  OR  current_f - fstar <= epsilon.
# --------------------------------------------------------------------
function conditional_gradient_adaptive(f, grad_f, lmo, x0, rho, gamma;
        epsilon=1e-3, delta=1e-10, beta=2, max_iter_cap=100000, max_time=500.0,
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
        if grad_norm <= epsilon #|| (current_f - fstar) <= epsilon
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

function conditional_gradient_adjustable_scaling(f, grad_f, lmo, x0, rho, gamma;
        epsilon=1e-3, delta=1e-10, beta=2, max_iter_cap=100000, max_time=500.0,
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
        if grad_norm <= epsilon #|| (current_f - fstar) <= epsilon
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

# --------------------------------------------------------------------
# Same fix as in matrix_balancing: `f_new` is pre-declared *before* the
# inner backtracking `while` loop so it survives past it, and the
# post-step objective is what gets logged.
# --------------------------------------------------------------------
function conditional_gradient_pure_backtracking(f, grad_f, lmo, x0, rho;
        epsilon=1e-3, max_iter_cap=100000, max_time=500.0, fstar=0.0)
    x_prev = copy(x0)
    values = [f(x_prev)]; times = [0.0]; gaps = Float64[]; L_ks = Float64[]
    k = 0; solved = false
    tau_bt = 2.0; eta = 0.9
    t_start_total = time()

    Random.seed!(23)
    d0 = randn(length(x0))
    L_minus1 = norm(grad_f(x0) - grad_f(x0 + 1e-3 * d0)) / (1e-3 * norm(d0))
    M = L_minus1 * eta

    f_prev = f(x_prev)                 # carried, not recomputed each sweep
    while true
        start = time()
        grad = grad_f(x_prev)
        v = lmo(grad)
        grad_norm = norm(grad)

        t_k = grad_norm / (M * rho)
        x_new = x_prev
        f_new = f_prev                 # pre-declared so it survives the inner loop
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
        f_prev = f_new                 # no extra f evaluation
        k += 1
        push!(L_ks, M)
        M = M * eta
        push!(times, time() - start)
        push!(gaps, grad_norm); push!(values, f_prev); 
        
        if grad_norm <= epsilon #|| (f_prev - fstar) <= epsilon
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

function conditional_gradient_short_step(f, grad_f, lmo, x0, rho, L;
        epsilon=1e-3, max_iter_cap=100000, max_time=500.0, fstar=0.0)
    x_curr = copy(x0)
    values = [f(x_curr)]; times = [0.0]; gaps = Float64[]; L_ks = Float64[]
    k = 0; solved = false
    current_f = f(x_curr)

    while true
        start = time()
        current_grad = grad_f(x_curr)
        v = lmo(current_grad)
        grad_norm = norm(current_grad)

        t_k = grad_norm / (rho * L)
        x_curr = x_curr + t_k * v
        k += 1
        push!(times, time() - start)
        current_f = f(x_curr)
        push!(gaps, grad_norm); push!(values, current_f); push!(L_ks, L)
        if grad_norm <= epsilon #|| (current_f - fstar) <= epsilon
            solved = true
            break
        end
        if k >= max_iter_cap
            solved = false
            break
        end
    end
    return (x=x_curr, values=values, times=times, gaps=gaps, L_ks=L_ks,
            k=k, solved=solved, total_time=sum(times))
end


function conditional_gradient_auto_conditioned(f, grad_f, lmo, x0, rho;
        epsilon=1e-3, delta=1e-10, max_iter_cap=100000, max_time=500.0,
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
            push!(L_history, L_k_realized)
        else
            push!(L_history, hat_L_k)
        end

        x_curr = x_new
        k += 1
        push!(times, time() - start)
        current_f = new_f
        push!(gaps, grad_norm); push!(values, current_f); push!(L_ks, hat_L_k)
        if grad_norm <= epsilon #|| (current_f - fstar) <= epsilon
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

function adabb(f, grad_f, x0, alpha0;
        epsilon=1e-3, max_iter_cap=100000, max_time=500.0, fstar=0.0)
    x_prev = copy(x0)
    grad_prev = grad_f(x_prev)
    gap = norm(grad_prev)
    x = x_prev - alpha0 * grad_prev

    grad_curr = grad_f(x)
    s_1 = x - x_prev; y_1 = grad_curr - grad_prev
    lambda_1 = dot(y_1, s_1) / dot(y_1, y_1)
    
    # Compute theta_0 based on lambda_1
    if lambda_1 >= sqrt(2) * alpha0
        theta0=lambda_1^2 / (2 * alpha0^2) - 1.0
    else
        theta0 = 0.0
    end

    alpha_km1 = alpha0; theta_km1 = theta0
    values = [f(x_prev), f(x)]; times = [0.0]; gaps = Float64[]
    k = 1; solved = false
    t_start_total = time()

    while true
        start = time()
        if k > 1
            grad_curr = grad_f(x)
        end

        s_k = x - x_prev             # x^k - x^{k-1}
        y_k = grad_curr - grad_prev  # grad f(x^k) - grad f(x^{k-1})
        # Compute lambda_k (BB step size)
        lambda_k = dot(y_k, s_k) / dot(y_k, y_k)
        

        # Adaptive step-size selection
        if lambda_k >= alpha_km1
            alpha_k = sqrt(1 + theta_km1) * alpha_km1
            theta_k = alpha_k / alpha_km1
        elseif alpha_km1 / 2 < lambda_k < alpha_km1
            # Case (i)
            alpha_k = lambda_k
            theta_k = 2 * alpha_k / alpha_km1 - alpha_k / lambda_k
        else
            # Case (iii)
            alpha_k = lambda_k / sqrt(2)
            theta_k = alpha_k / alpha_km1
        end
        if k==1
            theta_k=1.0
        end
        # Update iterates
        x_prev = copy(x)
        grad_prev = copy(grad_curr)
        x = x - alpha_k * grad_curr
        k += 1
        # Update for next iteration
        alpha_km1 = alpha_k
        theta_km1 = theta_k
        push!(times, time() - start)
        f_new = f(x); gap = norm(grad_prev)
        push!(gaps, gap); push!(values, f_new)
        if gap <= epsilon #|| (f_new - fstar) <= epsilon
            solved = true
            break
        end
        if k >= max_iter_cap # || (time() - t_start_total) > max_time
            solved = false
            break
        end
    end
    return (x=x, values=values, times=times, gaps=gaps,
            k=k, solved=solved, total_time=sum(times))
end

# --------------------------------------------------------------------
# 3. Run all methods on each of the N seeded problems
# --------------------------------------------------------------------
println("\n================ Logistic_Regression_with_Regularization ================")
seeds  = 1:100

gamma0 = 1/4
alpha0 = 1e-10




method_order = ["Adaptive constant", "Adaptive adjustable", "Pure backtracking",
                "Auto-conditioned",
                 "Short-step", 
                 "AdaBB"]

all_times  = Dict(name => fill(Inf, length(seeds)) for name in method_order)
all_solved = Dict(name => falses(length(seeds)) for name in method_order)

# base Julia Dict (NOT Dictionaries.jl's `Dictionary`)
per_iter     = Dict(name => Dict{String,Any}() for name in method_order)
report_lines = String[]                      # text report accumulator

for (idx, s) in enumerate(seeds)
    println("\n================ instance = $idx | seed = $s ================")
    prob  = generate_problem(s)
     #L_est = estimate_L(prob.A)   # only needed if a short-step variant is re-enabled
     L_est = estimate_L(prob.A, prob.m, prob.omega)
     #alpha0 = 1/L_est
     #println("\n alpha0= $alpha0")
    local fstar = prob.optimal_value   # true LS optimum (noise present), NOT 0

    runs = Dict(
        "AdaBB"               => adabb(prob.f, prob.grad_f, prob.x0, alpha0; fstar=fstar),
        "Auto-conditioned"    => conditional_gradient_auto_conditioned(prob.f, prob.grad_f, prob.lmo, prob.x0, prob.rho; fstar=fstar),
        "Adaptive constant"   => conditional_gradient_adaptive(prob.f, prob.grad_f, prob.lmo, prob.x0, prob.rho, gamma0; fstar=fstar),
        "Adaptive adjustable" => conditional_gradient_adjustable_scaling(prob.f, prob.grad_f, prob.lmo, prob.x0, prob.rho, gamma0; fstar=fstar),
        "Pure backtracking"   => conditional_gradient_pure_backtracking(prob.f, prob.grad_f, prob.lmo, prob.x0, prob.rho; fstar=fstar),
        "Short-step"          => conditional_gradient_short_step(prob.f, prob.grad_f, prob.lmo, prob.x0, prob.rho, L_est; fstar=fstar),
    )

    for name in method_order
        r = runs[name]
        all_solved[name][idx] = r.solved
        all_times[name][idx]  = r.solved ? r.total_time : Inf

        # AdaBB has no L_k sequence -> store empty vector
        Lk = hasproperty(r, :L_ks) ? collect(Float64, r.L_ks) : Float64[]

        f_last   = isempty(r.values) ? NaN : Float64(last(r.values))
        gap_last = isempty(r.gaps)   ? NaN : Float64(last(r.gaps))

        per_iter[name]["seed_$(s)"] = Dict(
            "instance"     => idx,
            "seed"         => s,
            "values"       => collect(Float64, r.values),   # f per iteration
            "gaps"         => collect(Float64, r.gaps),     # grad norm per iteration
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
                      "  gradnorm_last=", round(gap_last, sigdigits=4))
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

# plain-text report (per instance/seed: status, iters, time, last f, last grad norm)
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
            "   last grad norm: mean=", round(mean(gpv), sigdigits=4))
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
# Color mapping keyed by METHOD NAME, not by position in `method_order`.
# A positional color list silently shifts every color whenever a method is
# added/dropped/reordered (e.g. "Pure backtracking" rendering black instead
# of red), breaking visual consistency with the companion matrix_balancing
# figure. This dict is stable for any subset or ordering.
# --------------------------------------------------------------------
method_colors = Dict(
    "Adaptive constant"   => :blue,
    "Adaptive adjustable" => :green,
    "Pure backtracking"   => :red,
    "Auto-conditioned"    => :black,
    "Short-step"          => :orange,
    "Open-loop"           => :purple,
    "AdaBB"               => :olive,
)
for (i, series) in enumerate(p.series_list)
    series[:linecolor] = method_colors[method_order[i]]
end

Plots.savefig(p, joinpath(output_dir,
    "Logistic_Regression_with_Regularization_performance_profile_all_methods.eps"))
println("\n✅ Saved performance profile")
display(p)