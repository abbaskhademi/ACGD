# ==========================================================================
# Video Co-Localization (aeroplane_data_small.mat, quadratic over a
# product of 33 simplices). L_k-history plot for the 3 backtracking methods:
#   Adaptive constant, Adaptive adjustable, Pure backtracking
# Stopping: FW gap <= epsilon  OR  iteration cap (f* is unknown here)
# ==========================================================================

using LinearAlgebra, Statistics, Random, Downloads
using MAT, Plots, Measures, PyPlot

pyplot()
PyPlot.PyCall.pyimport("warnings").filterwarnings("ignore")

problem_name = "video_colocalization"
output_dir = joinpath(@__DIR__, problem_name)
mkpath(output_dir)
matvarname(s) = replace(s, r"[^A-Za-z0-9_]" => "_")

# --------------------------------------------------------------------
# 1. Problem
# --------------------------------------------------------------------
Random.seed!(23)
data_file = joinpath(output_dir, "aeroplane_data_small.mat")
if !isfile(data_file)
    Downloads.download(
        "https://raw.githubusercontent.com/cyrillewcombettes/boostfw/master/aeroplane_data_small.mat",
        data_file)
    println("File downloaded successfully.")
end

data = matread(data_file)
A = data["A"]
b = [data["b"][i][1] for i in 1:size(data["b"], 1)]

f(x)      = 0.5 * dot(x, A * x) + dot(b, x)
grad_f(x) = A * x + b

L_global = opnorm(Matrix(A), 2)
println("Lipschitz constant L = ||A||_2 = $L_global")

const n_blocks   = 33
const block_size = 20
x0 = vcat([vcat([1.0], zeros(block_size - 1)) for _ in 1:n_blocks]...)

function lmo(g)                               # block-wise simplex LMO
    v = zeros(length(g))
    for i in 0:(n_blocks - 1)
        blk = @view g[block_size*i+1 : block_size*(i+1)]
        v[block_size*i + argmin(blk)] = 1.0
    end
    return v
end

# --------------------------------------------------------------------
# 2. Solvers (same structure as the core code; f* criterion dropped)
# --------------------------------------------------------------------
function conditional_gradient_adaptive(f, grad_f, lmo, x0, gamma;
        epsilon=1e-5, delta=1e-10, beta=2, max_iter_cap=1000)
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
            if current_f - f(x_new) >= t_k * gap - (Lknormd2 / 2) * t_k^2
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
        k % 100 == 0 && println("  [Adaptive const] k=$k gap=$(round(gap, sigdigits=4)) L_k=$(round(L_k, sigdigits=4)) f=$(round(current_f, digits=6))")
        if gap <= epsilon;    solved = true;  break; end
        if k >= max_iter_cap; solved = false; break; end
    end
    return (x=x_curr, values=values, times=times, gaps=gaps, L_ks=L_ks,
            k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_adjustable_scaling(f, grad_f, lmo, x0, gamma;
        epsilon=1e-5, delta=1e-10, beta=2, max_iter_cap=1000)
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
            if current_f - f(x_new) >= t_k * gap - (Lknormd2 / 2) * t_k^2
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
        k % 100 == 0 && println("  [Adaptive adj]   k=$k gap=$(round(gap, sigdigits=4)) L_k=$(round(L_k, sigdigits=4)) f=$(round(current_f, digits=6))")
        if gap <= epsilon;    solved = true;  break; end
        if k >= max_iter_cap; solved = false; break; end
    end
    return (x=x_curr, values=values, times=times, gaps=gaps, L_ks=L_ks,
            gamma_history=gamma_history, k=k, solved=solved, total_time=sum(times))
end

function conditional_gradient_pure_backtracking(f, grad_f, lmo, x0;
        epsilon=1e-5, max_iter_cap=1000)
    x_prev = copy(x0)
    values = [f(x_prev)]; times = [0.0]; gaps = Float64[]; L_ks = Float64[]
    k = 0; solved = false
    tau_bt = 2.0; eta = 0.9
    Random.seed!(23)
    d0 = randn(length(x0))
    M = eta * norm(grad_f(x0) - grad_f(x0 + 1e-3 * d0)) / (1e-3 * norm(d0))
    f_prev = f(x_prev)
    while true
        start = time()
        grad = grad_f(x_prev)
        v = lmo(grad)
        d = v - x_prev
        normd2 = dot(d, d)
        gap = -dot(grad, d)
        t_k = min(gap / (M * normd2), 1.0)
        x_new = x_prev; f_new = f_prev
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
        push!(L_ks, M)            # accepted L_k of this iteration
        x_prev = x_new; f_prev = f_new
        M *= eta
        k += 1
        push!(times, time() - start)
        push!(gaps, gap); push!(values, f_new)
        k % 100 == 0 && println("  [Pure backtrack] k=$k gap=$(round(gap, sigdigits=4)) L_k=$(round(L_ks[end], sigdigits=4)) f=$(round(f_new, digits=6))")
        if gap <= epsilon;    solved = true;  break; end
        if k >= max_iter_cap; solved = false; break; end
    end
    return (x=x_prev, values=values, times=times, gaps=gaps, L_ks=L_ks,
            k=k, solved=solved, total_time=sum(times))
end

# --------------------------------------------------------------------
# 3. Run the 3 methods
# --------------------------------------------------------------------
gamma0   = 1/4
max_iter = 500

method_order = ["Adaptive constant", "Adaptive adjustable", "Pure backtracking"]

runs = Dict{String,Any}()
println("\n>> Adaptive constant");   runs["Adaptive constant"]   = conditional_gradient_adaptive(f, grad_f, lmo, x0, gamma0; max_iter_cap=max_iter)
println("\n>> Adaptive adjustable"); runs["Adaptive adjustable"] = conditional_gradient_adjustable_scaling(f, grad_f, lmo, x0, gamma0; max_iter_cap=max_iter)
println("\n>> Pure backtracking");   runs["Pure backtracking"]   = conditional_gradient_pure_backtracking(f, grad_f, lmo, x0; max_iter_cap=max_iter)

report_lines = String[]
for name in method_order
    r = runs[name]
    line = string(rpad(name, 22), r.solved ? "solved" : "cap reached",
            "  iters=", r.k, "  time=", round(r.total_time, digits=2), "s",
            "  f_last=", round(last(r.values), digits=6),
            "  gap_last=", round(last(r.gaps), sigdigits=4),
            "  max L_k=", round(maximum(r.L_ks), sigdigits=4),
            "  mean L_k=", round(mean(r.L_ks), sigdigits=4))
    println(line); push!(report_lines, line)

    matwrite(joinpath(output_dir, "$(problem_name)_iterations_$(matvarname(name)).mat"), Dict(
        "problem" => problem_name, "method" => name,
        "values" => collect(Float64, r.values), "gaps" => collect(Float64, r.gaps),
        "times" => collect(Float64, r.times), "L_ks" => collect(Float64, r.L_ks),
        "iterations" => r.k, "solved" => r.solved ? 1 : 0,
        "total_time" => r.total_time, "L_global" => L_global,
    ))
end
open(joinpath(output_dir, "$(problem_name)_summary.txt"), "w") do io
    println(io, "Problem: ", problem_name, "   L = ||A||_2 = ", L_global)
    foreach(l -> println(io, l), report_lines)
end

# --------------------------------------------------------------------
# 4. L_k plot (EPS) — layered scatter style
# --------------------------------------------------------------------
ped_Lk  = runs["Pure backtracking"].L_ks
our_Lk  = runs["Adaptive constant"].L_ks
our5_Lk = runs["Adaptive adjustable"].L_ks

style = Dict(
    :titlefont  => (20, "serif"),
    :guidefont  => (20, "serif"),
    :tickfont   => (20, "serif"),
    :legendfont => (14, "serif"),
    :grid       => false,
    :framestyle => :box,
    :margin     => 5mm,
    :size       => (900, 700),
    :linewidth  => 2
)

p1 = Plots.plot()  # Start with empty plot

# 1. RED first — hidden from legend (drawn underneath)
Plots.plot!(p1, ped_Lk,
            label=false,
            color=:red,
            seriestype=:scatter,
            markersize=8,
            marker=:star5,
            markerstrokewidth=0)

# 2. BLUE — legend entry
Plots.plot!(p1, our_Lk,
            label="Adaptive constant",
            color=:blue,
            seriestype=:scatter,
            markersize=0.1,
            marker=:rtriangle,
            markerstrokewidth=0)

# 3. GREEN — legend entry
Plots.plot!(p1, our5_Lk,
            label="Adaptive adjustable",
            color=:green,
            seriestype=:scatter,
            markersize=0.1,
            marker=:circle,
            markerstrokewidth=0)

# 4. RED again — legend entry (appears LAST in legend)
Plots.plot!(p1, ped_Lk,
            label="Pure backtracking",
            color=:red,
            seriestype=:scatter,
            markersize=5,
            marker=:star5,
            markerstrokewidth=0)

# 5. BLUE on top — not in legend
Plots.plot!(p1, our_Lk,
            label=false,
            color=:blue,
            seriestype=:scatter,
            markersize=7,
            marker=:rtriangle,
            markerstrokewidth=0)

# 6. GREEN on top — not in legend
Plots.plot!(p1, our5_Lk,
            label=false,
            color=:green,
            seriestype=:scatter,
            markersize=5,
            marker=:circle,
            markerstrokewidth=0)

# Axis labels and style
Plots.plot!(p1,
        xlabel="Iteration",
        ylabel="Local Lipschitz Estimate";
        style...)

fname = joinpath(output_dir, "Video_L.eps")
Plots.savefig(p1, fname)
println("✅ Saved: ", fname)
display(p1)