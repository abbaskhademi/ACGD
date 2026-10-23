module ACGDSolvers

using LinearAlgebra, Random
export solve, result_metadata, update_scaling

# Arrays and tuples of arrays use the same Euclidean/product Frobenius geometry.
_copy(x) = copy(x)
_copy(x::Tuple) = map(_copy, x)
_zero(x) = zero.(x)
_zero(x::Tuple) = map(_zero, x)
_sub(x, y) = x - y
_sub(x::Tuple, y::Tuple) = map(_sub, x, y)
_add(x, t, d) = x + t * d
_add(x::Tuple, t, d::Tuple) = map((a, b) -> _add(a, t, b), x, d)
_dot(x, y) = dot(x, y)
_dot(x::Tuple, y::Tuple) = sum(map(_dot, x, y))
_norm(x) = norm(x)
_norm(x::Tuple) = hypot(map(_norm, x)...)
_finite(x) = all(isfinite, x)
_finite(x::Tuple) = all(_finite, x)
_randn(rng, x) = randn(rng, size(x))
_randn(rng, x::Tuple) = map(a -> _randn(rng, a), x)
_seconds(t) = (time_ns() - t) / 1e9

"Update gamma after exactly ten completed iterations, counting rejections once."
function update_scaling(gamma, rejected, completed, block_total)
    block_total += rejected
    if completed % 10 == 0
        gamma = block_total == 0 ? 0.9gamma : block_total > 10 ? min(1.0, 1.1gamma) : gamma
        block_total = 0
    end
    return gamma, block_total
end

function _criterion(value, gap, epsilon, fstar, rule)
    if rule != :stationarity && isfinite(fstar) && value - fstar <= epsilon
        return "objective", value - fstar
    elseif rule != :objective && isfinite(gap) && gap <= epsilon
        return "stationarity", gap
    end
    return "none", rule == :objective ? value - fstar : gap
end

"""
    solve(f, grad_f, lmo, x0; method, constrained, stop_rule, ...)

Common timing and stopping implementation. `f`, `grad_f`, and `lmo` take one
state (possibly a tuple). Unconstrained LMOs must be Euclidean ball oracles.
`values`, `gaps`, and `iterate_indices` refer to the SAME iterate, including x0.
`L_ks` holds accepted curvature, before eta decay. `times[1]` is initialization;
later entries are loop attempts (including a failed final attempt, if any).
`total_time == elapsed_time` includes initialization, probes, stopping and in-memory
logging; `iteration_time` excludes initialization. No file or console I/O occurs.
"""
function solve(f, grad_f, lmo, x0; method=:adaptive, constrained=true,
        stop_rule=:either, epsilon=1e-5, fstar=NaN, gamma=0.25,
        delta=1e-10, beta=2.0, eta=0.9, L=nothing, alpha0=1e-10,
        max_iter_cap=100000, max_time=Inf, max_backtracks=100, probe_seed=23,
        open_schedule=:convex)
    started = time_ns()
    method in (:adaptive, :adjustable, :backtracking, :auto, :short, :open, :adabb) ||
        throw(ArgumentError("unknown method: $method"))
    stop_rule in (:either, :objective, :stationarity) || throw(ArgumentError("invalid stop_rule"))
    isfinite(epsilon) && epsilon >= 0 || throw(ArgumentError("epsilon must be finite and nonnegative"))
    isfinite(delta) && delta > 0 || throw(ArgumentError("delta must be finite and positive"))
    isfinite(gamma) && 0 < gamma <= 1 || throw(ArgumentError("gamma must be in (0,1]"))
    isfinite(beta) && beta > 1 || throw(ArgumentError("beta must exceed one"))
    isfinite(eta) && 0 < eta <= 1 || throw(ArgumentError("eta must be in (0,1]"))
    max_iter_cap >= 0 && max_backtracks >= 0 && max_time >= 0 || throw(ArgumentError("negative limit"))
    method == :short && !(L isa Real && isfinite(L) && L > 0) && throw(ArgumentError("short step requires positive L"))
    method == :adabb && constrained && throw(ArgumentError("AdaBB is unconstrained"))
    method == :adabb && !(isfinite(alpha0) && alpha0 > 0) && throw(ArgumentError("invalid alpha0"))
    stop_rule == :objective && !isfinite(fstar) && throw(ArgumentError("objective stopping requires finite fstar"))
    open_schedule in (:convex, :nonconvex) || throw(ArgumentError("invalid open_schedule"))

    objective_evaluations = 0
    function objective(z)
        objective_evaluations += 1
        return Float64(f(z))
    end
    function direction(z, g)
        method == :adabb && return g, _norm(g) # direction unused by AdaBB
        # A stationary point must not call an oracle that divides by zero / SVD(0).
        if _norm(g) == 0
            return _sub(z, z), 0.0
        end
        v = lmo(g)
        d = constrained ? _sub(v, z) : v
        gap = constrained ? -_dot(g, d) : _norm(g)
        return d, gap
    end

    x = _copy(x0)
    value = _finite(x) ? objective(x) : NaN
    g = isfinite(value) ? grad_f(x) : _zero(x)
    d, gap = isfinite(value) && _finite(g) ? direction(x, g) : (_zero(x), NaN)
    values = Float64[value]; gaps = Float64[gap]; iterate_indices = Int[0]
    times = Float64[]; L_ks = Float64[]; backtracks = Int[]
    gamma_history = Float64[gamma]; gamma_iterations = Int[0]
    k = 0; block_total = 0; failed_backtracks = 0
    criterion, residual = _criterion(value, gap, epsilon, fstar, stop_rule)
    valid = isfinite(value) && _finite(x) && _finite(g) && _finite(d) && isfinite(gap)
    status = !valid ? "nonfinite" : gap < -epsilon ? "invalid_direction" : criterion != "none" ? "converged" : "running"
    curvature = method == :short ? Float64(L) : delta
    prev_x = x; prev_g = g
    alpha = Float64(alpha0); theta = 0.0

    # All methods' initialization and first curvature probes share this boundary.
    if status == "running" && k < max_iter_cap && _seconds(started) < max_time &&
            method in (:adaptive, :adjustable, :backtracking, :auto)
        # Julia 1.11's seeded TaskLocalRNG uses the same stream as Xoshiro;
        # preserve the original probe while leaving the caller's RNG untouched.
        rng = Xoshiro(probe_seed)
        probe = _randn(rng, x)
        probe_norm = _norm(probe)
        if probe_norm == 0
            status = "zero_probe"
        else
            gp = grad_f(_add(x, 1e-3, probe))
            estimate = _finite(gp) ? _norm(_sub(gp, g)) / (1e-3 * probe_norm) : NaN
            if !isfinite(estimate)
                status = "nonfinite_probe"
            else
                curvature = method in (:adaptive, :adjustable) ? gamma * (estimate + delta) :
                    method == :backtracking ? max(delta, eta * estimate) : estimate + delta
            end
        end
    end
    initialization_time = _seconds(started)
    push!(times, initialization_time)

    while status == "running"
        attempt_started = time_ns()
        if k >= max_iter_cap
            status = "iteration_limit"
        elseif _seconds(started) >= max_time
            status = "time_limit"
        elseif (method != :adabb && _norm(d) == 0) || gap == 0
            status = "stationary_without_target"
        end
        if status != "running"
            push!(times, _seconds(attempt_started))
            break
        end

        if method in (:adaptive, :adjustable) && k > 0
            displacement = _norm(_sub(x, prev_x))
            if displacement == 0
                status = "stagnation"
                push!(times, _seconds(attempt_started))
                break
            end
            curvature = gamma * (_norm(_sub(g, prev_g)) / displacement + delta)
        elseif method == :adabb && k > 0
            s = _sub(x, prev_x); y = _sub(g, prev_g)
            yy = _dot(y, y)
            lambda = yy > 0 ? _dot(y, s) / yy : NaN
            # AdaBB is specified for convex objectives. Do not silently replace
            # its formula with an invented nonconvex fallback or an ascent step.
            if !(isfinite(lambda) && lambda > 0)
                status = yy == 0 ? "adabb_zero_curvature" : "adabb_nonpositive_curvature"
                push!(times, _seconds(attempt_started))
                break
            end
            if k == 1
                theta = lambda >= sqrt(2) * alpha ? lambda^2 / (2alpha^2) - 1 : 0.0
            end
            next_alpha = lambda >= alpha ? sqrt(1 + theta) * alpha :
                lambda > alpha / 2 ? lambda : lambda / sqrt(2)
            next_theta = alpha / 2 < lambda < alpha ? 2next_alpha / alpha - next_alpha / lambda : next_alpha / alpha
            theta = k == 1 ? 1.0 : next_theta # authors' numerical initialization
            alpha = next_alpha
        end

        normd2 = method == :adabb ? 0.0 : _dot(d, d)
        directional_gap = method == :adabb ? 0.0 : -_dot(g, d)
        rejected = 0
        new_x = x; new_value = value
        accepted = false
        while true
            if _seconds(started) >= max_time
                status = "time_limit"
                break
            end
            if method == :adabb
                step = alpha
                new_x = _add(x, -step, g)
            else
                if method == :open
                    step = open_schedule == :convex ? 2 / (k + 2) : 1 / sqrt(k + 1)
                else
                    if !(isfinite(curvature) && curvature > 0)
                        status = "invalid_curvature"
                        break
                    end
                    step = directional_gap / (curvature * normd2)
                    constrained && (step = min(step, 1.0))
                end
                new_x = _add(x, step, d)
            end
            if !isfinite(step) || step <= 0
                status = "invalid_step"
                break
            elseif _finite(new_x) && _norm(_sub(new_x, x)) == 0
                status = "stagnation"
                break
            end
            new_value = _finite(new_x) ? objective(new_x) : Inf
            is_backtracking = method in (:adaptive, :adjustable, :backtracking)
            # A few objective ulps allow equality at floating-point resolution;
            # the allowance is shared by both backtracking families.
            allowance = 8eps(Float64) * max(1.0, abs(value), abs(new_value))
            accepted = isfinite(new_value) && (!is_backtracking ||
                value - new_value + allowance >= step * directional_gap - curvature * normd2 * step^2 / 2)
            accepted && break
            if !is_backtracking
                status = "nonfinite"
                break
            end
            rejected += 1
            if rejected > max_backtracks
                status = "backtracking_limit"
                break
            end
            curvature *= beta
        end
        if !accepted
            failed_backtracks = rejected
            push!(times, _seconds(attempt_started))
            break
        end

        if method == :auto
            s = _sub(new_x, x)
            realized = 2 * (new_value - value - _dot(g, s)) / _dot(s, s)
            next_curvature = max(curvature, realized) # O(1), same running maximum
        else
            next_curvature = curvature
        end
        prev_x = x; prev_g = g
        x = new_x; value = new_value # accepted objective cached for every method
        k += 1
        method != :open && push!(L_ks, method == :adabb ? 1 / alpha : curvature)
        push!(backtracks, rejected)
        if method == :adjustable
            gamma, block_total = update_scaling(gamma, rejected, k, block_total)
            if k % 10 == 0
                push!(gamma_history, gamma); push!(gamma_iterations, k)
            end
        end
        curvature = method == :backtracking ? max(delta, curvature * eta) : next_curvature
        g = grad_f(x)
        d, gap = _finite(g) ? direction(x, g) : (_sub(x, x), NaN)
        push!(values, value); push!(gaps, gap); push!(iterate_indices, k)
        criterion, residual = _criterion(value, gap, epsilon, fstar, stop_rule)
        if !(_finite(g) && _finite(d) && isfinite(gap))
            status = "nonfinite"
        elseif gap < -epsilon
            status = "invalid_direction"
        elseif _seconds(started) >= max_time
            status = "time_limit"
        elseif criterion != "none"
            status = "converged"
        elseif k >= max_iter_cap
            status = "iteration_limit"
        end
        push!(times, _seconds(attempt_started))
    end

    iteration_time = sum(@view times[2:end])
    elapsed_time = _seconds(started)
    if status == "converged" && elapsed_time > max_time
        status = "time_limit"
    end
    return (x=x, values=values, gaps=gaps, iterate_indices=iterate_indices,
        times=times, L_ks=L_ks, backtracks=backtracks, failed_backtracks=failed_backtracks,
        gamma_history=gamma_history, gamma_iterations=gamma_iterations,
        k=k, solved=status == "converged", status=status,
        total_time=elapsed_time, elapsed_time=elapsed_time,
        initialization_time=initialization_time, iteration_time=iteration_time,
        final_value=value, final_gap=gap, final_residual=residual,
        termination_iteration=k, stopping_criterion=criterion,
        objective_evaluations=objective_evaluations)
end

"Additional fields for saved histories (all indices refer to the returned iterate)."
function result_metadata(r)
    fields = (:iterate_indices, :status, :elapsed_time, :initialization_time,
        :iteration_time, :backtracks, :failed_backtracks, :gamma_history,
        :gamma_iterations, :final_value, :final_gap, :final_residual,
        :termination_iteration, :stopping_criterion, :objective_evaluations)
    return Dict(string(key) => getproperty(r, key) for key in fields)
end

end
