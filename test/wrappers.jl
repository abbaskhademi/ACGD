# Public API coverage without executing benchmark entrypoints or loading datasets.
using LinearAlgebra, Test
root = normpath(joinpath(@__DIR__, ".."))
for filename in filter(p -> endswith(p, ".jl"), readdir(root))
    sandbox = Module(gensym(:WrapperSmoke))
    Core.eval(sandbox, :(using LinearAlgebra, Random))
    Base.include(sandbox, joinpath(root, "src", "ACGDSolvers.jl"))
    ast = Meta.parseall(read(joinpath(root, filename), String))
    names = Symbol[]
    for expression in ast.args
        if expression isa Expr && expression.head == :function
            signature = expression.args[1]
            name = signature.args[1]
            if name isa Symbol && (startswith(String(name), "conditional_gradient_") || name == :adabb)
                Core.eval(sandbox, expression)
                push!(names, name)
            end
        end
    end
    f(x) = sum(abs2, x) / 2
    g(x) = copy(x)
    constrained = !(filename in ["Least_Squares.jl", "Power_Shifted_Quadratic.jl", "Logistic_Regression_with_Regularization.jl"])
    oracle(x) = constrained ? -3sign.(x) : -x / norm(x)
    for name in names
        fun = getfield(sandbox, name)
        method = String(name)
        if endswith(method, "_matfac")
            fm(U,V) = sum(abs2,U)/2 + sum(abs2,V)/2
            gm(U,V) = (copy(U),copy(V))
            lm(U,V) = (-3sign.(U),-3sign.(V))
            args = Any[fm,gm,lm,ones(1,1),ones(1,1)]
        elseif name == :adabb
            args = Any[f,g,[1.0,2.0],0.1]
        else
            args = Any[f,g,oracle,[1.0,2.0]]
            !constrained && push!(args,1.0)
        end
        occursin("adaptive",method) || occursin("adjustable",method) ? push!(args,0.25) : nothing
        occursin("short_step",method) && push!(args,1.0)
        result = fun(args...; epsilon=1e-6,max_iter_cap=40)
        @test all(isfinite,result.values)
        @test result.k <= 40
        @test length(result.values) == length(result.gaps) == length(result.iterate_indices)
        @test result.total_time >= 0
    end
    println(filename, ": ", length(names), " wrappers passed")
end
