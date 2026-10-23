"Load only top-level function definitions; never execute downloads, plotting or sweeps."
function load_problem_definitions(path)
    parsed = Meta.parseall(read(path, String); filename=path)
    function syntax_error(expression)
        expression isa Expr || return false
        expression.head in (:error, :incomplete) && return true
        return any(syntax_error, expression.args)
    end
    syntax_error(parsed) && error("Julia syntax error in $path")
    target = Module(gensym(:ProblemDefinitions))
    Core.eval(target, :(using LinearAlgebra, Random, Statistics, SparseArrays))
    Core.eval(target, :(const ACGDSolvers = $ACGDSolvers))
    for expression in parsed.args
        expression isa Expr || continue
        long_function = expression.head == :function
        short_function = expression.head == :(=) && expression.args[1] isa Expr &&
                         expression.args[1].head == :call
        (long_function || short_function) && Core.eval(target, expression)
    end
    return target
end
