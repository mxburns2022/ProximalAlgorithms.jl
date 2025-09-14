using Random
using LinearAlgebra
using JuMP
using ProximalAlgorithms
using ProximalOperators
import SparseArrays as sp



struct BilinearSPP{TR,TA}
    A::TA
    N::Int
    M::Int
    λy::TR
    λx::TR
end




function subgradient!(game::BilinearSPP, g::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}) where T<:Real
    g .= game.A' * y
    maxargs = findall(x .== maximum(x))
    g[maxargs] .+= game.λx / size(maxargs, 1)
end

function supergradient!(game::BilinearSPP, g::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}) where T<:Real
    g .= game.A * x
    maxargs = findall(y .== maximum(y))
    g[maxargs] .-= game.λy / size(maxargs, 1)
end

function eval(game::BilinearSPP, x::AbstractVector{T}, y::AbstractVector{T}) where T<:Real
    return y' * game.A * x + game.λx * maximum(x) - game.λy * maximum(y)
end


function generate_random_zero_sum(n::Int, m::Int, density::Float64, stddev::Float64; seed::Int=0, sparse::Bool=true)
    generator = Xoshiro(seed)
    if !sparse
        A = ((randn(generator, m, n) .* stddev) .* convert.(Float64, rand(generator, m, n) .<= density))
    else
        rfn(gen, k) = randn(gen, k) .* stddev
        A = sp.sprand(generator, m, n, density, rfn)
    end

    return BilinearSPP(A, n, m, 1.0, 1.0)
end


function prox_mapping_y!(::BilinearSPP, z::AbstractVector{T}) where T<:Real
    yt = sort(z)
    n = size(z, 1)
    for i in 1:n-1
        t = (sum(yt[i+1:n]) - 1) / (n - i)
        if t >= yt[i]
            z .= max.(z .- t, 0)
            return
        end
    end
    t = (sum(z) - 1) / (n)
    z .= max.(z .- t, 0)
end
prox_mapping_x!(game::BilinearSPP, x) = prox_mapping_y!(game, x)


function primalv(game::BilinearSPP, x::AbstractVector{T}; solution::Bool=false) where T<:Real
    Ax = -game.A * x
    indices = sortperm(Ax)
    sorted_Ax = Ax[indices]
    index = 1
    value = sorted_Ax[1] + game.λy
    running_sum = sorted_Ax[1]

    for i in 2:game.M
        running_sum += sorted_Ax[i]
        newval = running_sum / i + game.λy / i
        if newval > value
            break
        end
        index = i
        value = newval
    end
    if !solution
        return -value + maximum(x)
    end
    optimizer = zeros(game.N)
    optimizer[indices[1:index]] .= 1 / index
    return -value + maximum(x) * game.λx, optimizer
end
function dualv(game::BilinearSPP, y::AbstractVector{T}; solution::Bool=false) where T<:Real
    Ay = game.A' * y
    indices = sortperm(Ay)
    sorted_Ay = Ay[indices]
    index = 1
    value = sorted_Ay[1] + game.λx
    running_sum = sorted_Ay[1]

    for i in 2:game.N
        running_sum += sorted_Ay[i]
        newval = running_sum / i + game.λx / i
        if newval > value
            break
        end
        index = i
        value = newval
    end
    if !solution
        return value - maximum(y)
    end
    println(sorted_Ay[index+1] / index, " ", (game.λx + 1) / index)
    optimizer = zeros(game.N)
    optimizer[indices[1:index]] .= 1 / index
    return value - maximum(y) * game.λy, optimizer
end
function subgradient_method(problem::BilinearSPP, target_accuracy::Float64; log_frequency::Int=1000)
    gx = zeros(problem.N)
    gy = zeros(problem.M)
    x = ones(problem.N) / problem.N
    y = ones(problem.M) / problem.M
    M = norm(problem.A, Inf) + 1
    h = IndSimplex()
    niter = convert(Int, ceil(128 * M^2 * 1 / (target_accuracy^2)))
    hₖ = target_accuracy / (32 * M^2)
    x̄ = zeros(problem.N)
    ȳ = zeros(problem.M)
    starttime = time_ns()
    # data = []
    println("elapsed_time(s),inner_steps,outer_steps,pd_gap")

    for k in 1:niter
        axpby!(1 / k, x, (k - 1) / k, x̄)
        axpby!(1 / k, y, (k - 1) / k, ȳ)
        if (k - 1) % log_frequency == 0
            println(round((time_ns() - starttime) / 1e9, sigdigits=6), ",", 2(k - 1), ",", k, ",", primalv(game, x̄) - dualv(game, ȳ))
            flush(stdout)
        end
        if k % 500 == 0 && primalv(problem, x̄) - dualv(problem, ȳ) <= target_accuracy
            return x, y
        end

        subgradient!(problem, gx, x, y)
        supergradient!(problem, gy, x, y)
        axpby!(-hₖ, gx, 1.0, x)
        axpby!(hₖ, gy, 1.0, y)

        prox!(x, h, x)
        prox!(y, h, y)
    end
    return x, y
end


