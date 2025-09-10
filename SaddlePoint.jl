using Random
using LinearAlgebra
using JuMP
using ProximalAlgorithms
using ProximalOperators
# import ProximalAlgorithm

# include("ProxBundle.jl")
abstract type SaddlePointProblem end



# Solve \min_{x ∈ Δₙ}\max_{y ∈ Δₘ}<Ax, y> + ||Bx||∞ -||-Cy||∞
mutable struct BilinearGame <: SaddlePointProblem
    A::Matrix{Float64}
    N::Int
    M::Int
    λy::Float64
    λx::Float64
end




function subgradient!(game::BilinearGame, g::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}) where T<:Real
    g .= game.A' * y
    g[argmax(x)] += game.λx
end

function supergradient!(game::BilinearGame, g::AbstractVector{T}, x::AbstractVector{T}, y::AbstractVector{T}) where T<:Real
    g .= game.A * x
    g[argmax(y)] -= game.λy
end

function eval(game::BilinearGame, x::AbstractVector{T}, y::AbstractVector{T}) where T<:Real
    return y' * game.A * x + game.λx * maximum(x) + game.λy * minimum(y)
end


function generate_random_zero_sum(n::Int, m::Int, density::Float64, range::Tuple; seed::Int=0)
    generator = Xoshiro(seed)
    A = (((rand(generator, m, n) .* (range[2] - range[1])) .+ range[1]) .* convert.(Float64, rand(generator, m, n) .<= density))
    return BilinearGame(A, n, m, 1.0, 1.0)
end


function prox_mapping_y!(::BilinearGame, z::AbstractVector{T}) where T<:Real
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
prox_mapping_x!(game::BilinearGame, x) = prox_mapping_y!(game, x)


function primalv(game::BilinearGame, x::AbstractVector{T}; solution::Bool=false) where T<:Real
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
function dualv(game::BilinearGame, y::AbstractVector{T}; solution::Bool=false) where T<:Real
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
function subgradient_method(problem::BilinearGame, R::Float64, target_accuracy::Float64)
    gx = zeros(problem.N)
    gy = zeros(problem.M)
    x = ones(problem.N) / problem.N
    y = ones(problem.M) / problem.M
    anorm = norm(problem.A)
    h = IndSimplex()
    niter = convert(Int, ceil(128 * anorm^2 * 1 / (target_accuracy^2)))
    hₖ = target_accuracy / (32 * anorm^2)
    x̄ = zeros(problem.N)
    ȳ = zeros(problem.M)
    for k in 1:niter
        hₖ = R / (sqrt(k))
        if k % 10000 == 0
            println(k, " ", primalv(problem, x̄) - dualv(problem, ȳ))
        end
        if k % 10000 == 0 && primalv(problem, x̄) - dualv(problem, ȳ) <= target_accuracy
            return x, y
            # println(niter, " ", eval(problem, x, y), " ", primalv(problem, x) - dualv(problem, y), " ", maximum(problem.A * x) - minimum(problem.A' * y))
        end
        # fill!(gx, 0.0)
        # fill!(gy, 0.0)
        subgradient!(problem, gx, x, y)
        supergradient!(problem, gy, x, y)
        normval = sqrt(norm(gx)^2 + norm(gy)^2)
        axpby!(-hₖ ./ normval, gx, 1.0, x)
        axpby!(hₖ ./ normval, gy, 1.0, y)

        prox!(x, h, x)
        prox!(y, h, y)
        axpby!(1 / k, x, (k - 1) / k, x̄)
        axpby!(1 / k, y, (k - 1) / k, ȳ)
        # println(y, " || ", sum(y))
        # Want 0 \in subdifferential of 
    end
    return x, y
end


