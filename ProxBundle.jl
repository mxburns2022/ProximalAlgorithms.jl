using ProximalAlgorithms
# using ProximalOperators
include("SaddlePoint.jl")
import ProximalOperators: prox!
using Test
using DataFrames
using CSV
using Ipopt
using JuMP



Base.@kwdef struct ProxBundleSubproblem{R,Tx,Tb,TA}
    x0::Tx
    A::TA
    b::Tb
    h::Any
    θ::R
end

Base.@kwdef mutable struct ProxBundleState{R,Tx}
    x::Tx                   # iterate
    y::Tx = copy(x)                   # null step iterate
    x0::Tx = copy(x)                   # null step iterate
    fmin::R = 0.0                 # best iterate
    g::Any
    α::Tx = [real(eltype(x))(1.0)]                  # qp dual values
    f_x::R = real(eltype(x))(0.0)                 # value of f at x
    f_y::R = real(eltype(x))(0.0)                 # value of f at y
    g_x::R = real(eltype(x))(0.0)                 # value of g at x
    sf_x::Tx = zero(x)                # subgradient of f at x
    sφ_x::Tx = zero(x)                # subgradient of model function φ at x
    rho::R = real(eltype(x))(1.0)                 # stepsize parameter of forward and backward steps
    sₖ::AbstractArray{Tx} = []   # subgradient history
    fₖ::AbstractArray{R} = []   # function value history
    eₖ::AbstractArray{R} = []    # error history
    δₖ::R = 1.0  # error history
    εₖ::R = 1.0  # error history
    subsolver::Any = ProximalAlgorithms.SFISTA(tol=1e-10)
    null_steps::Int = 0
    descent_steps::Int = 0
end

# Solve \min_{x ∈ Δₙ}\max_{y ∈ Δₘ}<Ax, y> + ||Bx||∞ -||-Cy||∞
Base.@kwdef mutable struct ProximalBilinearSPP
    game::BilinearSPP
    stepsize::Float64
    x::Vector{Float64} = zeros(game.N)
    y::Vector{Float64} = zeros(game.M)
    x⁺::Vector{Float64} = zeros(game.N)
    y⁺::Vector{Float64} = zeros(game.M)
    x̃::Vector{Float64} = zeros(game.N)
    ỹ::Vector{Float64} = zeros(game.M)
    setting::Symbol = :Min
    gradient_cache_x::Vector{Float64} = zeros(game.N)
    gradient_cache_y::Vector{Float64} = zeros(game.M)
    h = IndSimplex()
end
function bundle_management!(state::ProxBundleState, memory::Int)

    pnk = size(state.sₖ, 1)
    if pnk <= memory
        return 0
    end

    # Perform cut aggregation
    state.sₖ = [sum([αi * si for (αi, si) in zip(state.α, state.sₖ)])]
    state.fₖ = [sum([αi * fi for (αi, fi) in zip(state.α, state.fₖ)])]
end


function gradient(f::ProximalBilinearSPP, x)
    if f.setting == :Min
        subgradient!(f.game, f.gradient_cache_x, x, f.y)
        # f.gradient_cache_x .+= 1 / f.stepsize * (x - f.x)
        # value = eval(f.game, x, f.y) + 1 / 2f.stepsize * norm(x - f.x)^2
        return f.gradient_cache_x
    elseif f.setting == :Max
        supergradient!(f.game, f.gradient_cache_y, f.x, x)
        # Negate the gradient to convert the supergradient of the max problem to a subgradient of the min problem
        # f.gradient_cache_y .-= f.stepsize * (x - f.y)
        # value = -eval(f.game, f.x, x) + 1 / 2f.stepsize * norm(x - f.y)^2
        return -f.gradient_cache_y
    else
        throw(ArgumentError("Unrecognized subproblem setting: The subproblem setting should either be :Min or :Max"))
    end
end


function init_prox_step(pgame::ProximalBilinearSPP, setting::Symbol)
    @assert setting == :Min || setting == :Max
    fill!(pgame.gradient_cache_x, 0.0)
    fill!(pgame.gradient_cache_y, 0.0)
    pgame.setting = setting
end

function (a::ProximalBilinearSPP)(x)
    if a.setting == :Min
        return eval(a.game, x, a.y)
    else
        return -eval(a.game, a.x, x)
    end

end


function _twocut_minimize(a₁::Tx, b₁::R, a₂::Tx, b₂::R, λ::R, x₀::Tx, h::Any; tol::R=1e-12) where {R,Tx}
    τ_low = 0.0
    τ_high = 1.0
    x⁺ = zeros(size(x₀))
    τ = -1.0
    dϕ = 1.0
    niters = 0
    while abs(dϕ) > tol
        if (τ_high - τ_low) <= tol || abs(dϕ) < tol
            break
        end
        τ = (τ_high + τ_low) / 2
        prox!(x⁺, h, x₀ - λ * (τ * a₁ - (1 - τ) * a₂), λ)
        niters += 1
        if abs(dϕ) > 0
            τ_high = τ
        else
            τ_low = τ
        end

    end

    return τ, niters
end

function subproblem_2cut!(game::ProximalBilinearSPP, state::ProxBundleState{R,Tx}) where {R,Tx}

    # min1, v1g = prox(game.h, state.x0 - state.rho * state.sₖ[1], state.rho)
    # min2, v2g = prox(game.h, state.x0 - state.rho * state.sf_x, state.rho)
    # val1 = max((state.fₖ[1] + state.sₖ[1]' * min1), (state.f_x + state.sf_x' * (min1 - state.x))) + v1g
    # val2 = max((state.fₖ[1] + state.sₖ[1]' * min2), (state.f_x + state.sf_x' * (min2 - state.x))) + v2g
    # if val1 < val2
    #     state.y .= min2
    #     state.sφ_x .= state.sₖ[1]
    # else
    #     state.y .= min1
    #     state.sφ_x .= state.sf_x
    # end
    # println(iter.f(state.x), " ", iter.f(state.y), " ", iter.f(min1), " ", iter.f(min2))
    τ, niters = _twocut_minimize(state.sₖ[1], state.fₖ[1], gradient(game, state.y), game(state.y), state.rho, state.x0, game.h)
    state.α = [τ]
    prox!(state.y, game.h, state.x0 - state.rho * (τ * state.sₖ[1] + (1 - τ) * state.sf_x))
    return niters
end


# function ProximalAlgorithms.value_and_gradient(prob::ProxBundleSubproblem, λ)
#     y, _ = prox(prob.h, prob.x0 - prob.θ * prob.A * λ, prob.θ)
#     value = (prob.A * λ)' * y + λ' * prob.b + prob.h(y) + 1 / (2prob.θ) * norm(y - prob.x0)^2
#     return value, prob.A' * y + prob.b
# end
function ProximalAlgorithms.gradient(prob::ProxBundleSubproblem, λ)
    y, _ = prox(prob.h, prob.x0 - prob.θ * prob.A * λ, prob.θ)
    value = (prob.A * λ)' * y + λ' * prob.b + prob.h(y) + 1 / (2prob.θ) * norm(y - prob.x0)^2
    return -(prob.A' * y + prob.b), -value
end
function ProximalAlgorithms.gradient!(z, prob::ProxBundleSubproblem, λ)
    y, _ = prox(prob.h, prob.x0 - prob.θ * prob.A * λ, prob.θ)
    # println("grad ", -(prob.A' * y + prob.b))
    z .= -(prob.A' * y + prob.b)
end
function (prob::ProxBundleSubproblem)(λ)
    y, _ = prox(prob.h, prob.x0 - prob.θ * prob.A * λ, prob.θ)
    (prob.A * λ)' * y + λ' * prob.b + prob.h(y) + 1 / (2prob.θ) * norm(y - prob.x0)^2
end

function solve_composite!(state::ProxBundleState)
    perm = randperm(size(state.sₖ, 1))
    # perm = [1:size(state.sₖ, 1)...]
    S = hcat(state.sₖ[perm]...)
    # println(S)
    h = IndSimplex()
    subprob = ProxBundleSubproblem(x0=state.x0, A=S, b=state.fₖ[perm], h=state.g, θ=state.rho)
    state.α, iters = state.subsolver(f=subprob, Lf=norm(S)^2 * state.rho, g=h, x0=ones(size(S, 2)) ./ size(S, 2))
    # println(state.α)
    # println(S * state.α)
    # println(sum(S .* state.α', dims=2))
    state.y, _ = prox(state.g, state.x0 - state.rho * S * state.α)
    state.α = state.α[sortperm(perm)]
    return iters
    # println(iters)
    # state.f_y = state.α' * state.fₖ + state.y' * S * state.α + 1 / 2state.rho * norm(state.y - state.x0)^2 + state.g(state.y)
end


function test_solve_composite()
    gen = Xoshiro(0)
    game = generate_random_zero_sum(5, 5, 0.4, 10.0; seed=1)
    stepsize = 0.1

    proxgame = ProximalBilinearSPP(game=game, stepsize=stepsize)
    x0 = normalize(rand(gen, game.N), 1)
    proxgame.y = normalize(rand(gen, game.M), 1)
    proxgame.x = x0
    init_prox_step(proxgame, :Min)
    state = ProxBundleState(x=copy(proxgame.x), rho=proxgame.stepsize, g=proxgame.h)
    p = 40
    for i in 1:p
        y = normalize(rand(gen, game.N), 1)
        ∇f = gradient(proxgame, y)
        fp = proxgame(state.x)
        push!(state.fₖ, fp - y' * ∇f)
        push!(state.sₖ, copy(∇f))
    end
    solve_composite!(state)
    strength = 1e-2
    f_y = eval_bundle(state) + 1 / (2 * state.rho) * norm(state.y - state.x0)^2
    cache = copy(state.y)
    println(state.y)
    for i in 1:100000
        state.y, _ = prox(proxgame.h, state.y + (strength .* rand(gen, size(state.y, 1))))
        f_other = eval_bundle(state) + 1 / (2 * state.rho) * norm(state.y - state.x0)^2
        @test f_other >= f_y
        state.y .= cache
    end


end

function test_solve_2cut()
    gen = Xoshiro(0)
    game = generate_random_zero_sum(20, 20, 0.4, 10.0; seed=1)
    stepsize = 0.1

    proxgame = ProximalBilinearSPP(game=game, stepsize=stepsize)
    x0 = normalize(rand(gen, game.N), 1)
    proxgame.y = normalize(rand(gen, game.M), 1)
    proxgame.x = x0
    init_prox_step(proxgame, :Min)
    state = ProxBundleState(x=copy(proxgame.x), rho=proxgame.stepsize, g=proxgame.h)
    p = 1
    for i in 1:p
        y = normalize(rand(gen, game.N), 1)
        ∇f = gradient(proxgame, y)
        fp = proxgame(state.x)
        push!(state.fₖ, fp - y' * ∇f)
        push!(state.sₖ, copy(∇f))
    end
    subproblem_2cut!(proxgame, state)
    strength = 1e-3
    f_y = eval_bundle(state) + 1 / (2 * state.rho) * norm(state.y - state.x0)^2
    cache = copy(state.y)
    for i in 1:1000000
        state.y, _ = prox(proxgame.h, state.y + (strength .* rand(gen, size(state.y, 1))))
        f_other = eval_bundle(state) + 1 / (2 * state.rho) * norm(state.y - state.x0)^2
        @test f_other >= f_y
        state.y .= cache
    end


end

function eval_bundle(
    state::ProxBundleState
)
    value = maximum(
        [
        fk + sk' * state.y for (fk, sk) in zip(state.fₖ, state.sₖ)
    ]
    )
    # println(state.sₖ)
    # println([
    #     fk + sk' * state.y for (fk, sk) in zip(state.fₖ, state.sₖ)
    # ])
    return value
end

function PDCP!(game::ProximalBilinearSPP, ε::Float64, memory::Int=20)
    # Start by building the model
    if game.setting == :Min
        state = ProxBundleState(x=copy(game.x), g=game.h, rho=game.stepsize)
    else
        state = ProxBundleState(x=copy(game.y), g=game.h, rho=game.stepsize)
    end
    # ∇f = value_and_gradient(game, state.x)
    state.sf_x = gradient(game, state.x)
    state.f_x = game(state.x)
    fcomp = game(state.y)
    state.fmin = fcomp
    state.fₖ = [state.f_x - state.x' * state.sf_x]
    state.sₖ = [copy(state.sf_x)]
    nsteps = 0
    tj = 1.0
    while tj > ε
        if memory > 2
            nsteps += solve_composite!(state)
            # elseif memory == 2
        elseif memory == 2
            nsteps += subproblem_2cut!(game, state)
        elseif memory == 1
            prox!(state.y, state.g, state.x0 - state.rho * state.sₖ[1])
            nsteps += 1
        end
        fcomp = game(state.x)
        fy = eval_bundle(state)
        normx = 1 / (2 * state.rho) * norm(state.x - state.x0)^2
        normy = 1 / (2 * state.rho) * norm(state.y - state.x0)^2
        tj = fcomp + normx - (fy + normy)
        ϕy = game(state.y)
        sf_y = gradient(game, state.y)
        if ϕy <= fcomp - 1e-10
            copy!(state.x, state.y)
        end
        τⱼ = (nsteps - 1) / nsteps
        if memory == 1
            state.sₖ = [state.sₖ[1] * τⱼ + sf_y * (1 - τⱼ)]
            state.fₖ = [state.fₖ[1] * τⱼ + (ϕy - sf_y' * state.y) * (1 - τⱼ)]
        else
            push!(state.sₖ, copy(sf_y))
            push!(state.fₖ, ϕy - sf_y' * state.y)
            bundle_management!(state, memory)
        end
    end
    if game.setting == :Min
        copy!(game.x⁺, state.y)
        copy!(game.x̃, state.x)
    else
        copy!(game.y⁺, state.y)
        copy!(game.ỹ, state.x)
    end
    return nsteps

    # return (state.y, state.x)
end
function test_PDCP!()
    game = generate_random_zero_sum(15, 15, 0.2, 5.0; seed=2)
    η = 0.001
    proxgame = ProximalBilinearSPP(game=game, stepsize=η)
    gen = Xoshiro(1)
    x0 = normalize(rand(gen, game.N), 1)
    proxgame.x = x0
    y = normalize(rand(gen, game.M), 1)
    proxgame.y = y
    proxgame.x = copy(x0)
    algo = ProximalAlgorithms.SFISTA(tol=1e-10)
    sol, iters = algo(f=proxgame, g=IndSimplex(), Lf=1 / η, x0=copy(x0))
    println(x0)
    PDCP!(proxgame, 1e-8, 100)
    println(proxgame.x)
    println(sol)

end
function bundle_saddle_point(game::BilinearSPP, target_accuracy::Float64; memory::Int=typemax(Int), log_frequency=10)
    λ₁ = 1 / 4(norm(game.A, Inf) + 1)
    proxgame = ProximalBilinearSPP(game=game, stepsize=λ₁)


    proxgame.x .= ones(game.N) / game.N
    proxgame.y .= ones(game.M) / game.M
    copy!(proxgame.x̃, proxgame.x)
    copy!(proxgame.ỹ, proxgame.y)
    x̄ = zeros(game.N)
    ȳ = zeros(game.M)
    inner_steps = 0
    starttime = time_ns()
    # data = []
    println("elapsed_time(s),inner_steps,outer_steps,pd_gap")
    for k in 1:300000
        proxgame.stepsize = λ₁ / sqrt(k)
        axpby!(1 / k, proxgame.x̃, (k - 1) / k, x̄)
        axpby!(1 / k, proxgame.ỹ, (k - 1) / k, ȳ)
        # println(x̄)
        # sleep(1)
        if (k - 1) % log_frequency == 0
            # push!(data,
            #     (
            #         currtime=round((time_ns() - starttime) / 1e9, sigdigits=6),
            #         k=k,
            #         inner_steps=inner_steps,
            #         primal_dual_gap=primalv(game, x̄) - dualv(game, ȳ)
            #     ))
            println(round((time_ns() - starttime) / 1e9, sigdigits=6), ",", inner_steps, ",", k, ",", primalv(game, x̄) - dualv(game, ȳ))
            flush(stdout)
        end
        if k % 100 == 0 && primalv(game, x̄) - dualv(game, ȳ) <= target_accuracy
            # return x, y
            return
            # println(niter, " ", eval(problem, x, y), " ", primalv(problem, x) - dualv(problem, y), " ", maximum(problem.A * x) - minimum(problem.A' * y))
        end

        init_prox_step(proxgame, :Min)
        inner_steps += PDCP!(proxgame, target_accuracy / 4, memory)

        init_prox_step(proxgame, :Max)
        inner_steps += PDCP!(proxgame, target_accuracy / 4, memory)
        copy!(proxgame.x, proxgame.x⁺)
        copy!(proxgame.y, proxgame.y⁺)

    end
end

