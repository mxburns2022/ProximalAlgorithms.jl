# Díaz, Mateo and Grimmer, Benjamin "Optimal Convergence Rates for the Proximal Bundle Method"
# SIAM Journal on Optimization, Vol. 33, No. 2, pp. 394-423 (2023).

using Base.Iterators
using ProximalCore: Zero
using OSQP
using Printf
using LinearAlgebra
import SparseArrays as sp
using Random
using ProximalOperators: IndSimplex



"""
    ProxBundleIteration(; <keyword-arguments>)

Iterator implementing the proximal-bundle algorithm [1].

This iterator solves optimization problems of the form

    minimize f(x) + g(x)

where `f` is potentially non-smooth and `g` has an available proximal mapping.

# Arguments
"""
Base.@kwdef struct ProxBundleIteration{R,C<:Union{R,Complex{R}},Tx<:AbstractArray{C},Tf,Th}
    x0::Tx
    f::Tf = Zero()
    g::Th = Zero()
    Mf::R = 1.0
    mf::R = 0.0
    memory::Int = 10         # -1 for full memory, 1 for single cut, 2 for two-cut
    β::R = 0.9
end

Base.IteratorSize(::Type{<:ProxBundleIteration}) = Base.IsInfinite()
Base.@kwdef struct ProxBundleSubproblem{R,Tx,Tb,TA}
    x0::Tx
    A::TA
    b::Tb
    h::Any
    θ::R
end

function ProximalAlgorithms.value_and_gradient(prob::ProxBundleSubproblem, λ)
    y, _ = prox(prob.h, prob.x0 - prob.θ * prob.A * λ, prob.θ)
    value = (prob.A * λ)' * y + λ' * prob.b + prob.h(y) + 1 / (2prob.θ) * norm(y - prob.x0)^2
    return value, prob.A' * y + prob.b
end

function (prob::ProxBundleSubproblem)(λ)
    y, _ = prox(prob.h, prob.x0 - prob.θ * prob.A * λ, prob.θ)
    (prob.A * λ)' * y + λ' * prob.b + prob.h(y) + 1 / (2prob.θ) * norm(y - prob.x0)^2
end


Base.@kwdef mutable struct ProxBundleState{R,Tx}
    x::Tx                   # iterate
    y::Tx = copy(x)                   # null step iterate
    xmin::Tx = copy(x)                   # best iterate
    fmin::R = 0.0                 # best iterate
    α::Tx = [real(eltype(x))(1.0)]                  # qp dual values
    f_x::R = real(eltype(x))(0.0)                 # value of f at x
    g_x::R = real(eltype(x))(0.0)                 # value of g at x
    sf_x::Tx = zero(x)                # subgradient of f at x
    sφ_x::Tx = zero(x)                # subgradient of model function φ at x
    rho::R = real(eltype(x))(1.0)                 # stepsize parameter of forward and backward steps
    sₖ::AbstractArray{Tx} = [zero(x)]   # subgradient history
    fₖ::AbstractArray{R} = []   # function value history
    eₖ::AbstractArray{R} = []    # error history
    δₖ::R = 1.0  # error history
    εₖ::R = 1.0  # error history
    subsolver::Any = ProximalAlgorithms.SFISTA()
    null_steps::Int = 0
    descent_steps::Int = 0
end



function subproblem_composite(iter::ProxBundleIteration, state::ProxBundleState)
    S = hcat([
        si for si in state.sₖ
    ]...)
    subprob = ProxBundleSubproblem(x0=state.x, A=S, b=state.fₖ, h=iter.g, θ=state.rho)
    state.α, iters = state.subsolver(f=subprob, Lf=norm(S)^2 * state.rho, g=IndSimplex(), x0=ones(size(state.sₖ, 1)) / size(state.sₖ, 1))
    prox!(state.y, iter.g, state.x - state.rho * S * state.α)
    state.sφ_x = (state.x - state.y) / state.rho
end


function bundle_management!(iter::ProxBundleIteration, state::ProxBundleState)
    if iter.memory == 2 && size(state.sₖ, 1) > 1 && false
        state.sₖ[1] = state.sₖ[2] * state.α[1] + state.sf_x * (1 - state.α[1])
        deleteat!(state.sₖ, 2)
        return 0
    end

    pnk = size(state.eₖ, 1)
    if pnk <= iter.memory
        return 0
    end
    # println(state.sₖ)
    state.sₖ = [sum([αi * si for (αi, si) in zip(state.α, state.sₖ)])]
    # println(state.sₖ)
    # sleep(10)
    state.fₖ = [sum([αi * fi for (αi, fi) in zip(state.α, state.fₖ)])]
    # perm = sort(sortperm(state.α)[1:end-iter.memory])
    # deleteat!(state.fₖ, perm)
    # deleteat!(state.sₖ, perm)
end
"""
ProxBundle Iteration

"""
function _twocut_minimize(a₁::Tx, b₁::R, a₂::Tx, b₂::R, λ::R, x₀::Tx, h::Any; tol::R=1e-6) where {R,Tx}
    τ_low = 0.0
    τ_high = 1.0
    x⁺ = zeros(size(x₀))
    _ = prox!(x⁺, h, x₀ - λ * (τ_high * a₁ - (1 - τ_high) * a₂), λ)
    dϕ⁺ = (a₁ - a₂)' * x⁺ + b₁ - b₂
    _ = prox!(x⁺, h, x₀ - λ * (τ_low * a₁ - (1 - τ_low) * a₂), λ)
    dϕ⁻ = (a₁ - a₂)' * x⁺ + b₁ - b₂
    τ = -1.0
    dϕ = 1.0
    while abs(dϕ) > tol
        τ = (τ_high + τ_low) / 2
        # println("$(abs(dϕ⁻)), $(abs(dϕ⁺)), [$(τ_high), $(τ_low)]")
        if abs(dϕ⁺) > abs(dϕ⁻)
            τ_high = τ
            _ = prox!(x⁺, h, x₀ - λ * (τ_high * a₁ - (1 - τ_high) * a₂), λ)
            dϕ⁺ = (a₁ - a₂)' * x⁺ + b₁ - b₂
        else
            τ_low = τ
            _ = prox!(x⁺, h, x₀ - λ * (τ_low * a₁ - (1 - τ_low) * a₂), λ)
            dϕ⁻ = (a₁ - a₂)' * x⁺ + b₁ - b₂
        end

        if (τ_high - τ_low) <= tol
            break
        end
    end
    return τ
end

function subproblem_2cut!(iter::ProxBundleIteration, state::ProxBundleState{R,Tx}) where {R,Tx}

    min1, v1g = prox(iter.g, state.x - state.rho * state.sₖ[1], state.rho)
    min2, v2g = prox(iter.g, state.x - state.rho * state.sf_x, state.rho)
    val1 = max((state.fₖ[1] + state.sₖ[1]' * min1), (state.f_x + state.sf_x' * (min1 - state.x))) + v1g
    val2 = max((state.fₖ[1] + state.sₖ[1]' * min2), (state.f_x + state.sf_x' * (min2 - state.x))) + v2g
    if val1 < val2
        state.y .= min1
        state.sφ_x .= state.sₖ[1]
    else
        state.y .= min2
        state.sφ_x .= state.sf_x
    end
    # println(iter.f(state.x), " ", iter.f(state.y), " ", iter.f(min1), " ", iter.f(min2))
    τ = _twocut_minimize(state.sₖ[1], state.fₖ[1], state.sf_x, state.f_x, state.rho, state.x, iter.g)
    state.α = [τ]
end


function subproblem_qp!(state::ProxBundleState{R,Tx}) where {R,Tx}
    P = sp.sparse(
        hcat([
            si for si in state.sₖ
        ]...)
    )
    state.eₖ = state.f_x .- [fi + si' * state.x for (fi, si) in zip(state.fₖ, state.sₖ)]
    # println(state.f_x)
    # println(state.fₖ[1] + state.sₖ[1]' * state.x)
    pnk = size(state.eₖ, 1)

    P = P' * P
    q = Float64.(state.eₖ) ./ state.rho
    l = vcat([1.0], [0.0 for _ in 1:pnk])
    u = vcat([1.0], [1.0 for _ in 1:pnk])
    # fx = rand(rng)
    # u = [ei - fx + si' * x for (ei, si) in zip(ek, sk)]
    row_indices = vcat(ones(Int, pnk), 2:1+pnk)
    col_indices = vcat(1:pnk, 1:pnk)
    values = ones(2pnk)
    A = sp.sparse(row_indices, col_indices, values)
    m = OSQP.Model()
    OSQP.setup!(m; P=P, q=q, A=A, l=l, u=u, eps_abs=1e-8, eps_rel=1e-8, verbose=0, eps_prim_inf=1e-8, eps_dual_inf=1e-8)
    results = OSQP.solve!(m)

    state.α = results.x
    state.α = max.(state.α, 0.0)
    state.sφ_x .= sum(state.sₖ .* state.α)
    state.y .= state.x - state.rho .* state.sφ_x

    return results
end
function solve_subproblem!(iter::ProxBundleIteration, state::ProxBundleState{R,Tx}) where {R,Tx}
    # j = argmax(
    #     [
    #     fx + sx'state.x for (fx, sx) in zip(state.fₖ, state.sₖ)
    # ]
    # )
    # state.sφ_x .= state.sₖ[j]
    # prox!(state.y, iter.g, state.x - state.rho * state.sφ_x, state.rho)[1]

    if iter.memory == 2 && false
        subproblem_2cut!(iter, state)
    else
        # subproblem_qp!(state)
        subproblem_composite(iter, state)
        # prox!(state.y, iter.g, state.y, state.rho)
    end
end

function Base.iterate(
    iter::ProxBundleIteration
)
    state = ProxBundleState(x=iter.x0, rho=1 / iter.Mf)
    state.x, _ = prox(iter.g, state.x)
    state.xmin .= state.x
    fp, ∇f = value_and_gradient(iter.f, state.x)
    state.sf_x = ∇f
    state.f_x = fp
    state.fmin = fp
    state.g_x = iter.g(state.x)
    state.fₖ = [fp - state.x' * ∇f]
    state.sₖ = [∇f]
    state.eₖ = [0.0]
    solve_subproblem!(iter, state)
    # subproblem_qp!(state)
    # prox!(state.y, iter.g, state.y, state.rho)
    state.εₖ = state.α' * state.eₖ
    # println("Error", state.eₖ)
    state.δₖ = state.εₖ + state.rho / (2) * norm(state.sφ_x)^2

    state.f_x = iter.f(state.x)

    f_y, sf_y = value_and_gradient(iter.f, state.y)
    if iter.f(state.xmin) - f_y ≥ iter.β * state.δₖ
        # Descent step
        state.x .= state.y
        if f_y < state.fmin
            state.xmin .= state.y
            state.fmin = f_y
        end
        state.descent_steps += 1
    else
        push!(state.sₖ, sf_y)
        push!(state.fₖ, f_y - sf_y' * state.y)
        state.null_steps += 1
    end
    return state, state
end
function eval_bundle(
    iter::ProxBundleIteration,
    state::ProxBundleState
)
    value = maximum(
                [
                fk + sk' * state.y for (fk, sk) in zip(state.fₖ, state.sₖ)
            ]
            ) + 1 / (2state.rho) * norm(state.x - state.y)^2 + iter.g(state.y)

    return value
end
function Base.iterate(
    iter::ProxBundleIteration,
    state::ProxBundleState
)
    solve_subproblem!(iter, state)
    # subproblem_qp!(state)
    # prox!(state.y, iter.g, state.y, state.rho)
    # state.εₖ = state.α' * state.eₖ
    # println(state.εₖ)
    # @assert state.εₖ > 0

    d1 = state.εₖ + state.rho / (2) * norm(state.sφ_x)^2
    state.f_x = iter.f(state.x)
    f_y, sf_y = value_and_gradient(iter.f, state.y)
    state.δₖ = state.fmin + iter.g(state.xmin) - eval_bundle(iter, state)

    if state.f_x - iter.f(state.y) ≥ iter.β * state.δₖ
        # Descent step
        state.x .= state.y
        state.descent_steps += 1
        if f_y < state.fmin
            state.xmin .= state.y
            state.fmin = f_y
        end
    else
        state.null_steps += 1
        # Null step, update the model
        push!(state.sₖ, sf_y)
        push!(state.fₖ, f_y - sf_y' * state.y)
    end
    bundle_management!(iter, state)
    return state, state
end

default_solution(::ProxBundleIteration, state::ProxBundleState) = state.x, state.f_x, state.descent_steps, state.null_steps, state.εₖ

ProxBundle(;
    maxit=1000,
    tol=1e-8,
    # termination_type="",
    stop=(iter, state) -> (tol >= state.δₖ) && state.null_steps + state.descent_steps >= 2,
    solution=default_solution,
    verbose=true,
    freq=10,
    display=(it, iter, state) ->
        @printf("%5d | %.3e | %.3e\n", it, iter.f(state.x), state.δₖ),
    kwargs...,
) = IterativeAlgorithm(
    ProxBundleIteration;
    maxit,
    stop,
    solution,
    verbose,
    freq,
    display,
    kwargs...,
)
