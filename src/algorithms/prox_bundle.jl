# Díaz, Mateo and Grimmer, Benjamin "Optimal Convergence Rates for the Proximal Bundle Method"
# SIAM Journal on Optimization, Vol. 33, No. 2, pp. 394-423 (2023).

using Base.Iterators
using ProximalCore: Zero
using OSQP
using Printf
using LinearAlgebra
import SparseArrays as sp
using Random



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



Base.@kwdef mutable struct ProxBundleState{R,Tx}
    x::Tx                   # iterate
    y::Tx = copy(x)                   # null step iterate
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
    null_steps::Int = 0
    descent_steps::Int = 0
end

function bundle_management!(iter::ProxBundleIteration, state::ProxBundleState)
    pnk = size(state.eₖ, 1)
    if pnk <= iter.memory
        return 0
    end
    perm = sort(sortperm(state.α)[1:end-iter.memory])
    deleteat!(state.fₖ, perm)
    deleteat!(state.sₖ, perm)
end
"""
ProxBundle Iteration

"""



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
    state.sφ_x .= sum(state.sₖ .* state.α)
    state.y .= state.x - state.rho .* state.sφ_x

    return results
end

function Base.iterate(
    iter::ProxBundleIteration
)
    state = ProxBundleState(x=iter.x0, rho=1 / iter.Mf)
    fp, ∇f = value_and_gradient(iter.f, state.x)
    state.sf_x = ∇f
    state.f_x = fp
    state.g_x = iter.g(state.x)
    state.fₖ = [fp - state.x' * ∇f]
    state.sₖ = [∇f]
    state.eₖ = [0.0]
    subproblem_qp!(state)
    prox!(state.y, iter.g, state.y, state.rho)
    state.εₖ = state.α' * state.eₖ
    # println("Error", state.eₖ)
    state.δₖ = state.εₖ + state.rho / (2) * norm(state.sφ_x)^2

    state.f_x = iter.f(state.x)
    f_y, sf_y = value_and_gradient(iter.f, state.y)
    if state.f_x - iter.f(state.y) ≥ iter.β * state.δₖ
        # Descent step
        state.x .= state.y
        state.descent_steps += 1
    else
        push!(state.sₖ, sf_y)
        push!(state.fₖ, f_y - sf_y' * state.y)
        state.null_steps += 1
    end
    return state, state
end

function Base.iterate(
    iter::ProxBundleIteration,
    state::ProxBundleState
)
    subproblem_qp!(state)
    prox!(state.y, iter.g, state.y, state.rho)
    state.εₖ = state.α' * state.eₖ
    state.δₖ = state.εₖ + state.rho / (2) * norm(state.sφ_x)^2
    state.f_x = iter.f(state.x)
    f_y, sf_y = value_and_gradient(iter.f, state.y)
    println("ε = ", state.εₖ)
    # println(state.δₖ)
    if state.f_x - iter.f(state.y) ≥ iter.β * state.δₖ
        # Descent step
        state.x .= state.y
        state.descent_steps += 1
    else
        # Null step, update the model
        push!(state.sₖ, sf_y)
        push!(state.fₖ, f_y - sf_y' * state.y)
        state.null_steps += 1
    end
    bundle_management!(iter, state)
    return state, state
end

default_solution(::ProxBundleIteration, state::ProxBundleState) = state.x, state.f_x, state.descent_steps, state.null_steps, state.εₖ

ProxBundle(;
    maxit=100,
    tol=1e-8,
    # termination_type="",
    stop=(iter, state) -> tol >= state.εₖ && state.null_steps + state.descent_steps >= 2,
    solution=default_solution,
    verbose=false,
    freq=10,
    display=(it, iter, state) ->
        @printf("%5d | %.3e\n", it, iter.f(state.x)),
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
