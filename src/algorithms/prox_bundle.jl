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
    eta::R = 1.0
    memory::Int = 2000         # -1 for full memory, 1 for single cut, 2 for two-cut
    β::R = 0.9
    tol::R = 1e-4
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
    y::Tx = copy(x)                   # null step iterateiterate
    x0::Tx = copy(x)                   # prox center
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
    descent_cond::Bool = false
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


# Bundle Management, if cut limit is reached then aggregate using dual multipliers 
function bundle_management!(state::ProxBundleState, memory::Int)

    pnk = size(state.sₖ, 1)
    if pnk <= memory
        return 0
    end

    # Perform cut aggregation
    state.sₖ = [sum([αi * si for (αi, si) in zip(state.α, state.sₖ)])]
    state.fₖ = [sum([αi * fi for (αi, fi) in zip(state.α, state.fₖ)])]
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
    niters = 0
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
        niters += 1
        if (τ_high - τ_low) <= tol
            break
        end
    end
    return τ, niters
end



# Solve the dual maximization problem for the multicut auxiliary step 
function solve_composite!(state::ProxBundleState)
    perm = randperm(size(state.sₖ, 1)) # Randomize the permutations to break ties as needed to avoid cycling
    S = hcat(state.sₖ[perm]...)
    h = IndSimplex()
    subprob = ProxBundleSubproblem(x0=state.x0, A=S, b=state.fₖ[perm], h=state.g, θ=state.rho)
    state.α, iters = state.subsolver(f=subprob, Lf=norm(S)^2 * state.rho, g=h, x0=ones(size(S, 2)) ./ size(S, 2))
    state.y, _ = prox(state.g, state.x0 - state.rho * S * state.α)

    state.α = state.α[sortperm(perm)]
    indices = findall(<(1e-6), state.α)
    deleteat!(state.fₖ, indices)
    deleteat!(state.sₖ, indices)
    deleteat!(state.α, indices)
    return iters
end

function subproblem_qp!(state::ProxBundleState{R,Tx}) where {R,Tx}
    P = sp.sparse(
        hcat([
            si for si in state.sₖ
        ]...)
    )
    state.eₖ = state.f_x .- [fi + si' * state.x for (fi, si) in zip(state.fₖ, state.sₖ)]
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
    OSQP.setup!(m; P=P, q=q, A=A, l=l, u=u, eps_abs=1e-9, eps_rel=1e-9, verbose=0, eps_prim_inf=1e-9, eps_dual_inf=1e-9)
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

    # TODO Add 1 and 2-cut subroutines
    if iter.g == Zero()
        subproblem_qp!(state)
        nsteps = 1
    else
        nsteps = subproblem_composite(iter, state)
    end
    # end
    return nsteps
end




# Implementation of the Primal-Dual Cutting Plane (PDCP) subroutine
function PDCP!(iter::ProxBundleIteration, state::ProxBundleState, ε::Float64, memory::Int=20)

    # Initialize state parameters
    copy!(state.x, state.xmin)
    copy!(state.y, state.xmin)
    state.f_x, state.sf_x = value_and_gradient(iter.f, state.x)
    state.εₖ = 1.0# iter.f(state.x) - eval_bundle(iter, state)
    state.δₖ = 1.0
    if any(state.sf_x .== NaN)
        state.rho /= 2
        println("$(state.rho)")
        state.descent_cond = true
        return
    end
    fcomp = iter.f(state.y)
    state.fmin = fcomp
    state.fₖ = [state.f_x - state.x' * state.sf_x]
    state.sₖ = [copy(state.sf_x)]
    nsteps = 0
    tj = 1.0
    pdcp_steps = 0
    condition_met = true
    gap_prev = Inf
    # While IPP condition not satisfied
    while tj > ε

        # Find the exact minimizer to the bundle model
        if any(isnan.(state.y))
            state.rho /= 2
            state.y .= state.x
            state.descent_cond = true
            return
        end

        nsteps += solve_subproblem!(iter, state)

        # Compute the loop termination sequence tj

        fy = eval_bundle(iter, state.y, state)
        normx = 1 / (2 * state.rho) * norm(state.x - state.x0)^2
        normy = 1 / (2 * state.rho) * norm(state.y - state.x0)^2
        state.δₖ = fcomp - fy
        tj = fcomp + normx - (fy + normy)
        condition_met = condition_met && (1 - iter.β) * tj < gap_prev
        gap_prev = tj
        # Update the best point if we have a decrease in function value
        ϕy = iter.f(state.y)
        _, sf_y = value_and_gradient(iter.f, state.y)

        if any(isnan.(sf_y))
            state.rho /= 2
            state.y .= state.x
            println("$(state.rho)")
            state.descent_cond = true
            return
        end
        # sleep(0.1)
        if ϕy <= fcomp - 1e-10
            copy!(state.x, state.y)
            fcomp = iter.f(state.x)
            state.descent_steps += 1
        else
            state.null_steps += 1
        end
        τⱼ = (nsteps - 1) / nsteps
        if memory == 1
            state.sₖ = [state.sₖ[1] * τⱼ + sf_y * (1 - τⱼ)]
            state.fₖ = [state.fₖ[1] * τⱼ + (ϕy - sf_y' * state.y) * (1 - τⱼ)]
        else
            push!(state.sₖ, copy(sf_y))
            push!(state.fₖ, ϕy - sf_y' * state.y)
            bundle_management!(state, memory)
            if any(state.sₖ[1] .== NaN)
                state.rho /= 2
                state.y .= state.x
                return
            end
        end
        # end

        # Update the bundle model

        pdcp_steps += 1
    end
    state.descent_cond = condition_met
    # Copy back the final iterates
    copy!(state.xmin, state.x)
    copy!(state.x, state.y)
    state.εₖ = norm(state.x0 - state.xmin)# iter.f(state.x) - eval_bundle(iter, state)
    state.δₖ = iter.f(state.xmin) - eval_bundle(iter, state.xmin, state)
    return nsteps, pdcp_steps
end

function Base.iterate(
    iter::ProxBundleIteration
)
    state = ProxBundleState(x=iter.x0, x0=iter.x0, rho=iter.eta)
    state.x, _ = prox(iter.g, state.x)
    state.xmin .= state.x
    PDCP!(iter, state, iter.tol, iter.memory)
    if !state.descent_cond
        state.rho /= 2
    end
    return state, state
end
function eval_bundle(
    iter::ProxBundleIteration,
    val,
    state::ProxBundleState
)
    value = maximum(
        [
        fk + sk' * val for (fk, sk) in zip(state.fₖ, state.sₖ)
    ]
    ) + iter.g(val)

    return value
end
function Base.iterate(
    iter::ProxBundleIteration,
    state::ProxBundleState
)
    copy!(state.x0, state.xmin)
    PDCP!(iter, state, iter.tol, iter.memory)
    if !state.descent_cond
        state.rho /= 2
    end
    return state, state
    # solve_subproblem!(iter, state)
    # state.f_x = iter.f(state.x)
    # f_y, sf_y = value_and_gradient(iter.f, state.y)
    # state.δₖ = state.fmin + iter.g(state.xmin) - eval_bundle(iter, state)

    # if state.f_x - iter.f(state.y) ≥ iter.β * state.δₖ
    #     # Descent step
    #     state.x .= state.y
    #     state.descent_steps += 1
    #     if f_y < state.fmin
    #         state.xmin .= state.y
    #         state.fmin = f_y
    #     end
    # else
    #     state.null_steps += 1
    #     # Null step, update the model
    #     push!(state.sₖ, sf_y)
    #     push!(state.fₖ, f_y - sf_y' * state.y)
    # end
    # bundle_management!(iter, state)
    # return state, state
end

default_solution(::ProxBundleIteration, state::ProxBundleState) = state.x, state.f_x, state.descent_steps, state.null_steps, state.εₖ

ProxBundle(;
    maxit=1000,
    # termination_type="",
    stop=(iter, state) -> false,# state.δₖ <= iter.tol,
    solution=default_solution,
    verbose=true,
    freq=10,
    display=(it, iter, state) ->
        @printf("%5d | %.7e | %.7e | %.7e | %.3e\n", it, iter.f(state.xmin), state.δₖ, state.εₖ, state.rho),
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
