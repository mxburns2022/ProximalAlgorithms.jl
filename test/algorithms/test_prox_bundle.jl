using Test
using ProximalAlgorithms
using ProximalAlgorithms: ProxBundleIteration, ProxBundleState, subproblem_qp!, subproblem_composite
using ProximalOperators
using LinearAlgebra
using Random
using Base.Iterators
using ProximalAlgorithms.IterationTools
# using Zygote
using DifferentiationInterface: AutoZygote


function test_qp_subproblem()
    N = 5
    gen = Xoshiro(0)
    Q = rand(gen, N, N)
    Q = Q' * Q
    h = NormL1(1.0)
    f = ProximalAlgorithms.AutoDifferentiable(x -> 0.5 * x' * Q * x, AutoZygote())
    x0 = rand(gen, N)
    state = ProxBundleState(
        x=x0,
        rho=0.1
    )
    fp = 0.5 * x0' * Q * x0
    ∇f = Q * x0
    state.sf_x = ∇f
    state.g_x = h(state.x)
    state.fₖ = [fp - state.x' * ∇f]
    state.sₖ = [∇f]
    state.eₖ = [0.0]
    subproblem_qp!(state)
    actual_optimum = state.x - state.rho * ∇f
    @test norm(state.y - actual_optimum) ≈ 0.0 atol = 1e-8
    state.x .= state.y
    push!(state.fₖ, -1 / 2 * state.y' * Q * state.y)
    push!(state.sₖ, Q * state.y)
    subproblem_qp!(state)
    function φ(x)
        fx = 0.5 * state.x' * Q * state.x
        return fx + maximum(
                   [-(fx - fi + si' * (state.x)) + si' * (x - state.x) for (fi, si) in zip(state.fₖ, state.sₖ)]
               ) + 1 / (2state.rho) * norm(x - state.x)^2
    end
    vy = φ(state.y)
    # for i in range(200)
    values = [φ(state.y + (randn(gen, size(x0, 1)) .* 1e-4)) for i in 1:2000]
    @test minimum(values) >= vy


    state.x .= state.y
    push!(state.fₖ, -1 / 2 * state.y' * Q * state.y)
    push!(state.sₖ, Q * state.y)
    println(state.y)
    subproblem_qp!(state)
    println(f(state.y))
    function φ(x)
        fx = 0.5 * state.x' * Q * state.x
        return fx + maximum(
                   [-(fx - fi + si' * (state.x)) + si' * (x - state.x) for (fi, si) in zip(state.fₖ, state.sₖ)]
               ) + 1 / (2state.rho) * norm(x - state.x)^2
    end
    vy = φ(state.y)
    println(state.y)
    # for i in range(200)
    values = [φ(state.y + (randn(gen, size(x0, 1)) .* 1e-4)) for i in 1:2000]
    @test minimum(values) >= vy

    # println(state.eₖ)
    # println(state.α)
    # println(state.y)
    # println(x1)
    # println(state.x - ∇f)
    # println(state.fₖ[1] + state.sₖ[1]' * state.x, " ", state.fₖ[2] + state.sₖ[2]' * state.x)
    # state.sₖ = [∇f]
end


function test_fista_subproblem()
    N = 5
    gen = Xoshiro(0)
    Q = rand(gen, N, N)
    Q = Q' * Q
    h = IndSimplex(1.0)
    f = ProximalAlgorithms.AutoDifferentiable(x -> 0.5 * x' * Q * x, AutoZygote())
    x0 = rand(gen, N)
    state = ProxBundleState(
        x=x0,
        rho=0.1
    )
    iter = ProxBundleIteration(
        x0=x0,
        f=f,
        g=h
    )
    fp = 0.5 * x0' * Q * x0
    ∇f = Q * x0
    state.sf_x = ∇f
    state.g_x = h(state.x)
    state.fₖ = [fp - state.x' * ∇f]
    state.sₖ = [∇f]
    state.eₖ = [0.0]
    subproblem_composite(iter, state)
    println(state.α)
    actual_optimum, _ = prox(h, state.x - state.rho * ∇f)
    @test norm(state.y - actual_optimum) ≈ 0.0 atol = 1e-8

end