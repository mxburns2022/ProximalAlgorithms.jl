using ProximalAlgorithms
# using ProximalOperators
include("SaddlePoint.jl")
import ProximalOperators: prox!


Base.@kwdef struct ProxBundleSubproblem{R,Tx,Tb,TA}
    x0::Tx
    A::TA
    b::Tb
    h::Any
    θ::R
end
function simplex_proj_condat!(y, a, x)
    # Implements algorithm proposed in:
    # Condat, L. "Fast projection onto the simplex and the l1 ball",
    # Mathematical Programming, 158:575–585, 2016.
    R = eltype(x)
    v = [x[1]]
    v_tilde = R[]
    rho = x[1] - a
    N = length(x)
    for k in 2:N
        if x[k] > rho
            rho += (x[k] - rho) / (length(v) + 1)
            if rho > x[k] - a
                push!(v, x[k])
            else
                append!(v_tilde, v)
                v = [x[k]]
                rho = x[k] - a
            end
        end
    end
    for z in v_tilde
        if z > rho
            push!(v, z)
            rho += (z - rho) / length(v)
        end
    end
    v_changed = true
    while v_changed == true
        v_changed = false
        k = 1
        while k <= length(v)
            z = v[k]
            if z <= rho
                deleteat!(v, k)
                v_changed = true
                rho += (rho - z) / length(v)
            else
                k = k + 1
            end
        end
    end
    y .= max.(x .- rho, R(0))
end

function prox!(y, f::IndSimplex, x, gamma)
    simplex_proj_condat!(y, f.a, x)
    return eltype(x)(0)
end



Base.@kwdef mutable struct ProxBundleState{R,Tx}
    x::Tx                   # iterate
    y::Tx = copy(x)                   # null step iterate
    x0::Tx = copy(x)                   # null step iterate
    fmin::R = 0.0                 # best iterate
    g::Any
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

# Solve \min_{x ∈ Δₙ}\max_{y ∈ Δₘ}<Ax, y> + ||Bx||∞ -||-Cy||∞
Base.@kwdef mutable struct ProximalBilinearGame <: SaddlePointProblem
    game::BilinearGame
    stepsize::Float64
    x::Vector{Float64} = zeros(game.N)
    y::Vector{Float64} = zeros(game.M)
    x̃::Vector{Float64} = zeros(game.N)
    ỹ::Vector{Float64} = zeros(game.M)
    setting::Symbol = :Min
    gradient_cache_x::Vector{Float64} = zeros(game.N)
    gradient_cache_y::Vector{Float64} = zeros(game.M)
    h = IndSimplex()
end
function bundle_management!(state::ProxBundleState, memory::Int)

    pnk = size(state.eₖ, 1)
    if pnk <= memory
        return 0
    end

    # Perform cut aggregation
    state.sₖ = [sum([αi * si for (αi, si) in zip(state.α, state.sₖ)])]
    state.fₖ = [sum([αi * fi for (αi, fi) in zip(state.α, state.fₖ)])]
end


function value_and_gradient(f::ProximalBilinearGame, x)
    if f.setting == :Min
        subgradient!(f.game, f.gradient_cache_x, x, f.y)
        f.gradient_cache_x .+= f.stepsize * (x - f.x)
        value = eval(f.game, x, f.y) + f.stepsize / 2 * norm(x - f.x)^2
        return value, f.gradient_cache_x
    elseif f.setting == :Max
        supergradient!(f.game, f.gradient_cache_y, f.x, x)
        # Negate the gradient to convert the supergradient of the max problem to a subgradient of the min problem
        f.gradient_cache_y .-= f.stepsize * (x - f.y)
        value = -eval(f.game, f.x, x) + f.stepsize / 2 * norm(x - f.y)^2
        return value, -f.gradient_cache_y
    else
        throw(ArgumentError("Unrecognized subproblem setting: The subproblem setting should either be :Min or :Max"))
    end
end


function init_prox_step(pgame::ProximalBilinearGame, setting::Symbol)
    @assert setting == :Min || setting == :Max
    pgame.setting = setting
end

function (a::ProximalBilinearGame)(x)
    if a.setting == :Min
        return eval(a.game, x, a.y) + a.stepsize / 2 * norm(x - a.x)^2
    else
        return -eval(a.game, a.x, x) + a.stepsize / 2 * norm(x - a.y)^2
    end

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

function solve_composite!(state::ProxBundleState)
    S = hcat([
        si for si in state.sₖ
    ]...)
    h = IndSimplex()
    subprob = ProxBundleSubproblem(x0=state.x0, A=S, b=state.fₖ, h=state.g, θ=state.rho)
    state.α, _ = state.subsolver(f=subprob, Lf=norm(S)^2 * state.rho, g=h, x0=ones(size(state.sₖ, 1)) / size(state.sₖ, 1))
    state.y, _ = prox(state.g, state.x - state.rho * S * state.α)
    state.sφ_x = (state.x0 - state.y) / state.rho
end


function eval_bundle(
    game::ProximalBilinearGame,
    state::ProxBundleState
)
    value = maximum(
                [
                fk + sk' * state.y for (fk, sk) in zip(state.fₖ, state.sₖ)
            ]
            ) + 1 / (2state.rho) * norm(state.x - state.y)^2 + game.h(state.y)

    return value
end

function PDCP(game::ProximalBilinearGame, x0::Vector{Float64}, M::Float64, ε::Float64, memory::Int=10)
    # Start by building the model
    state = ProxBundleState(x=x0, rho=1 / M, g=game.h)
    fp, ∇f = value_and_gradient(game, state.x)
    state.sf_x = ∇f
    state.f_x = fp
    state.fmin = fp
    state.fₖ = [fp - state.x' * ∇f]
    state.sₖ = [∇f]
    state.eₖ = [0.0]

    while state.δₖ > ε
        println(state.δₖ)
        solve_composite!(state)
        # subproblem_qp!(state)
        # prox!(state.y, iter.g, state.y, state.rho)
        # state.εₖ = state.α' * state.eₖ
        # println("Error", state.eₖ)
        state.δₖ = state.fmin + game.h(state.x) - eval_bundle(game, state)
        println(state.δₖ)
        state.f_x = game(state.x)

        f_y, sf_y = value_and_gradient(game, state.y)

        if f_y < state.fmin
            state.x .= state.y
            state.fmin = f_y
        end
        state.descent_steps += 1

        push!(state.sₖ, sf_y)
        push!(state.fₖ, f_y - sf_y' * state.y)
        state.null_steps += 1

        bundle_management!(state, memory)

    end
end

function bundle_saddle_point(game::BilinearGame, R::Float64, target_accuracy::Float64, stepsize::Float64)
    proxgame = ProximalBilinearGame(game=game, stepsize=stepsize)

    algo = ProximalAlgorithms.ProxBundle(verbose=false, memory=5, tol=1e-10)
    # algo = ProximalAlgorithms.SFISTA(verbose=false)
    g = IndSimplex()
    anorm = norm(game.A)

    proxgame.x .= normalize(rand(game.N), 1)
    proxgame.y .= normalize(rand(game.M), 1)
    copy!(proxgame.x̃, proxgame.x)
    copy!(proxgame.ỹ, proxgame.y)
    x̄ = zeros(game.N)
    ȳ = zeros(game.M)
    for k in 1:300000

        axpby!(1 / k, proxgame.x̃, (k - 1) / k, x̄)

        axpby!(1 / k, proxgame.ỹ, (k - 1) / k, ȳ)
        if k % 1000 == 0
            println(k, " ", primalv(game, x̄), " ", dualv(game, ȳ), " ", primalv(game, x̄) - dualv(game, ȳ))
        end
        if k % 1000 == 0 && primalv(game, x̄) - dualv(game, ȳ) <= target_accuracy
            # return x, y
            # println(niter, " ", eval(problem, x, y), " ", primalv(problem, x) - dualv(problem, y), " ", maximum(problem.A * x) - minimum(problem.A' * y))
        end
        # init_prox_step(proxgame, :Min)
        # sol, steps = algo(f=proxgame, Mf=anorm + 1000, g=g, x0=copy(proxgame.x))
        # proxgame.x .= sol[1]
        # descent = sol[end-2]
        # null = sol[end-1]


        # init_prox_step(proxgame, :Min)
        # sol, steps = algo(f=proxgame, Mf=10000., g=g, x0=copy(proxgame.x))
        # proxgame.x .= sol[1]

        # proxgame.x̃ .= sol[2]
        # init_prox_step(proxgame, :Max)
        # sol, steps = algo(f=proxgame, g=g, Mf=10000., x0=copy(proxgame.y))
        proxgame.y .= sol[1]
        proxgame.ỹ .= sol[2]

        # println(y, " || ", sum(y))
        # Want 0 \in subdifferential of 
    end
    # return x, y
end


game = generate_random_zero_sum(50, 30, 0.2, (-10.0, 10.0); seed=1)
stepsize = 1.0
proxgame = ProximalBilinearGame(game=game, stepsize=stepsize)
x0 = normalize(rand(game.N), 1)
proxgame.y = normalize(rand(game.M), 1)
proxgame.x = x0
init_prox_step(proxgame, :Min)