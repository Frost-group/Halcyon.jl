# PermutationFamily code, sort of following DuBois, or at least what I assume they did
#  
#  ≈ conjugacy class of S_N 
#  ≈ DeBois 'cycle type' and 'permutation sector' (though Nb: we use λ notation)
#  ≈ formally we exploit the isomorphism to the integer partition of N
#
# I know this name is not very standard, but after trying a fe different ones, I felt
# 'family' actually felt vague enough to be flexible and not get confused with other things
#
# λ 'integar partition' is the canonical form used in this code: 
#   Vector{Int} length N (particle count), 
#   nonnegative, 
#   sum(λ)=N.



"""
  integer_partition_count_table: integer_partition_count_table(nmax::Int)
  
  returnss P[n+1,m+1] = #{integer partitions of n with parts ≤ m}.
  
  See https://discrete.openmathbooks.org/more/mdm/sec_adv-linearparts.html
"""
function integer_partition_count_table(nmax::Int)
    P = zeros(Int, nmax + 1, nmax + 1)
    for m in 0:nmax
        P[1, m + 1] = 1
    end
    for n in 1:nmax
        for m in 1:nmax
            P[n + 1, m + 1] = P[n + 1, m] + (n >= m ? P[n - m + 1, m + 1] : 0)
        end
    end
    return P
end

"""Number of permutation families (conjugacy classes) for `N` particles: p(N)."""
permutation_family_count(N::Int, P::Matrix{Int}) = P[N + 1, N + 1]

"""
    permutation_family_lambda(N, cycle_lengths) -> Vector{Int}

Padded λ from multiset of cycle lengths (`sum(cycle_lengths) == N`). Sorts descending, pads to length `N`.
`N` is the particle count (e.g. `sys.N`).
"""
function permutation_family_lambda(N::Int, cycle_lengths::AbstractVector{Int})
    s = sum(cycle_lengths)
    s == N || throw(ArgumentError("sum(cycle_lengths)=$s != N=$N"))
    ls = sort!(collect(cycle_lengths); rev=true)
    λ = zeros(Int, N)
    lp = length(ls)
    λ[1:lp] .= ls
    return λ
end

"""
    permutation_family_lambda(cfg::WormConfiguration) -> Vector{Int}

Padded λ for the current permutation on `cfg` (`N = size(cfg.r,1)`).
"""
function permutation_family_lambda(cfg::WormConfiguration)
    N = size(cfg.r, 1) # bit of a hack? 
    λ = zeros(Int, N) # emptied, so we can accumulate as we stripe
    visited = falses(N)
    
    @inbounds for i in 1:N
        if !visited[i]
            cyc = get_cycle(cfg, i)
            λ[i]=length(cyc) # stash dah number
            for p in cyc
                visited[p] = true
            end
        end
    end
    # so now we have Lambda, with lots of zeros if big cycles 

    sort!(λ; rev=true) # TA DA
    return λ
end

"""
    C_permutation_sector(λ) -> Vector{Int}

DuBois-style counts ``C_ℓ`` (cycles of length ℓ), 
`Vector` of length `N`, but now _element of vector_ states how big the cycle is. 

So ΣC[i]*i==N
(Wheres Σλ=N)
"""
function C_permutation_sector(λ::Vector{Int})
    N = length(λ)
    C = zeros(Int, N)
    for i in 1:N
        m = λ[i]
        m == 0 && break
        C[m] += 1
    end
    return C
end


"""Uses a reverse-lexigraphic ordering of canonical partition (largest first, like European format or something?). Follows Matehmatica IntegerPartitions order.

This seems to follow DeBois; from looking at the comparitive figures in the SI. 

BUT it seems a bit stupid, as it ranks stuff lexigraphically, it goes like [6], [5,1], [4,2], etc. so has quite common cycles spread through the index. 

I had a thought to re-order this to be more physical, by following the kind DeBois reasoning that larger cycles should be rare so pushing them to the end of the list, meaning like 99.9% of the MC moves fit in the L1 cache (till you go superfluid!)
"""
function recursive_partition_rank(partition::AbstractVector{Int}, n::Int, m::Int, P::Matrix{Int}; start_idx::Int=1)
    n == 0 && return 1
    
    a1 = partition[start_idx]
    hi = min(n, m)
    s = 0
    for b in (a1 + 1):hi
        s += P[n - b + 1, b + 1]
    end
    if length(partition) == start_idx
        return s + 1
    end
    return s + recursive_partition_rank(partition, n - a1, a1, P, start_idx=start_idx+1)
end

"""
    permutation_family_index(λ, P, N) -> Int

Dense index into `1:permutation_family_count(N,P)` via recursive ranking of canonical partition (padded λ).
"""
function permutation_family_index(λ::Vector{Int}, P::Matrix{Int}, N::Int)
    r = findfirst(iszero, λ)
    partition = isnothing(r) ? view(λ, 1:N) : view(λ, 1:(r - 1))
    recursive_partition_rank(partition, N, N, P)
end

function recursive_partition_unrank(n::Int, m::Int, k::Int, P::Matrix{Int})
    n == 0 && return Int[]
    for b in min(n, m):-1:1
        cnt = P[n - b + 1, b + 1]
        if k > cnt
            k -= cnt
            continue
        end
        rest = recursive_partition_unrank(n - b, b, k, P)
        return vcat([b], rest)
    end
    error("unrank failed")
end

"""Padded λ for family `k` ∈ `1:p(N)` (inverse of `permutation_family_index`)."""
function permutation_family_lambda_from_rank(k::Int, N::Int, P::Matrix{Int})
    partition = recursive_partition_unrank(N, N, k, P)
    λ = zeros(Int, N)
    lp = length(partition)
    λ[1:lp] .= partition
    return λ
end

"""
    DensePermutationFamilyStats(N::Int)

OK, here's the magic! 
  (OK, maybe not magic. Maybe its dumb. But anyhoo, gotta start somewhere!)
 This is a Dense set of P(N) (permutation families, perhaps more formally conjugacy classes
 of the symmetric group S_N for N particles) elements, 

'observations' to count the number of contributions
'estimator' with a running mean (Welford-style incremental mean), which sounds more fancy than it is. 

The size of this scales exponentially, but that exponential (O(sqrt(N))) is a HELLA LOT BETTER than N!

Name is a little historic: first I was playing with just Dict's (i.e. just hash unique cycle
defn), but thought that standarising on λ representation and reading enough
Wikipedia/Mathematica/Scary books on S_n group to get a dense object was more satisfying. 

"""
 struct DensePermutationFamilyStats
    N::Int
    P::Matrix{Int}
    n_families::Int
    count::Vector{Int64}
    estimator::Vector{Float64}
end
# FIXME: now immutable struct, but still have count & estimator as mutable vectors?
#  Not sure if this is still a problem? Or whether once instantiated, Julia kmows everything there is to know 

function DensePermutationFamilyStats(N::Int)
    P = integer_partition_count_table(N)
    pn = P[N + 1, N + 1]
    DensePermutationFamilyStats(N, P, pn, zeros(Int64, pn), zeros(Float64, pn))
end

""" Just a mock to play with the code appriach currently"""
function observe_permutation_family!(acc::DensePermutationFamilyStats, λ::Vector{Int}, x::Float64)
    N = acc.N
    k = permutation_family_index(λ, acc.P, N)
    acc.count[k] += 1
    t = acc.count[k]
    acc.estimator[k] += (x - acc.estimator[k]) / t
end

# ===================================================================
# Accessor functions on DensePermutationFamilyStats
# ===================================================================

using LinearAlgebra: dot
using Optim: optimize, minimizer, NelderMead, LBFGS

"""
    log_multiplicities(stats) -> Vector{Float64}

log M_k = log(N! / ∏_ℓ (ℓ^{C_ℓ} C_ℓ!)) for each of the p(N) permutation families.
This is the combinatorial weight of sector k in S_N.
"""
function log_multiplicities(stats::DensePermutationFamilyStats)
    N = stats.N
    logM = zeros(Float64, stats.n_families)
    logN! = sum(log.(1:N))
    for k in 1:stats.n_families
        λk = permutation_family_lambda_from_rank(k, N, stats.P)
        C_k = C_permutation_sector(λk)
        logM[k] = logN!
        @inbounds for ℓ in 1:N
            C_k[ℓ] > 0 && (logM[k] -= sum(log.(1:C_k[ℓ])) + C_k[ℓ] * log(ℓ))
        end
    end
    logM
end

"""
    cycle_count_matrix(stats) -> Matrix{Int}

(n_families × N) matrix where C[k,ℓ] = number of ℓ-cycles in permutation family k.
"""
function cycle_count_matrix(stats::DensePermutationFamilyStats)
    N = stats.N
    C = zeros(Int, stats.n_families, N)
    for k in 1:stats.n_families
        λk = permutation_family_lambda_from_rank(k, N, stats.P)
        C_k = C_permutation_sector(λk)
        C[k, :] .= C_k
    end
    C
end

"""
    empirical_probabilities(stats) -> Vector{Float64}

Normalised empirical histogram P̂(k) = count(k) / Σcount.
"""
function empirical_probabilities(stats::DensePermutationFamilyStats)
    n_tot = sum(stats.count)
    n_tot == 0 && error("No observations recorded")
    Float64.(stats.count) ./ n_tot
end

# ===================================================================
# Permutation sector models: exponential family Q(k) ∝ M_k exp(C_k⋅θ)
# ===================================================================

abstract type AbstractPermutationModel end

"""Infinite-T limit: Q(k) ∝ M_k. All θ = 0 (no exchange penalties)."""
struct MultiplicityModel <: AbstractPermutationModel
    θ::Vector{Float64}
end

"""DuBois 1-parameter: θ_ℓ = -κ(ℓ-1). Penalises non-trivial exchanges via a
single pair-exchange penalty κ. The sector weight is M_k exp[-κ(N-K)] where
K = total number of cycles."""
struct DuBoisModel <: AbstractPermutationModel
    κ::Float64
    θ::Vector{Float64}
end

"""Full MaxEnt: unconstrained θ₂…θ_N with weak isotropic L2 regularisation.
Maximises entropy subject to matching empirical cycle-count expectations ⟨C_ℓ⟩."""
struct MaxEntModel <: AbstractPermutationModel
    l2_reg::Float64
    θ::Vector{Float64}
end

"""MAP hybrid: DuBois baseline + ridge-regularised deviations δ_ℓ.
θ_ℓ = θ_ℓ^DuBois + δ_ℓ, with prior δ_ℓ ~ N(0, τ₀²).
At short cycles, data dominates; at long cycles, the prior pulls back to the
DuBois exponential decay rather than letting θ → -∞."""
struct MAPHybridModel{P<:AbstractPermutationModel} <: AbstractPermutationModel
    prior::P
    τ₀::Float64
    δ::Vector{Float64}
    θ::Vector{Float64}
end

Base.show(io::IO, m::MultiplicityModel)  = print(io, "MultiplicityModel(N=$(length(m.θ)), θ=$(round.(m.θ, digits=2)))")
Base.show(io::IO, m::DuBoisModel)        = print(io, "DuBoisModel(N=$(length(m.θ)), κ=$(round(m.κ, digits=4)), p₂=$(round(exp(-m.κ), digits=4)), θ=$(round.(m.θ, digits=2)))")
Base.show(io::IO, m::MaxEntModel)        = print(io, "MaxEntModel(N=$(length(m.θ)), l2_reg=$(m.l2_reg), θ=$(round.(m.θ, digits=2)))")
Base.show(io::IO, m::MAPHybridModel)     = print(io, "MAPHybridModel(N=$(length(m.θ)), τ₀=$(round(m.τ₀, digits=2)), prior=$(m.prior), θ=$(round.(m.θ, digits=2)))")

# ===================================================================
# Shared helpers
# ===================================================================

"""Normalised model distribution Q(k) for a given θ, using log-sum-exp for numerical stability."""
function _normalised_q(θ::Vector{Float64}, logM::Vector{Float64}, C::Matrix{Int})
    lu = logM .+ C * θ
    mx = maximum(lu)
    exp.(lu .- mx .- log(sum(exp.(lu .- mx))))
end

"""KL(P̂ ‖ Q) in nats, skipping sectors where P̂(k)=0."""
function kl_divergence(p::Vector{Float64}, q::Vector{Float64})
    s = 0.0
    @inbounds for i in eachindex(p)
        p[i] > 0 && (s += p[i] * log(p[i] / q[i]))
    end
    s
end

# ===================================================================
# probabilities / kl_divergence dispatched on model + stats
# ===================================================================

"""Model distribution Q(k) for all p(N) sectors."""
function probabilities(model::AbstractPermutationModel, stats::DensePermutationFamilyStats)
    _normalised_q(model.θ, log_multiplicities(stats), cycle_count_matrix(stats))
end

"""KL(P̂ ‖ Q_model) in nats."""
function kl_divergence(model::AbstractPermutationModel, stats::DensePermutationFamilyStats)
    kl_divergence(empirical_probabilities(stats), probabilities(model, stats))
end

# ===================================================================
# fit(ModelType, stats; kwargs...) -> fitted model
# ===================================================================

function fit(::Type{MultiplicityModel}, stats::DensePermutationFamilyStats)
    MultiplicityModel(zeros(stats.N))
end

"""Fit scalar κ by minimising NLL via Nelder-Mead."""
function fit(::Type{DuBoisModel}, stats::DensePermutationFamilyStats)
    N = stats.N
    logM = log_multiplicities(stats)
    Cmat = cycle_count_matrix(stats)
    emp_p = empirical_probabilities(stats)

    # n_minus_K[k] = N - K_k = min transpositions from identity
    n_minus_K = Float64[N - sum(Cmat[k, :]) for k in 1:stats.n_families]

    function nll(κ_vec)
        κ = κ_vec[1]
        lu = logM .- κ .* n_minus_K
        mx = maximum(lu)
        log(sum(exp.(lu .- mx))) + mx - dot(emp_p, lu)
    end

    res = optimize(nll, [0.0], NelderMead())
    κ_hat = minimizer(res)[1]
    θ = [-(ℓ - 1) * κ_hat for ℓ in 1:N]
    DuBoisModel(κ_hat, θ)
end

"""Fit N-1 unconstrained cycle-length penalties θ₂…θ_N via L-BFGS, with weak L2 safeguard."""
function fit(::Type{MaxEntModel}, stats::DensePermutationFamilyStats; l2_reg::Float64=1e-4)
    N = stats.N
    logM = log_multiplicities(stats)
    Cmat = cycle_count_matrix(stats)
    emp_p = empirical_probabilities(stats)

    function f(θr)
        θ = vcat(0.0, θr)
        lu = logM .+ Cmat * θ
        mx = maximum(lu)
        log(sum(exp.(lu .- mx))) + mx - dot(emp_p, lu) + l2_reg * sum(θr .^ 2)
    end

    function g!(G, θr)
        θ = vcat(0.0, θr)
        lu = logM .+ Cmat * θ
        mx = maximum(lu)
        unnorm = exp.(lu .- mx)
        q = unnorm ./ sum(unnorm)
        G .= ((q' * Cmat) .- (emp_p' * Cmat))[2:end] .+ 2 * l2_reg .* θr
    end

    res = optimize(f, g!, zeros(N - 1), LBFGS())
    θ = vcat(0.0, minimizer(res))
    MaxEntModel(l2_reg, θ)
end

"""Fit deviations δ₂…δ_N from a prior model's θ via L-BFGS with Gaussian ridge."""
function fit(::Type{MAPHybridModel}, stats::DensePermutationFamilyStats;
             prior::AbstractPermutationModel, τ₀::Float64=1.0)
    N = stats.N
    logM = log_multiplicities(stats)
    Cmat = cycle_count_matrix(stats)
    emp_p = empirical_probabilities(stats)
    θ_prior = prior.θ

    function f(δr)
        δ = vcat(0.0, δr)
        θ = θ_prior .+ δ
        lu = logM .+ Cmat * θ
        mx = maximum(lu)
        logZ = log(sum(exp.(lu .- mx))) + mx
        reg = sum(δ[ℓ]^2 for ℓ in 2:N) / (2 * τ₀^2)
        logZ - dot(emp_p, lu) + reg
    end

    function g!(G, δr)
        δ = vcat(0.0, δr)
        θ = θ_prior .+ δ
        lu = logM .+ Cmat * θ
        mx = maximum(lu)
        unnorm = exp.(lu .- mx)
        q = unnorm ./ sum(unnorm)
        G .= ((q' * Cmat) .- (emp_p' * Cmat))[2:end] .+ [δ[ℓ] / τ₀^2 for ℓ in 2:N]
    end

    res = optimize(f, g!, zeros(N - 1), LBFGS())
    δ = vcat(0.0, minimizer(res))
    θ = θ_prior .+ δ
    MAPHybridModel(prior, τ₀, δ, θ)
end


