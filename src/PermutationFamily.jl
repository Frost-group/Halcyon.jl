# PermutationFamily — conjugacy class / permutation sector representation
#
#  ≈ conjugacy class of S_N 
#  ≈ DeBois 'cycle type' and 'permutation sector' 
#  ≈ formally we exploit the isomorphism to the integer partition table of N,
#    https://discrete.openmathbooks.org/more/mdm/sec_adv-linearparts.html
#
# I know this name is not very standard, but after trying a fe different ones, I felt
# 'family' actually felt vague enough to be flexible and not get confused with other things

# Dense ranking into 1:p(N) uses the standard integer partition count table P
# to establish a bijection between C-vectors and ranks.

# ===================================================================
# Integer partition count table P[n+1, m+1]
# ===================================================================

"""
    integer_partition_count_table(nmax) -> Matrix{Int}

Dynamically-programmed table of restricted partition counts.

`P[n+1, m+1]` = number of integer partitions of `n` whose parts are all ≤ `m`.
In particular, `P[N+1, N+1]` = p(N) = total number of unrestricted partitions
(= number of conjugacy classes of Sₙ).

Recurrence: P[n,m] = P[n,m-1] + P[n-m,m]   (take ≥1 copy of part m, or skip m).
Boundary:   P[0,m] = 1 ∀m,  P[n,0] = 0 for n>0.

Ref: https://discrete.openmathbooks.org/more/mdm/sec_adv-linearparts.html
"""
function integer_partition_count_table(nmax::Int)
    P = zeros(Int, nmax + 1, nmax + 1)
    for m in 0:nmax
        P[1, m+1] = 1  # ONLY one way to partition 0: the empty partition; seeds the construction below
    end
    for n in 1:nmax
        for m in 1:nmax
            # P[n+1, m+1] = count with parts ≤ m
            #             = (partitions of n with parts ≤ m-1)     [skip m]
            #             + (partitions of n-m with parts ≤ m)     [use ≥1 copy of m]
            P[n+1, m+1] = P[n+1, m] + (n >= m ? P[n-m+1, m+1] : 0)
        end
    end
    return P
end

"""Number of permutation families (conjugacy classes) for `N` particles: p(N).
This should be an exact equivalent to the Ramachandran-Hardy approximation. 
"""
permutation_family_count(N::Int, P::Matrix{Int}) = P[N+1, N+1]

# ===================================================================
# C-vector construction
# ===================================================================

"""
    permutation_family_C(N, cycle_lengths) -> Vector{Int}

Build the cycle-count vector C from a multiset of cycle lengths.
`C[ℓ]` = number of cycles of length `ℓ`.  Satisfies `∑ ℓ C[ℓ] = N`.
"""
function permutation_family_C(N::Int, cycle_lengths::AbstractVector{Int})
    s = sum(cycle_lengths)
    s == N || throw(ArgumentError("sum(cycle_lengths)=$s != N=$N"))
    C = zeros(Int, N)
    for ℓ in cycle_lengths
        C[ℓ] += 1
    end
    return C
end

"""
    permutation_family_C(cfg::WormConfiguration) -> Vector{Int}

Extract the cycle-count vector C directly from a WormConfiguration.
Walks each permutation cycle exactly once.
"""
function permutation_family_C(cfg::WormConfiguration)
    N = size(cfg.r, 1)
    C = zeros(Int, N)
    visited = falses(N)
    @inbounds for i in 1:N
        if !visited[i]
            cyc = get_cycle(cfg, i)
            C[length(cyc)] += 1
            for p in cyc
                visited[p] = true
            end
        end
    end
    return C
end

# ===================================================================
# Ranking and unranking of C-vectors
# ===================================================================
#
# Strategy: internally convert between C-vectors and the descending-sorted
# integer partition representation (which the P table natively ranks).
# The user never sees the partition; these are private helpers.

# -- private: C-vector → descending partition --
function _C_to_partition(C::Vector{Int})
    parts = Int[]
    for ℓ in length(C):-1:1  # descending cycle length → descending partition
        for _ in 1:C[ℓ]
            push!(parts, ℓ)
        end
    end
    return parts
end

# -- private: descending partition → C-vector of length N --
function _partition_to_C(partition::Vector{Int}, N::Int)
    C = zeros(Int, N)
    for ℓ in partition
        C[ℓ] += 1
    end
    return C
end

# -- private: rank a descending partition using the P table --
# Uses reverse-lexicographic enumeration: partitions are ordered by
# first part descending, then recursively.  The P table gives the
# cumulative count of partitions whose leading part exceeds a given value.
function _partition_rank(partition::AbstractVector{Int}, n::Int, m::Int, P::Matrix{Int}; start_idx::Int=1)
    n == 0 && return 1
    a1 = partition[start_idx]
    # count partitions with leading part > a1 (they come before us)
    s = 0
    for b in (a1+1):min(n, m)
        s += P[n-b+1, b+1]
    end
    length(partition) == start_idx && return s + 1
    return s + _partition_rank(partition, n - a1, a1, P, start_idx=start_idx + 1)
end

# -- private: unrank to a descending partition using the P table --
function _partition_unrank(n::Int, m::Int, k::Int, P::Matrix{Int})
    n == 0 && return Int[]
    for b in min(n, m):-1:1
        cnt = P[n-b+1, b+1]
        if k > cnt
            k -= cnt
            continue
        end
        rest = _partition_unrank(n - b, b, k, P)
        return vcat([b], rest)
    end
    error("unrank failed: n=$n m=$m k=$k")
end

"""
    C_to_rank(C, P, N) -> Int

Dense rank in `1:p(N)` for cycle-count vector `C`.

Internally reconstructs the descending partition from `C` and ranks it
against the integer partition count table `P`.
"""
function C_to_rank(C::Vector{Int}, P::Matrix{Int}, N::Int)
    parts = _C_to_partition(C)
    _partition_rank(parts, N, N, P)
end

"""
    C_from_rank(k, N, P) -> Vector{Int}

Cycle-count vector for family rank `k ∈ 1:p(N)`.  Inverse of `C_to_rank`.

Unranks through the partition count table `P`, then converts the resulting
descending partition directly to cycle counts.
"""
function C_from_rank(k::Int, N::Int, P::Matrix{Int})
    partition = _partition_unrank(N, N, k, P)
    _partition_to_C(partition, N)
end

# ===================================================================
# DensePermutationFamilyStats — dense histogram over conjugacy classes
# ===================================================================

"""
    DensePermutationFamilyStats(N)

OK, here's the magic! 
  (OK, maybe not magic. Maybe its dumb. But anyhoo, gotta start somewhere!)
 This is a Dense set of P(N) (permutation families, perhaps more formally conjugacy classes
 of the symmetric group S_N for N particles) elements, 

'count' to count the raw number of MC obsrvations
'estimator' with a running increment of ALL observations. 
(changed to this rather than a running mean as I remembered IEEE floats are exaft in addition?) 

The size of this scales exponentially, but that exponential (O(sqrt(N))) is a HELLA LOT BETTER than N!

Name is a little historic: first I was playing with just Dict's (i.e. just hash unique cycle
defn), but thought that standarising on λ representation and reading enough
Wikipedia/Mathematica/Scary books on S_n group to get a dense object was more satisfying. 

ToDo: Think about manipulating the ranking so that most probably cycles are near each other? 
But then again, if I'm going down that fancy MC reweighting route, this is irrelevant and probably just extra complexity. 


Dense histogram over all p(N) conjugacy classes of Sₙ.

Fields:
- `N::Int`:              particle count
- `P::Matrix{Int}`:      integer partition count table (for ranking)
- `n_families::Int`:     p(N) = total number of conjugacy classes
- `count::Vector{Int64}`: observation count per sector
- `estimator::Vector{Float64}`: running sum of observables per sector
"""
struct DensePermutationFamilyStats
    N::Int
    P::Matrix{Int}
    n_families::Int
    count::Vector{Int64}
    estimator::Vector{Float64}
    estimator2::Vector{Float64}
end
# FIXME: now immutable struct, but still have count & estimator as mutable vectors?
#  Not sure if this is still a problem? Or whether once instantiated, Julia kmows everything there is to know 

function DensePermutationFamilyStats(N::Int)
    P = integer_partition_count_table(N)
    pn = P[N+1, N+1]
    DensePermutationFamilyStats(N, P, pn, zeros(Int64, pn), zeros(Float64, pn), zeros(Float64, pn))
end

"""
    observe_permutation_family!(acc, C, E)

Record an observation in sector defined by cycle-count vector `C`, accumulating
the observable value `E`.  Dividing `estimator[k]` by `count[k]` gives the mean.
"""
function observe_permutation_family!(acc::DensePermutationFamilyStats, C::Vector{Int}, E::Float64)
    k = C_to_rank(C, acc.P, acc.N)
    acc.count[k] += 1
    acc.estimator[k] += E
    acc.estimator2[k] += E^2
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

Can also be seen as the probability of the permutation family, n the
high-temperature / classical limit (T=∞), where all the quantumness drops away
and we are just left with the degeneracy of the microstates.  
"""
function log_multiplicities(stats::DensePermutationFamilyStats)
    N = stats.N
    logM = zeros(Float64, stats.n_families)
    logN_fact = sum(log.(1:N))
    for k in 1:stats.n_families
        C_k = C_from_rank(k, N, stats.P)
        logM[k] = logN_fact
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
        C[k, :] .= C_from_rank(k, N, stats.P) # much more simple now!
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

Base.show(io::IO, m::MultiplicityModel) = print(io, "MultiplicityModel(N=$(length(m.θ)), θ=$(round.(m.θ, digits=2)))")
Base.show(io::IO, m::DuBoisModel) = print(io, "DuBoisModel(N=$(length(m.θ)), κ=$(round(m.κ, digits=4)), p₂=$(round(exp(-m.κ), digits=4)), θ=$(round.(m.θ, digits=2)))")
Base.show(io::IO, m::MaxEntModel) = print(io, "MaxEntModel(N=$(length(m.θ)), l2_reg=$(m.l2_reg), θ=$(round.(m.θ, digits=2)))")
Base.show(io::IO, m::MAPHybridModel) = print(io, "MAPHybridModel(N=$(length(m.θ)), τ₀=$(round(m.τ₀, digits=2)), prior=$(m.prior), θ=$(round.(m.θ, digits=2)))")

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
        G .= ((q'*Cmat).-(emp_p'*Cmat))[2:end] .+ 2 * l2_reg .* θr
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
        G .= ((q'*Cmat).-(emp_p'*Cmat))[2:end] .+ [δ[ℓ] / τ₀^2 for ℓ in 2:N]
    end

    res = optimize(f, g!, zeros(N - 1), LBFGS())
    δ = vcat(0.0, minimizer(res))
    θ = θ_prior .+ δ
    MAPHybridModel(prior, τ₀, δ, θ)
end


