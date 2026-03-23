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
"""
function recursive_partition_rank(partition::AbstractVector{Int}, n::Int, m::Int, P::Matrix{Int})
    n == 0 && return 1
    
    a1 = partition[1]
    hi = min(n, m)
    s = 0
    for b in (a1 + 1):hi
        s += P[n - b + 1, b + 1]
    end
    if length(partition) == 1
        return s + 1
    end
    return s + recursive_partition_rank(partition[2:end], n - a1, a1, P)
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

