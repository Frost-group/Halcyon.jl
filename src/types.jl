using StaticArrays

Base.@kwdef struct System{TV<:ExternalPotential,TU<:PairPotential}
    M::Int
    N::Int
    m::Float64 = 1.0
    D::Int = 3
    β::Float64 = 100.0
    V::TV = HarmonicPotential()
    U::TU = NullPairPotential()
    λ::Float64 = 0.5
    τ::Float64 = 0.0   # imaginary time step; set via validated constructors
    L::Float64 = 1.0   # box length; assumes cubic box for now
end

function System(M::Integer, N::Integer; m::Float64=1.0, D::Integer=3, β::Float64=100.0,
    V::TV=HarmonicPotential(), U::TU=NullPairPotential(),
    λ::Float64=0.5, τ::Union{Nothing,Float64}=nothing, L::Float64=1.0) where {TV<:ExternalPotential,TU<:PairPotential}
    M ≤ 0 && throw(ArgumentError("M must be positive"))
    N ≤ 0 && throw(ArgumentError("N must be positive"))
    D ≤ 0 && throw(ArgumentError("D must be positive"))
    β ≤ 0 && throw(ArgumentError("β must be positive"))
    λ ≤ 0 && throw(ArgumentError("λ must be positive"))
    τval = isnothing(τ) ? β / Int(M) : τ
    τval ≤ 0 && throw(ArgumentError("τ must be positive"))
    return System{TV,TU}(M=Int(M), N=Int(N), m=m, D=Int(D), β=β, V=V, U=U, λ=λ, τ=τval, L=L)
end

Base.show(io::IO, s::System) = print(io,
    "PIMC system with $(s.M) beads and $(s.N) particles. \n" *
    "β=$(s.β), τ=$(s.τ). V=$(s.V) U=$(s.U)\n" *
    "Particle m of $(s.m), λ=$(s.λ). $(s.D) spatial dimensions." *
    "Box length $(s.L).")

# ═══════════════════════════════════════════════════════════════════════════════
# Worm Algorithm Types (Spada et al. 2022)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Sector indicator for worm algorithm.
- Z_SECTOR: Diagonal configurations (closed polymers)
- G_SECTOR: Off-diagonal configurations (worm present with head/tail)

Ref: Spada et al. 2022, Section II.C
"""
@enum Sector Z_SECTOR G_SECTOR

"""
    WormConfiguration

Worm algorithm configuration for PIMC with periodic boundary conditions.

Ref: Spada et al. 2022, Section II.C

Fields:
- sector: Z (diagonal/closed) or G (off-diagonal/worm present)
- next: Forward permutation vector, next[i] = particle that polymer i connects to
- prev: Backward permutation vector, prev[j] = particle that connects to polymer j
- i_head: Particle index of worm head (only meaningful in G-sector)
- i_tail: Particle index of worm tail = next[i_head] (meaningful in G-sector)
- r_c_head: Compact head coordinate in [0, L)^D
- r: Bead positions [particle, bead, dimension], 1:M for beads 0:M-1
- w: Winding numbers [particle, dimension]
"""
@kwdef mutable struct WormConfiguration
    sector::Sector = Z_SECTOR
    next::Vector{Int}
    prev::Vector{Int}
    i_head::Int = 1
    i_tail::Int = 1
    r_c_head::Vector{Float64} = Float64[]
    r::Array{Float64,3}
    w::Matrix{Int}
end

"""
    WormParams

Monte Carlo parameters for worm algorithm moves.

Fields:
- C: Open/close swap ratio (higher C → prefer Z-sector)
- j_max: Maximum segment length for moves
- r_max: Maximum displacement for moves
"""
@kwdef struct WormParams
    C::Float64 = 1.0      # Open/close swap ratio
    j_max::Int = 50       # Max segment length
    r_max::Float64 = 2.0  # Max displacement
end
