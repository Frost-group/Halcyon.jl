using StaticArrays

Base.@kwdef struct System
    M::Int
    N::Int
    m::Float64 = 1.0
    D::Int = 3
    β::Float64 = 100.0
    V::ExternalPotential = HarmonicPotential()
    U::PairPotential = NullPairPotential()
    λ::Float64 = 0.5
    τ::Float64 = 0.0   # imaginary time step; set via validated constructors
    L::Float64 = 1.0   # box length; assumes cubic box for now
end

function System(M::Integer, N::Integer; m::Float64=1.0, D::Integer=3, β::Float64=100.0,
                V::ExternalPotential=HarmonicPotential(), U::PairPotential=NullPairPotential(),
                λ::Float64=0.5, τ::Union{Nothing,Float64}=nothing, L::Float64=1.0)
    M ≤ 0 && throw(ArgumentError("M must be positive"))
    N ≤ 0 && throw(ArgumentError("N must be positive"))
    D ≤ 0 && throw(ArgumentError("D must be positive"))
    β ≤ 0 && throw(ArgumentError("β must be positive"))
    λ ≤ 0 && throw(ArgumentError("λ must be positive"))
    τval = isnothing(τ) ? β/Int(M) : τ
    τval ≤ 0 && throw(ArgumentError("τ must be positive"))
    return System(M=Int(M), N=Int(N), m=m, D=Int(D), β=β, V=V, U=U, λ=λ, τ=τval, L=L)
end

Base.show(io::IO, s::System) = print(io,
    "PIMC system with $(s.M) beads and $(s.N) particles. \n" *
    "β=$(s.β), τ=$(s.τ). V=$(s.V) U=$(s.U)\n" *
    "Particle m of $(s.m), λ=$(s.λ). $(s.D) spatial dimensions." *
    "Box length $(s.L).")

Base.@kwdef mutable struct Path
    r::Array{Float64,3}
    next::Array{Int,2}
    prev::Array{Int,2}
end

function Path(S::System)
    r = randn(S.N, S.M, S.D)
    # Identity particle mapping across beads
    ids = reshape(collect(1:S.N), S.N, 1)
    list = repeat(ids, 1, S.M)
    return Path(r=r, next=list, prev=list)
end

function Base.show(io::IO, p::Path)
    print(io, "Path $(p.r) \n Next $(p.next) \n Prev $(p.prev)")
end
