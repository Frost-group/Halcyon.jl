using LinearAlgebra # for norm

abstract type AbstractPotential end

abstract type ExternalPotential <: AbstractPotential end

abstract type PairPotential <: AbstractPotential end

Base.@kwdef struct HarmonicPotential <: ExternalPotential
    k::Float64 = 1.0
end

(V::HarmonicPotential)(r::AbstractVector{<:Real}) = V.k * (norm(r)^2)

Base.show(io::IO, obj::HarmonicPotential) = print(io, "HarmonicPotential(k=$(obj.k))")

Base.@kwdef struct DoubleWellPotential <: ExternalPotential
    A::Float64 = 10.0
    B::Float64 = 1.0
end

(V::DoubleWellPotential)(r::AbstractVector{<:Real}) = begin
    r′ = norm(r)
    V.A * (r′^4 - r′^2) - V.B * r′^2
end

Base.show(io::IO, obj::DoubleWellPotential) = print(io, "DoubleWellPotential(A=$(obj.A), B=$(obj.B))")

Base.@kwdef struct LennardJonesPotential <: PairPotential
    ϵ::Float64 = 1.0
    σ::Float64 = 1.0
end

(U::LennardJonesPotential)(r::AbstractVector{<:Real}) = begin
    r′ = norm(r)
    4U.ϵ * ((U.σ / r′)^12 - (U.σ / r′)^6)
end

Base.show(io::IO, obj::LennardJonesPotential) = print(io, "LennardJonesPotential(ϵ=$(obj.ϵ), σ=$(obj.σ))")

Base.@kwdef struct CoulombPotential <: PairPotential
    g::Float64 = 1.0
end

(U::CoulombPotential)(r::AbstractVector{<:Real}) = begin
    r′ = norm(r)
    r′ == 0.0 ? 0.0 : -U.g / r′
end

Base.show(io::IO, obj::CoulombPotential) = print(io, "CoulombPotential(g=$(obj.g))")

Base.@kwdef struct YukawaPotential <: PairPotential
    g::Float64 = 1.0
    m::Float64 = 1.0
end

(U::YukawaPotential)(r::AbstractVector{<:Real}) = begin
    r′ = norm(r)
    -U.g * exp(-U.m * r′) / r′
end

Base.show(io::IO, obj::YukawaPotential) = print(io, "YukawaPotential(g=$(obj.g), m=$(obj.m))")

struct NullPairPotential <: PairPotential end

(::NullPairPotential)(r::AbstractVector{<:Real}) = 0.0

Base.show(io::IO, obj::NullPairPotential) = print(io, "NullPairPotential()")

 