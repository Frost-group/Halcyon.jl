# Definition of some simple testing potentials.

using LinearAlgebra # for norm

function Harmonic(r::AbstractVector{<:Real}; k=1)
    r′ = norm(r)
    return k*r′^2
end

function DoubleWell(r::AbstractVector{<:Real}; A=10, B=1)
    r′ = norm(r)
    return A*(r′^4 - r′^2) - B*r′^2
end

"""
    LennardJones(r::AbstractVector{<:Real}; ϵ=1.0, σ=1.0)

Lennard-Jones potential. 
- ϵ is depth of potential
- σ is distance at which potential is zero (NOT minimum of well)

"""
function LennardJones(r::AbstractVector{<:Real}; ϵ=1.0, σ=1.0)
    r′ = norm(r)
    return 4ϵ * ((σ / r′)^12 - (σ / r′)^6)
end

"""
    Coulomb(r::AbstractVector{<:Real}; g=1.0)

Coulomb potential.
Strength g. (Same as Yukawa.)
[Maybe we should just chain call Yukawa with m=0.0?]
"""
function Coulomb(r::AbstractVector{<:Real}; g=1.0)
    r′ = norm(r)
    if r′==0.0 return 0.0 end # hack for avoiding infinite self energy 
    return -g / r′
end

"""
    Yukawa(r::AbstractVector{<:Real}; g=1.0, m=1.0)

Yukawa potential, also known as the screened Coulomb potential.

The Yukawa potential is given by:

    U(r) = -g * exp(-m * r) / r

where:
- g is the strength of the potential
- m is the screening mass or inverse screening length
- r is the distance between the particles
"""
function Yukawa(r::AbstractVector{<:Real}; g=1.0, m=1.0)
    r′ = norm(r)
    return -g * exp(-m * r′) / r′
end

