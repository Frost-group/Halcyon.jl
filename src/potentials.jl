using LinearAlgebra # for norm

abstract type AbstractPotential end

abstract type ExternalPotential <: AbstractPotential end

abstract type PairPotential <: AbstractPotential end

(U::PairPotential)(r_norm::Float64) = U([r_norm, 0.0, 0.0]) # Fallback to vector call

Base.@kwdef struct HarmonicPotential <: ExternalPotential
    k::Float64 = 1.0
end

(V::HarmonicPotential)(r_sq::Float64) = V.k * r_sq
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

(U::LennardJonesPotential)(r_norm::Float64) = 4U.ϵ * ((U.σ / r_norm)^12 - (U.σ / r_norm)^6)
(U::LennardJonesPotential)(r::AbstractVector{<:Real}) = begin
    r′ = norm(r)
    4U.ϵ * ((U.σ / r′)^12 - (U.σ / r′)^6)
end

Base.show(io::IO, obj::LennardJonesPotential) = print(io, "LennardJonesPotential(ϵ=$(obj.ϵ), σ=$(obj.σ))")

Base.@kwdef struct CoulombPotential <: PairPotential
    g::Float64 = 1.0
end

(U::CoulombPotential)(r_norm::Float64) = r_norm == 0.0 ? 0.0 : -U.g / r_norm
(U::CoulombPotential)(r::AbstractVector{<:Real}) = begin
    r′ = norm(r)
    r′ == 0.0 ? 0.0 : -U.g / r′
end

Base.show(io::IO, obj::CoulombPotential) = print(io, "CoulombPotential(g=$(obj.g))")

Base.@kwdef struct YukawaPotential <: PairPotential
    g::Float64 = 1.0
    m::Float64 = 1.0
end

(U::YukawaPotential)(r_norm::Float64) = -U.g * exp(-U.m * r_norm) / r_norm
(U::YukawaPotential)(r::AbstractVector{<:Real}) = begin
    r′ = norm(r)
    -U.g * exp(-U.m * r′) / r′
end

Base.show(io::IO, obj::YukawaPotential) = print(io, "YukawaPotential(g=$(obj.g), m=$(obj.m))")

Base.@kwdef struct HardSpherePotential <: PairPotential
    a::Float64 = 1.0
end

(U::HardSpherePotential)(r_norm::Float64) = r_norm < U.a ? Inf : 0.0
(U::HardSpherePotential)(r::AbstractVector{<:Real}) = norm(r) < U.a ? Inf : 0.0

Base.show(io::IO, obj::HardSpherePotential) = print(io, "HardSpherePotential(a=$(obj.a))")

"""
    AzizPotential (HFD-B)

Helium-4 Aziz potential (HFD-B form).
Ref: Aziz et al., 1987, Mol. Phys. 61, 1487.
Units: K for energy, Å for distance.
"""
@kwdef struct AzizPotential <: PairPotential
    ϵ::Float64 = 10.948
    rm::Float64 = 2.963
    A::Float64 = 1.84431e5
    α::Float64 = 10.43329
    β::Float64 = -2.27965
    c6::Float64 = 1.3674521
    c8::Float64 = 0.4212356
    c10::Float64 = 0.1747339
    D::Float64 = 1.4826
end

function (U::AzizPotential)(r::Float64)
    x = r / U.rm
    # Damping function
    f_d = x < U.D ? exp(-(U.D / x - 1)^2) : 1.0
    attr = (U.c6 / x^6 + U.c8 / x^8 + U.c10 / x^10) * f_d
    rep = U.A * exp(-U.α * x + U.β * x^2)
    return U.ϵ * (rep - attr)
end

function (U::AzizPotential)(r::AbstractVector{<:Real})
    return U(norm(r))
end

Base.show(io::IO, obj::AzizPotential) = print(io, "AzizPotential(ϵ=$(obj.ϵ), rm=$(obj.rm))")

"""
    cao_berne_ratio(r1, r2, dot_product, a, λ, δτ) -> Float64

Cao-Berne approximation for the relative density matrix ratio ρ_rel/ρ_rel0
for the hard-sphere potential with diameter a.
Ref: Spada et al. 2022, Eq. 56.
"""
@inline function cao_berne_ratio(r1::Float64, r2::Float64, dot_product::Float64, a::Float64, λ::Float64, δτ::Float64)
    if r1 <= a || r2 <= a
        return 0.0
    end

    cos_θ = dot_product / (r1 * r2)
    cos_θ = clamp(cos_θ, -1.0, 1.0)

    exponent = -(r1 * r2 + a^2 - a * (r1 + r2)) * (1.0 + cos_θ) / (4.0 * λ * δτ)

    term = (a * (r1 + r2) - a^2) / (r1 * r2)
    ratio = 1.0 - term * exp(exponent)

    return max(0.0, ratio)
end

"""
    cao_berne_derivatives(r1, r2, dot_product, a, λ, δτ) -> (du_dr1, du_dr2, du_dcos, du_dδτ)

Derivatives of the Cao-Berne action u = -log(ρ_rel/ρ_rel0).
Returns derivatives with respect to |r1|, |r2|, cos(θ), and δτ.
"""
@inline function cao_berne_derivatives(r1::Float64, r2::Float64, dot_product::Float64, a::Float64, λ::Float64, δτ::Float64)
    cos_θ = dot_product / (r1 * r2)
    cos_θ = clamp(cos_θ, -1.0, 1.0)

    C = 4.0 * λ * δτ
    K = r1 * r2 + a^2 - a * (r1 + r2)
    B = (a * (r1 + r2) - a^2) / (r1 * r2)

    expr = exp(-K * (1.0 + cos_θ) / C)
    ratio = 1.0 - B * expr

    if ratio <= 0.0
        return (Inf, Inf, Inf, Inf)
    end

    common = expr / ratio

    dB_dr1 = (a^2 - a * r2) / (r1^2 * r2)
    dB_dr2 = (a^2 - a * r1) / (r2^2 * r1)

    dExp_dr1 = -(r2 - a) * (1.0 + cos_θ) / C
    dExp_dr2 = -(r1 - a) * (1.0 + cos_θ) / C
    dExp_dcos = -K / C
    dExp_dδτ = K * (1.0 + cos_θ) / (C * δτ)

    du_dr1 = common * (dB_dr1 + B * dExp_dr1)
    du_dr2 = common * (dB_dr2 + B * dExp_dr2)
    du_dcos = common * (B * dExp_dcos)
    du_dδτ = common * (B * dExp_dδτ)

    return (du_dr1, du_dr2, du_dcos, du_dδτ)
end

struct NullPairPotential <: PairPotential end

(::NullPairPotential)(r_norm::Float64) = 0.0
(::NullPairPotential)(r::AbstractVector{<:Real}) = 0.0

Base.show(io::IO, obj::NullPairPotential) = print(io, "NullPairPotential()")

