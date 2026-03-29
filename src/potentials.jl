using LinearAlgebra # for norm
using ForwardDiff

# -----------------------------------------------------------------------------
# Abstract Types & Generic Interfaces
# -----------------------------------------------------------------------------

abstract type AbstractPotential end
abstract type ExternalPotential <: AbstractPotential end
abstract type PairPotential <: AbstractPotential end

# Fallback: if called with a scalar, assume it's |r| and convert to vector call if needed,
(U::PairPotential)(r_norm::Float64) = U([r_norm, 0.0, 0.0])

"""
    potential_derivative(V::AbstractPotential, r) -> Vector

Return the gradient ∇V(r) of the potential.
Fallback uses ForwardDiff for automatic differentiation.
"""
function potential_derivative(V::AbstractPotential, r::AbstractVector{<:Real})
    @warn "ForwardDiff fallback for $(typeof(V)). Expect SEVERE SLOW DOWN."
    return ForwardDiff.gradient(V, r)
end

"""
    potential_derivative!(out, V::AbstractPotential, r)

In-place evaluation of the gradient ∇V(r), storing the result in `out`.
Fallback calls `potential_derivative` and copies.
"""

function potential_derivative!(out::AbstractVector, V::AbstractPotential, r::AbstractVector{<:Real})
    out .= potential_derivative(V, r)
    return out
end

# -----------------------------------------------------------------------------
# Null Potential
# -----------------------------------------------------------------------------

struct NullPairPotential <: PairPotential end

(::NullPairPotential)(r_norm::Float64) = 0.0
(::NullPairPotential)(r::AbstractVector{<:Real}) = 0.0

Base.show(io::IO, obj::NullPairPotential) = print(io, "NullPairPotential()")


# -----------------------------------------------------------------------------
# Harmonic Potential
# -----------------------------------------------------------------------------

Base.@kwdef struct HarmonicPotential <: ExternalPotential
    k::Float64 = 1.0
end

(V::HarmonicPotential)(r_sq::Float64) = V.k * r_sq
(V::HarmonicPotential)(r::AbstractVector{<:Real}) = V.k * (norm(r)^2)

Base.show(io::IO, obj::HarmonicPotential) = print(io, "HarmonicPotential(k=$(obj.k))")

function potential_derivative(V::HarmonicPotential, r::AbstractVector{<:Real})
    # V(r) = k * |r|^2 = k * (x^2 + y^2 + ...)
    # ∇V = 2k * r
    return 2 * V.k * r
end

function potential_derivative!(out::AbstractVector, V::HarmonicPotential, r::AbstractVector{<:Real})
    for d in eachindex(r, out)
        @inbounds out[d] = 2 * V.k * r[d]
    end
    return out
end


# -----------------------------------------------------------------------------
# Double Well Potential
# -----------------------------------------------------------------------------

Base.@kwdef struct DoubleWellPotential <: ExternalPotential
    A::Float64 = 10.0
    B::Float64 = 1.0
end

(V::DoubleWellPotential)(r::AbstractVector{<:Real}) = begin
    r′ = norm(r)
    V.A * (r′^4 - r′^2) - V.B * r′^2
end

Base.show(io::IO, obj::DoubleWellPotential) = print(io, "DoubleWellPotential(A=$(obj.A), B=$(obj.B))")

function potential_derivative(V::DoubleWellPotential, r::AbstractVector{<:Real})
    # V(r) = A(r^4 - r^2) - B r^2
    #      = A r^4 - (A + B) r^2
    # Let u = |r|^2 = r^2. V = A u^2 - (A+B) u
    # dV/dr_i = dV/du * du/dr_i
    # du/dr_i = 2 r_i
    # dV/du = 2A u - (A+B) = 2A r^2 - (A+B)
    # ∇V_i = (2A r^2 - (A+B)) * 2 r_i

    r2 = dot(r, r)
    prefactor = 2 * (2 * V.A * r2 - (V.A + V.B))
    return prefactor * r
end


# -----------------------------------------------------------------------------
# Lennard-Jones Potential
# -----------------------------------------------------------------------------

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

function potential_derivative(U::LennardJonesPotential, r::AbstractVector{<:Real})
    # V = 4ε[(σ/r)¹² - (σ/r)⁶], so ∇V = -24ε/r² [(σ/r)¹² - ½(σ/r)⁶] * r̂ * r = ...
    # dV/dr = 4ε[-12σ¹²/r¹³ + 6σ⁶/r⁷] = -24ε/r [2(σ/r)¹² - (σ/r)⁶]
    # ∇V = (dV/dr)(r/|r|) = -24ε/r² [2(σ/r)¹² - (σ/r)⁶] * r
    r_norm = norm(r)
    r_norm == 0.0 && return zero(r)
    sr6 = (U.σ / r_norm)^6
    sr12 = sr6 * sr6
    return (-24U.ϵ / r_norm^2) * (2sr12 - sr6) * r
end


# -----------------------------------------------------------------------------
# Coulomb Potential
# -----------------------------------------------------------------------------

Base.@kwdef struct CoulombPotential <: PairPotential
    g::Float64 = 1.0 # Note: positive g is REPULSIVE here if V = +g/r
end

(U::CoulombPotential)(r_norm::Float64) = r_norm == 0.0 ? 0.0 : U.g / r_norm
(U::CoulombPotential)(r::AbstractVector{<:Real}) = begin
    r′ = norm(r)
    r′ == 0.0 ? 0.0 : U.g / r′
end

Base.show(io::IO, obj::CoulombPotential) = print(io, "CoulombPotential(g=$(obj.g))")

function potential_derivative(U::CoulombPotential, r::AbstractVector{<:Real})
    # V(r) = g/|r|, so ∇V = -g*r/|r|³
    r_norm = norm(r)
    r_norm == 0.0 && return zero(r)
    return (-U.g / r_norm^3) * r
end


# -----------------------------------------------------------------------------
# Yakub–Ronchi (spherically averaged periodic Coulomb, OCP / UEG)
# Ref: Yakub & Ronchi, J. Chem. Phys. 119, 11556 (2003); Dornheim et al. (2025) Eq. (Yakub_potential)
# From reference to the ISHTAR source; Dornheim also has some 3x3x3 image expansoon
# -----------------------------------------------------------------------------

"""Volume-equivalent sphere radius for a cubic box of side `L`: ``r_{\\mathrm{cut}} = L (3/4\\pi)^{1/3}``."""
@inline function yakub_ronchi_r_cut(L::Float64)::Float64
    L * cbrt(3.0 / (4.0 * π))
end

"""
    yakub_ronchi_phi(r, L; g=1.0)

Isotropic Yakub–Ronchi pair potential ``\\phi_{\\mathrm{YR}}(r)`` for ``0 < r < r_{\\mathrm{cut}}``,
zero otherwise. Coupling `g` scales the strength (``g=1`` is the usual ``1/r``-normalised form).
"""
function yakub_ronchi_phi(r::Float64, L::Float64; g::Float64=1.0)::Float64
    r <= 0.0 && return Inf
    r_cut = yakub_ronchi_r_cut(L)
    r >= r_cut && return 0.0
    t = r / r_cut
    # phi = (g/r) * (1 + 0.5*t*(t^2 - 3))
    #     = g/r + g/r * 0.5 * (r^3/r_cut^3 - 3r/r_cut)
    #     = g/r + 0.5*g * (r^2/r_cut^3 - 3/r_cut)
    return (g / r) + 0.5 * g * (r^2 / r_cut^3 - 3.0 / r_cut)
end

"""
    yakub_ronchi_derivative(r, L; g=1.0)

Derivative dphi/dr = -g/r^2 + g*r/r_cut^3.
"""
function yakub_ronchi_derivative(r::Float64, L::Float64; g::Float64=1.0)::Float64
    r_cut = yakub_ronchi_r_cut(L)
    r >= r_cut && return 0.0
    return -g / (r * r) + g * r / (r_cut^3)
end


"""
    yakub_ronchi_background_constant(L, N)

Constant term in the YR Hamiltonian, ``- (3/(4 r_{\\mathrm{cut}}))(1 + N/5)`` (extensive scalar added once to `H`).

Needed for absolute energies; but currently not integrated into the estimators.
"""
@inline function yakub_ronchi_background_constant(L::Float64, N::Integer)::Float64
    r_cut = yakub_ronchi_r_cut(L)
    -(3.0 / (4.0 * r_cut)) * (1.0 + N / 5.0)
end

"""
    YakubRonchiPotential(L; g=1.0)

Spherically averaged periodic Coulomb pair interaction for a cubic box of side `L` (Yakub–Ronchi).
- Scalar call `U(r)` / `U(r_norm)` uses the single-distance form `yakub_ronchi_phi` (one term; same limitation as bare MIC Coulomb if the integrator only passes `|Δr|`).
- Vector call `U(Δr)` uses `yakub_ronchi_periodic_sum` with raw `Δr = r_a - r_b` (recommended for correctness).
"""
Base.@kwdef struct YakubRonchiPotential <: PairPotential
    L::Float64
    g::Float64 = 1.0
end

function (U::YakubRonchiPotential)(r_norm::Float64)
    yakub_ronchi_phi(r_norm, U.L; g=U.g)
end

function (U::YakubRonchiPotential)(r::AbstractVector{<:Real})
    yakub_ronchi_phi(norm(r), U.L; g=U.g)
end

Base.show(io::IO, obj::YakubRonchiPotential) =
    print(io, "YakubRonchiPotential(L=$(obj.L), g=$(obj.g))")

function potential_derivative(U::YakubRonchiPotential, r::AbstractVector{T}) where {T<:Real}
    r_norm = norm(r)
    if r_norm == 0.0
        return zero(r)
    end
    dv_dr = yakub_ronchi_derivative(r_norm, U.L; g=U.g)
    # Return a new vector with the same type and size as r, but scaled
    return (dv_dr / r_norm) .* r
end

function potential_derivative!(out::AbstractVector, U::YakubRonchiPotential, r::AbstractVector{<:Real})
    r_norm = norm(r)
    if r_norm == 0.0
        fill!(out, 0.0)
        return out
    end
    dv_dr = yakub_ronchi_derivative(r_norm, U.L; g=U.g)
    factor = dv_dr / r_norm
    for d in eachindex(r, out)
        @inbounds out[d] = factor * r[d]
    end
    return out
end

# -----------------------------------------------------------------------------
# Yukawa Potential
# -----------------------------------------------------------------------------

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

function potential_derivative(U::YukawaPotential, r::AbstractVector{<:Real})
    # V = -g*exp(-m*r)/r, so dV/dr = g*exp(-mr)*(1/r² + m/r) = g*exp(-mr)(1+mr)/r²
    # ∇V = (dV/dr)(r/|r|) = g*exp(-mr)(1+mr)/r³ * r
    r_norm = norm(r)
    r_norm == 0.0 && return zero(r)
    emr = exp(-U.m * r_norm)
    return (U.g * emr * (1 + U.m * r_norm) / r_norm^3) * r
end


# -----------------------------------------------------------------------------
# Hard Sphere Potential
# -----------------------------------------------------------------------------

Base.@kwdef struct HardSpherePotential <: PairPotential
    a::Float64 = 1.0
end

(U::HardSpherePotential)(r_norm::Float64) = r_norm < U.a ? Inf : 0.0
(U::HardSpherePotential)(r::AbstractVector{<:Real}) = norm(r) < U.a ? Inf : 0.0

Base.show(io::IO, obj::HardSpherePotential) = print(io, "HardSpherePotential(a=$(obj.a))")

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


# -----------------------------------------------------------------------------
# Aziz Potential (HFD-B)
# -----------------------------------------------------------------------------

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

function potential_derivative(u::AzizPotential, r::Real)
    # dV/dr for Aziz Potential
    x = r / u.rm
    x2 = x * x

    # Repulsive part: A * exp(-alpha * x)
    # d/dr (exp(-alpha * r/rm)) = -alpha/rm * exp(-alpha * r/rm)
    dV_rep = u.A * (-u.α / u.rm) * exp(-u.α * x)

    # Attractive part: (C6/x^6 + C8/x^8 + C10/x^10) * F(x)
    # For r > rm (x > 1), F(x) = 1.
    # For r < rm (x < 1), F(x) = exp(-(D/x - 1)^2)

    C_term = u.c6 / (x^6) + u.c8 / (x^8) + u.c10 / (x^10)
    dC_term = (-6 * u.c6 / x^7 - 8 * u.c8 / x^9 - 10 * u.c10 / x^11) / u.rm

    if x >= 1.0
        dV_att = u.ϵ * dC_term
    else
        f_x = exp(-(u.D / x - 1.0)^2)
        # dF/dx = F(x) * (2*D/x^2) * (D/x - 1)
        df_dx = f_x * (2 * u.D / (x^2)) * (u.D / x - 1.0)
        df_dr = df_dx / u.rm

        dV_att = u.ϵ * (dC_term * f_x + C_term * df_dr)
    end

    return dV_rep + dV_att
end

# AzizPotential was written as a scalar, now we wrap this vector interface
function potential_derivative(u::AzizPotential, r::AbstractVector{<:Real})
    r_val = norm(r)
    if r_val == 0
        return zeros(length(r))
    end
    # dV/dr_i = (dV/dr) * (dr/dr_i) = V'(r) * (r_i / r)
    dv_dr = potential_derivative(u, r_val)
    return (dv_dr / r_val) * r
end
