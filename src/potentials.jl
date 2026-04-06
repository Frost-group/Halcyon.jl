using LinearAlgebra # for norm
using ForwardDiff

# -----------------------------------------------------------------------------
# Abstract Types & Generic Interfaces
# -----------------------------------------------------------------------------

abstract type AbstractPotential end
abstract type ExternalPotential <: AbstractPotential end

"""
    virial_contribution(U, r)

Return the radial virial contribution ``r \\cdot \\nabla V(r) + L \\frac{\\partial V}{\\partial L}``.
This is essential for the virial energy estimator.
"""
function virial_contribution end

abstract type PairPotential <: AbstractPotential end

# Fallback: if called with a scalar, assume it's |r| and convert to vector call if needed,
(U::PairPotential)(r_norm::Float64) = U([r_norm, 0.0, 0.0])

"""
    potential_derivative(V::AbstractPotential, r) -> Vector

Return the gradient ∇V(r) of the potential.
Fallback uses ForwardDiff for automatic differentiation.

Almost certainly unacceptably slow. 

Also no careful treatment of limits, so Virial estimator likely to explode with the gradients. 
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

"""
    is_null(V::AbstractPotential) -> Bool

Return `true` if the potential contributes nothing to the action or energy.
"""
is_null(::AbstractPotential) = false

"""
    centroid_virial_term(V, r, centroid_or_Δcentroid) -> Float64

Centroid virial kinetic contribution `(r − R̄) · ∇V(r)`.

For external potentials, `centroid_or_Δcentroid` is the particle centroid R̄ᵢ.
For pair potentials, `r` is the MIC pair displacement Δrᵢₖ and
`centroid_or_Δcentroid` is the MIC centroid difference R̄ᵢ − R̄ₖ.

Specialisations avoid allocations for hot-path potentials.
"""
function centroid_virial_term(V::ExternalPotential, r::AbstractVector, centroid::AbstractVector)
    g = potential_derivative(V, r)
    return dot(r .- centroid, g)
end

function centroid_virial_term(U::PairPotential, rij::AbstractVector, Δcentroid::AbstractVector)
    g = potential_derivative(U, rij)
    return dot(rij .- Δcentroid, g)
end

# -----------------------------------------------------------------------------
# Null Potential
# -----------------------------------------------------------------------------

struct NullPairPotential <: PairPotential end

(::NullPairPotential)(r_norm::Float64) = 0.0
(::NullPairPotential)(r::AbstractVector{<:Real}) = 0.0
potential_derivative(::NullPairPotential, r::AbstractVector{<:Real}) = zero(r)
virial_contribution(U::NullPairPotential, r) = 0.0

is_null(::NullPairPotential) = true
centroid_virial_term(::NullPairPotential, ::AbstractVector, ::AbstractVector) = 0.0

Base.show(io::IO, obj::NullPairPotential) = print(io, "NullPairPotential()")


# -----------------------------------------------------------------------------
# Harmonic Potential
# -----------------------------------------------------------------------------

Base.@kwdef struct HarmonicPotential <: ExternalPotential
    k::Float64 = 1.0
end

(V::HarmonicPotential)(r_sq::Float64) = 0.5 * V.k * r_sq
(V::HarmonicPotential)(r::AbstractVector{<:Real}) = 0.5 * V.k * sum(abs2, r)

Base.show(io::IO, obj::HarmonicPotential) = print(io, "HarmonicPotential(k=$(obj.k))")

function potential_derivative(V::HarmonicPotential, r::AbstractVector{<:Real})
    # V(r) = 0.5 * k * |r|^2 = 0.5 * k * (x^2 + y^2 + ...)
    # ∇V = k * r
    return V.k * r
end

function potential_derivative!(out::AbstractVector, V::HarmonicPotential, r::AbstractVector{<:Real})
    for d in eachindex(r, out)
        @inbounds out[d] = V.k * r[d]
    end
    return out
end

is_null(V::HarmonicPotential) = (V.k == 0.0)

function virial_contribution(V::HarmonicPotential, r::AbstractVector)
    return 2.0 * V(r)
end

function centroid_virial_term(V::HarmonicPotential, r::AbstractVector, centroid::AbstractVector)
    s = 0.0
    @inbounds for d in eachindex(r, centroid)
        s += (r[d] - centroid[d]) * V.k * r[d]
    end
    return s
end


# -----------------------------------------------------------------------------
# Harmonic Pair Potential
# -----------------------------------------------------------------------------

Base.@kwdef struct HarmonicPairPotential <: PairPotential
    k::Float64 = 1.0
end

(U::HarmonicPairPotential)(r_norm::Float64) = 0.5 * U.k * (r_norm^2)
(U::HarmonicPairPotential)(r::AbstractVector{<:Real}) = 0.5 * U.k * sum(abs2, r)

Base.show(io::IO, obj::HarmonicPairPotential) = print(io, "HarmonicPairPotential(k=$(obj.k))")

function potential_derivative(U::HarmonicPairPotential, r::AbstractVector{<:Real})
    return U.k * r
end

function potential_derivative!(out::AbstractVector, U::HarmonicPairPotential, r::AbstractVector{<:Real})
    for d in eachindex(r, out)
        @inbounds out[d] = U.k * r[d]
    end
    return out
end

is_null(U::HarmonicPairPotential) = (U.k == 0.0)

function virial_contribution(U::HarmonicPairPotential, r::AbstractVector)
    return 2.0 * U(r)
end

function centroid_virial_term(U::HarmonicPairPotential, rij::AbstractVector, Δcentroid::AbstractVector)
    s = 0.0
    @inbounds for d in eachindex(rij, Δcentroid)
        s += (rij[d] - Δcentroid[d]) * U.k * rij[d]
    end
    return s
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
# 1/r nastiness - Kelbg corrections
# -----------------------------------------------------------------------------
"""
Approximates the core polynomial P(t) for the complementary error function.
Used to factor out the exp(-x^2) term for computational efficiency.
"""
@inline function hastings_poly(x::Float64)::Float64
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    t = 1.0 / (1.0 + p * x)

    # Horner's method for polynomials
    return t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
end

const π_sqrt = sqrt(π)

# -----------------------------------------------------------------------------
# Kelbg-Smoothed Coulomb Potential (Quantum Dot)
# -----------------------------------------------------------------------------

Base.@kwdef struct KelbgCoulombPotential <: PairPotential
    g::Float64       # Coupling strength (+ is repulsive)
    λ::Float64       # Segment thermal wavelength : ħ²τ / (2μ)

    # Pre-calculated for performance
    inv_λ::Float64 = 1.0 / λ
    inv_λ2::Float64 = 1.0 / (λ^2)
end

Base.show(io::IO, obj::KelbgCoulombPotential) = print(io, "KelbgCoulombPotential(g=$(obj.g), λ=$(obj.λ))")

# --- Action Evaluator ---
(U::KelbgCoulombPotential)(r_norm::Float64) = begin
    r_norm < 1e-10 && return U.g * π_sqrt * U.inv_λ

    x = r_norm * U.inv_λ

    # Avoid transcendental math for long range where correction is negligible (< 1e-15)
    x > 6.0 && return U.g / r_norm

    exp_mx2 = exp(-x^2)
    poly_erfc = hastings_poly(x)

    return (U.g / r_norm) * (1.0 - exp_mx2 * (1.0 - x * π_sqrt * poly_erfc))
end

(U::KelbgCoulombPotential)(r::AbstractVector{<:Real}) = U(norm(r))

# --- Virial Forces and Estimators ---
function potential_derivative(U::KelbgCoulombPotential, r::AbstractVector)
    r_norm = norm(r)
    r_norm < 1e-10 && return zeros(length(r))

    # ∇U = (g / r³) * expm1(-r² / λ²) * r
    x2 = r_norm^2 * U.inv_λ2
    prefactor = (U.g / r_norm^3) * (x2 > 36.0 ? -1.0 : expm1(-x2))
    return prefactor * r
end

function potential_derivative!(out::AbstractVector, U::KelbgCoulombPotential, r::AbstractVector{<:Real})
    r_norm = norm(r)
    if r_norm < 1e-10
        fill!(out, 0.0)
    else
        x2 = r_norm^2 * U.inv_λ2
        prefactor = (U.g / r_norm^3) * (x2 > 36.0 ? -1.0 : expm1(-x2))
        out .= prefactor .* r
    end
    return out
end

function virial_contribution(U::KelbgCoulombPotential, r::AbstractVector)
    r_norm = norm(r)
    r_norm < 1e-10 && return 0.0

    # r · ∇U = (g / r) * expm1(-r² / λ²)
    x2 = r_norm^2 * U.inv_λ2
    return (U.g / r_norm) * (x2 > 36.0 ? -1.0 : expm1(-x2))
end

function centroid_virial_term(U::KelbgCoulombPotential, rij::AbstractVector, Δcentroid::AbstractVector)
    r_norm = norm(rij)
    r_norm < 1e-10 && return 0.0

    s = 0.0
    @inbounds for d in eachindex(rij, Δcentroid)
        s += (rij[d] - Δcentroid[d]) * rij[d]
    end

    x2 = r_norm^2 * U.inv_λ2
    prefactor = (U.g / r_norm^3) * (x2 > 36.0 ? -1.0 : expm1(-x2))
    return prefactor * s
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

function potential_derivative(U::CoulombPotential, r::AbstractVector)
    r_norm = norm(r)
    return (-U.g / r_norm^3) * r
end

function virial_contribution(U::CoulombPotential, r::AbstractVector)
    return -1.0 * U(r)
end

function centroid_virial_term(U::CoulombPotential, rij::AbstractVector, Δcentroid::AbstractVector)
    r_norm = norm(rij)
    r_norm == 0.0 && return 0.0
    # ∇U = −g/r³ · r ⟹ (Δr − ΔR̄)·∇U = −(g/r³) Σ_d (rij_d − Δc_d) rij_d
    s = 0.0
    @inbounds for d in eachindex(rij, Δcentroid)
        s += (rij[d] - Δcentroid[d]) * rij[d]
    end
    return (-U.g / r_norm^3) * s
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
    # t = r / r_cut
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
    out .= (dv_dr / r_norm) .* r
    return out
end

function virial_contribution(U::YakubRonchiPotential, r::AbstractVector)
    r_norm = norm(r)
    (r_norm == 0.0 || r_norm >= yakub_ronchi_r_cut(U.L)) && return 0.0
    return -1.0 * U(r_norm)
end

function centroid_virial_term(U::YakubRonchiPotential, rij::AbstractVector, Δcentroid::AbstractVector)
    r_norm = norm(rij)
    (r_norm == 0.0 || r_norm >= yakub_ronchi_r_cut(U.L)) && return 0.0
    dv_dr = yakub_ronchi_derivative(r_norm, U.L; g=U.g)
    # ∇U = (dV/dr)(r̂) = (dV/dr / r) · r ⟹ (Δr − ΔR̄)·∇U = (dV/dr / r) Σ (rij_d − Δc_d) rij_d
    s = 0.0
    @inbounds for d in eachindex(rij, Δcentroid)
        s += (rij[d] - Δcentroid[d]) * rij[d]
    end
    return (dv_dr / r_norm) * s
end

# -----------------------------------------------------------------------------
# Yakub-Ronchi Potential with Kelbg Short-Range Smoothing (UEG)
# -----------------------------------------------------------------------------
"""
A PairPotential for Uniform Electron Gas (UEG) simulations.
Applies the Yakub-Ronchi isotropic spatial cutoff for long-range interactions,
and the exact diagonal Kelbg two-body action for short-range quantum smoothing.
"""
Base.@kwdef struct YakubRonchiKelbgPotential <: PairPotential
    g::Float64             # Coupling strength
    λ::Float64       # Segment thermal de Broglie wavelength: ħ²τ / (2μ)
    L::Float64             # Simulation box length

    # Pre-calculated to save CPU cycles in the inner loop
    r_cut::Float64 = yakub_ronchi_r_cut(L)
    inv_λ::Float64 = 1.0 / λ
    inv_λ2::Float64 = 1.0 / (λ^2)
end

Base.show(io::IO, obj::YakubRonchiKelbgPotential) = print(io, "YakubRonchiKelbgPotential(g=$(obj.g), λ=$(obj.λ), L=$(obj.L))")

(U::YakubRonchiKelbgPotential)(r::Float64) = begin
    # Beyond the cutoff, particles do not interact
    r >= U.r_cut && return 0.0

    # The macroscopic background polynomial (Yakub-Ronchi)
    yr_bg = 0.5 * U.g * ((r^2 / U.r_cut^3) - (3.0 / U.r_cut))

    # Safely bounded finite potential at the origin
    if r < 1e-10
        return U.g * π_sqrt * U.inv_λ + yr_bg
    end

    x = r * U.inv_λ

    # TODO: Check this is actually OK and doesn't include artefacts!
    if x > 6.0 # if long range, return bare YakubRonchi to avoid polynomial + exponential
        return (U.g / r) + yr_bg
    end

    # Compute the exponential once for both the Kelbg term and Hastings poly
    exp_mx2 = exp(-x^2)
    poly_erfc = hastings_poly(x)

    # Factored Kelbg short-range action + YR long-range background
    kelbg_term = (U.g / r) * (1.0 - exp_mx2 * (1.0 - x * π_sqrt * poly_erfc))

    return kelbg_term + yr_bg
end

(U::YakubRonchiKelbgPotential)(r::AbstractVector{<:Real}) = U(norm(r))

function potential_derivative(U::YakubRonchiKelbgPotential, r::AbstractVector)
    r_norm = norm(r)

    r_norm >= U.r_cut && return zeros(length(r))
    r_norm < 1e-10 && return zeros(length(r))

    x2 = r_norm^2 * U.inv_λ2
    # ∇U = [ (g/r³) * expm1(-r²/λ²) + (g/r_cut³) ] * r

    # The first term is the exact Kelbg force; the second is the continuous YR background force.
    scalar_force_over_r = (U.g / r_norm^3) * (x2 > 36.0 ? -1.0 : expm1(-x2)) + (U.g / U.r_cut^3)

    return scalar_force_over_r * r
end

"""
Follows above, but optimised to avoid allocations.
"""
function potential_derivative!(out::AbstractVector, U::YakubRonchiKelbgPotential, r::AbstractVector{<:Real})
    r_norm = norm(r)

    if r_norm >= U.r_cut || r_norm < 1e-10
        fill!(out, 0.0)
    else
        # ∇U = [ (g/r³) * expm1(-r²/λ²) + (g/r_cut³) ] * r
        x2 = r_norm^2 * U.inv_λ2
        scalar_force_over_r = (U.g / r_norm^3) * (x2 > 36.0 ? -1.0 : expm1(-x2)) + (U.g / U.r_cut^3)
        out .= scalar_force_over_r .* r
    end
    return out
end

function virial_contribution(U::YakubRonchiKelbgPotential, r::AbstractVector)
    r_norm = norm(r)

    r_norm >= U.r_cut && return 0.0
    r_norm < 1e-10 && return 0.0

    # r · ∇U
    x2 = r_norm^2 * U.inv_λ2
    kelbg_virial = (U.g / r_norm) * (x2 > 36.0 ? -1.0 : expm1(-x2))
    yr_virial = (U.g * r_norm^2) / (U.r_cut^3)

    return kelbg_virial + yr_virial
end

function centroid_virial_term(U::YakubRonchiKelbgPotential, rij::AbstractVector, Δcentroid::AbstractVector)
    r_norm = norm(rij)

    r_norm >= U.r_cut && return 0.0
    r_norm < 1e-10 && return 0.0

    s = 0.0
    @inbounds for d in eachindex(rij, Δcentroid)
        s += (rij[d] - Δcentroid[d]) * rij[d]
    end
    # Multiply the positional dot-product `s` by the scalar magnitude of the force over distance

    x2 = r_norm^2 * U.inv_λ2
    scalar_force_over_r = (U.g / r_norm^3) * (x2 > 36.0 ? -1.0 : expm1(-x2)) + (U.g / U.r_cut^3)

    return scalar_force_over_r * s
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
