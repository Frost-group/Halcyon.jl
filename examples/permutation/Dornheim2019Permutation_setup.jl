# Shared setup for Dornheim et al. (2019) permutation-cycle work
# Ref: refs/Dornheim2019Permutation/main.tex — 3D spin-polarised UEG / ideal Fermi gas, θ = k_B T/E_F

using Halcyon

"""Wigner–Seitz density n = 3/(4π r_s³) in 3D (atomic units)."""
ueg_density_3d(r_s::Float64) = 3.0 / (4π * r_s^3)

"""Cubic box side L for N electrons at given r_s (continuum UEG density)."""
function ueg_box_length(N::Int, r_s::Float64)
    n = ueg_density_3d(r_s)
    return (N / n)^(1 / 3)
end

"""Fermi wavevector for fully spin-polarised 3D gas: N = V k_F³/(6π²)."""
fermi_wavenumber_polarised(n::Float64) = (6π^2 * n)^(1 / 3)

"""Fermi energy E_F = λ k_F² with λ = ℏ²/(2m) (Halcyon convention)."""
fermi_energy(n::Float64, λ::Float64) = λ * fermi_wavenumber_polarised(n)^2

"""
    ueg_theta_parameters(; N, θ, r_s, λ=0.5)

Degeneracy temperature θ = k_B T/E_F (k_B=1), inverse temperature β = 1/(θ E_F).
Returns `(L, β, E_F, n, k_F)` in atomic units (ℏ=m_e=1 as in the paper).
"""
function ueg_theta_parameters(; N::Int, θ::Float64, r_s::Float64, λ::Float64=0.5)
    L = ueg_box_length(N, r_s)
    n = N / L^3
    kF = fermi_wavenumber_polarised(n)
    EF = λ * kF^2
    β = 1.0 / (θ * EF)
    return (; L, β, E_F=EF, n, k_F=kF)
end

"""
    make_periodic_fermion_system(; M, N, β, L, λ=0.5, pair=NullPairPotential())

Free boundary: `V = HarmonicPotential(k=0)` (no trap). Fermi statistics for sign.
"""
function make_periodic_fermion_system(; M::Int, N::Int, β::Float64, L::Float64,
                                    λ::Float64=0.5, pair::PairPotential=NullPairPotential())
    System(M, N; D=3, β=β, λ=λ, L=L, V=HarmonicPotential(k=0.0), U=pair, statistics=Fermions)
end

default_worm_params(sys::System; C::Float64=1.0) =
    WormParams(C=C, j_max=sys.M ÷ 2, r_max=sys.L / 2)
