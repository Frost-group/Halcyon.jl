# exact.jl - Exact partition function and energy for non-interacting Bose gas
# Ref: Spada et al. 2022, Section III, Eqs. 22-26

"""
    z1(β, L, λ; nmax=50) -> Float64

Single-particle partition function with periodic boundary conditions.
Ref: Spada et al. 2022, Eq. 25

z(β) = Σ_{n_x,n_y,n_z=-∞}^{∞} exp(-β ε(n_x, n_y, n_z))

where the single-particle energies are:
ε(n_x, n_y, n_z) = λ (2π/L)² (n_x² + n_y² + n_z²)

Arguments:
- β: Inverse temperature
- L: Box side length
- λ: ℏ²/(2m)
- nmax: Truncation for sum over quantum numbers
"""
function z1(β::Float64, L::Float64, λ::Float64; nmax::Int=50)
    prefactor = λ * (2π / L)^2
    
    # 1D sum: Σ_{n=-nmax}^{nmax} exp(-β λ (2π/L)² n²)
    z1d = 0.0
    for n in -nmax:nmax
        z1d += exp(-β * prefactor * n^2)
    end
    
    # 3D: product of three 1D sums (separable)
    return z1d^3
end

"""
    z1_deriv(β, L, λ; nmax=50) -> Float64

Derivative of single-particle partition function with respect to β.
dz/dβ = -Σ ε(n) exp(-β ε(n))
"""
function z1_deriv(β::Float64, L::Float64, λ::Float64; nmax::Int=50)
    prefactor = λ * (2π / L)^2
    
    # Need: d/dβ [z1d³] = 3 z1d² dz1d/dβ
    z1d = 0.0
    dz1d = 0.0
    for n in -nmax:nmax
        ε_n = prefactor * n^2
        e_term = exp(-β * ε_n)
        z1d += e_term
        dz1d += -ε_n * e_term
    end
    
    return 3 * z1d^2 * dz1d
end

"""
    E1_exact(β, L, λ; nmax=50) -> Float64

Exact internal energy for N=1 non-interacting particle.
Ref: Spada et al. 2022, Eq. 22

E = -∂ln(Z₁)/∂β = -(1/Z₁) ∂Z₁/∂β
"""
function E1_exact(β::Float64, L::Float64, λ::Float64; nmax::Int=50)
    Z = z1(β, L, λ; nmax=nmax)
    dZ = z1_deriv(β, L, λ; nmax=nmax)
    return -dZ / Z
end

"""
    thermal_wavelength(β, λ) -> Float64

Thermal de Broglie wavelength.
Ref: Spada et al. 2022, below Eq. 21

λ_T = √(2π ℏ² β / m) = √(4π λ β)
"""
thermal_wavelength(β::Float64, λ::Float64) = sqrt(4π * λ * β)

"""
    critical_temperature(n, λ) -> Float64

BEC critical temperature for ideal gas in thermodynamic limit.
Ref: Spada et al. 2022, Eq. 34

k_B T_c^0 = 4π λ (n / ζ(3/2))^{2/3}

Returns k_B T_c^0 (i.e., 1/β_c).
"""
function critical_temperature(n::Float64, λ::Float64)
    ζ_3_2 = 2.612375348685488  # Riemann zeta(3/2)
    return 4π * λ * (n / ζ_3_2)^(2/3)
end

"""
    β_from_T_ratio(T_ratio, n, λ) -> Float64

Compute inverse temperature β given T/T_c^0 ratio.
"""
function β_from_T_ratio(T_ratio::Float64, n::Float64, λ::Float64)
    T_c = critical_temperature(n, λ)
    T = T_ratio * T_c
    return 1.0 / T
end

"""
    λT_over_L(β, L, λ) -> Float64

Compute the ratio λ_T / L (thermal wavelength to box size).
This is the key parameter controlling quantum effects with PBC.
"""
λT_over_L(β::Float64, L::Float64, λ::Float64) = thermal_wavelength(β, λ) / L

"""
    β_from_λT_ratio(λT_L, L, λ) -> Float64

Compute inverse temperature β given λ_T/L ratio.
λ_T = √(4π λ β) = ratio × L
=> β = (ratio × L)² / (4π λ)
"""
function β_from_λT_ratio(ratio::Float64, L::Float64, λ::Float64)
    λT = ratio * L
    return λT^2 / (4π * λ)
end

# ═══════════════════════════════════════════════════════════════════════════════
# G_1/Z_1 ratio for Figure 2
# ═══════════════════════════════════════════════════════════════════════════════

"""
    jacobi_theta3(z, q; nmax=100) -> Float64

Jacobi theta function θ₃(z, q).
Ref: Abramowitz & Stegun, Chapter 16

θ₃(z, q) = 1 + 2 Σ_{n=1}^∞ q^{n²} cos(2nz)

For z=0: θ₃(0, q) = 1 + 2 Σ_{n=1}^∞ q^{n²}
"""
function jacobi_theta3(z::Float64, q::Float64; nmax::Int=100)
    result = 1.0
    for n in 1:nmax
        term = 2 * q^(n^2) * cos(2n * z)
        result += term
        abs(term) < 1e-15 && break
    end
    return result
end

"""
    G1_Z1_ratio(λT_L) -> Float64

Ratio G₁/Z₁ for a single particle with periodic boundary conditions.
Ref: Spada et al. 2022, Eq. 35

G₁/Z₁ = [θ₃(0, exp(-π λ_T²/L²))]^{-3}

This is used to verify the open/close moves satisfy detailed balance.
The ratio N_G/N_Z in simulation should equal C × G₁/Z₁.
"""
function G1_Z1_ratio(λT_L::Float64)
    q = exp(-π * λT_L^2)
    θ3 = jacobi_theta3(0.0, q)
    return θ3^(-3)
end

# ═══════════════════════════════════════════════════════════════════════════════
# N-particle partition function for Figures 3-4
# ═══════════════════════════════════════════════════════════════════════════════

"""
    Z_N(N, β, L, λ; cache=nothing, nmax=50) -> Float64

N-particle partition function for non-interacting bosons via recursion.
Ref: Spada et al. 2022, Eq. 23

Z_N(β) = (1/N) Σ_{k=1}^N z(kβ) Z_{N-k}(β)

with Z_0 = 1 and z(β) the single-particle partition function.
"""
function Z_N(N::Int, β::Float64, L::Float64, λ::Float64; 
             cache::Union{Nothing,Dict}=nothing, nmax::Int=50)
    N == 0 && return 1.0
    N == 1 && return z1(β, L, λ; nmax=nmax)
    
    # Use cache if provided
    if cache !== nothing
        key = (N, β)
        haskey(cache, key) && return cache[key]
    end
    
    # Recursion: Z_N = (1/N) Σ_{k=1}^N z(kβ) Z_{N-k}
    result = 0.0
    for k in 1:N
        z_k = z1(k * β, L, λ; nmax=nmax)
        Z_Nmk = Z_N(N - k, β, L, λ; cache=cache, nmax=nmax)
        result += z_k * Z_Nmk
    end
    result /= N
    
    # Store in cache
    if cache !== nothing
        cache[(N, β)] = result
    end
    
    return result
end

"""
    E_N_exact(N, β, L, λ; nmax=50) -> Float64

Exact internal energy for N non-interacting bosons.
Ref: Spada et al. 2022, Eq. 22

E = -∂ln(Z_N)/∂β

Computed via numerical differentiation.
"""
function E_N_exact(N::Int, β::Float64, L::Float64, λ::Float64; nmax::Int=50)
    N == 1 && return E1_exact(β, L, λ; nmax=nmax)
    
    # Numerical derivative with small δβ
    δβ = β * 1e-6
    cache_plus = Dict{Tuple{Int,Float64},Float64}()
    cache_minus = Dict{Tuple{Int,Float64},Float64}()
    
    Z_plus = Z_N(N, β + δβ, L, λ; cache=cache_plus, nmax=nmax)
    Z_minus = Z_N(N, β - δβ, L, λ; cache=cache_minus, nmax=nmax)
    
    # E = -d(ln Z)/dβ = -(1/Z) dZ/dβ ≈ -(ln Z₊ - ln Z₋)/(2δβ)
    return -(log(Z_plus) - log(Z_minus)) / (2δβ)
end

"""
    E_thermodynamic_limit(T_ratio, λ) -> Float64

Internal energy per particle in the thermodynamic limit for ideal Bose gas.
Ref: Standard statistical mechanics

For T > T_c: E/N = (3/2) k_B T × g_{5/2}(z) / g_{3/2}(z)
For T < T_c: E/N = (3/2) k_B T × ζ(5/2) / ζ(3/2) × (T/T_c)^{3/2}

Returns E/(N k_B T_c^0).
"""
function E_thermodynamic_limit(T_ratio::Float64, λ::Float64)
    ζ_3_2 = 2.612375348685488  # ζ(3/2)
    ζ_5_2 = 1.341487257250917  # ζ(5/2)
    
    if T_ratio <= 1.0
        # Below T_c: condensate present
        return (3/2) * T_ratio * (ζ_5_2 / ζ_3_2) * T_ratio^(3/2)
    else
        # Above T_c: need to solve for fugacity (approximate)
        # For simplicity, use high-T classical limit correction
        # E/N ≈ (3/2) k_B T for T >> T_c
        return (3/2) * T_ratio
    end
end

