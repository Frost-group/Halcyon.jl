using Halcyon
using Test
using Statistics
using Printf

# =============================================================================
# ω–k–m–λ mapping for the quantum harmonic oscillator
# =============================================================================
#
# The PIMC path integral uses kinetic parameter  λ = ℏ²/(2m).
# In atomic units (ℏ = 1):  m = 1/(2λ).
#
# HarmonicPotential(k) gives  V(r) = ½ k |r|².
# The simulated Hamiltonian is:
#
#     H = −λ ∇² + ½ k |r|²
#
# This is a QHO with effective mass  m_eff = 1/(2λ)  and frequency
#
#     ω = √(k / m_eff) = √(2 k λ).
#
# Inverting:  k = ω² / (2λ)  =  m_eff ω².
#
# The exact energy of a D-dimensional isotropic QHO at inverse temperature β:
#
#     E(β, ω, D) = D · (ω/2) · coth(β ω / 2).
#
# =============================================================================

"""
    analytical_ho_energy(β, ω; D=3)

Exact thermal energy of a D-dimensional isotropic quantum harmonic oscillator:
``E = D \\cdot (\\omega/2) \\coth(\\beta\\omega/2)``.
"""
analytical_ho_energy(β, ω; D=3) = D * (ω / 2.0) * coth(β * ω / 2.0)

"""
    ho_spring_constant(ω, λ)

Return the HarmonicPotential spring constant `k` that yields oscillator
frequency `ω` for a particle with kinetic parameter `λ = ℏ²/(2m)`.

    k = ω² / (2λ)   [equivalently  k = m_eff · ω²  with  m_eff = 1/(2λ)]
"""
ho_spring_constant(ω, λ) = ω^2 / (2λ)

function default_worm_params(sys::System)
    return WormParams(C=1.0, j_max=sys.M ÷ 2, r_max=sys.L / 4.0)
end

"""
    run_estimator_audit(; kwargs...) -> NamedTuple

Run PIMC worm simulation and compare thermodynamic/virial energy estimators
against an exact value.  Returns `(; mean_th, mean_vir, E_exact, std_th, std_vir)`.
"""
function run_estimator_audit(; N=1, D=3, M=100, β=1.0, steps=100_000,
                             ext::Halcyon.ExternalPotential=HarmonicPotential(k=0.0),
                             pair::Halcyon.PairPotential=NullPairPotential(),
                             E_exact::Float64, L=10.0, λ=0.5,
                             equilibration=100_000, label="custom")

    sys = System(M, N; D=D, β=β, λ=λ, L=L, V=ext, U=pair)
    params = default_worm_params(sys)
    cfg = WormConfiguration(sys)

    for _ in 1:equilibration
        worm_step!(cfg, sys, params)
    end

    E_therm_samples = Float64[]
    E_vir_samples = Float64[]

    @printf("\n--- Audit: %s  N=%d D=%d M=%d β=%.2f L=%.1f ---\n", label, N, D, M, β, L)

    for t in 1:steps
        worm_step!(cfg, sys, params)
        if t % 5 == 0 && cfg.sector == Z_SECTOR
            Eth, Evir = energy_estimators(cfg, sys)
            push!(E_therm_samples, Eth)
            push!(E_vir_samples, Evir)
        end
    end

    mean_th = mean(E_therm_samples)
    std_th  = std(E_therm_samples) / sqrt(length(E_therm_samples))
    mean_vir = mean(E_vir_samples)
    std_vir  = std(E_vir_samples) / sqrt(length(E_vir_samples))

    @printf("Analytical E:  %.6f\n", E_exact)
    @printf("Thermodynamic: %.6f ± %.6f  (err %+.2f%%)\n",
            mean_th, std_th, 100 * (mean_th - E_exact) / E_exact)
    @printf("Virial:        %.6f ± %.6f  (err %+.2f%%)\n",
            mean_vir, std_vir, 100 * (mean_vir - E_exact) / E_exact)
    @printf("Samples:       %d (from %d MC steps)\n", length(E_vir_samples), steps)

    return (; mean_th, mean_vir, E_exact, std_th, std_vir)
end

# =============================================================================
# Test cases
# =============================================================================

@testset "Estimator Audit" begin
    @testset "Free particle (N=1, β=0.1)" begin
        D = 3; β = 0.1
        res = run_estimator_audit(D=D, β=β, L=10.0, steps=200_000, label="free",
                                  ext=HarmonicPotential(k=0.0),
                                  E_exact=Float64(D / (2β)))
        @test isapprox(res.mean_vir, res.E_exact, atol=0.1)
    end

    @testset "3D QHO (N=1, β=1, ω=1)" begin
        ω = 1.0; λ = 0.5; β = 1.0; D = 3
        k = ho_spring_constant(ω, λ)
        res = run_estimator_audit(D=D, β=β, L=10.0, steps=200_000, label="QHO-N1",
                                  ext=HarmonicPotential(k=k),
                                  E_exact=analytical_ho_energy(β, ω; D=D))
        @test isapprox(res.mean_vir, res.E_exact, atol=0.15)
    end

    @testset "Non-interacting N=2 in 3D QHO" begin
        # High T (β=0.5) to suppress bosonic exchange (~1.5% correction),
        # so E ≈ 2 × E_single tests multi-particle bookkeeping.
        ω = 1.0; λ = 0.5; β = 0.5; D = 3; N = 2
        k = ho_spring_constant(ω, λ)
        E_exact = N * analytical_ho_energy(β, ω; D=D)
        res = run_estimator_audit(N=N, D=D, β=β, L=10.0, M=50, steps=200_000,
                                  label="QHO-N2",
                                  ext=HarmonicPotential(k=k),
                                  E_exact=E_exact)
        @test isapprox(res.mean_th, res.E_exact, atol=1.0)  # thermo is very noisy at high T
        @test isapprox(res.mean_vir, res.E_exact, atol=0.5)
    end

    @testset "Ideal Bose gas (N=4, periodic box)" begin
        N = 4; D = 3; λ = 0.5; L = 5.0
        β = β_from_λT_ratio(0.5, L, λ)
        E_exact = E_N_exact(N, β, L, λ)
        res = run_estimator_audit(N=N, D=D, β=β, L=L, M=50, steps=300_000,
                                  equilibration=150_000, label="IdealBose-N4",
                                  ext=HarmonicPotential(k=0.0),
                                  E_exact=E_exact)
        @test isapprox(res.mean_vir, res.E_exact, atol=0.5)
    end

    @testset "Ideal Fermi gas (N=4, periodic box)" begin
        # Sign-reweighted bosonic PIMC: ⟨E⟩_F = ⟨E·S⟩_B / ⟨S⟩_B
        # Moderate degeneracy (λ_T/L = 0.5) keeps the average sign manageable.
        N = 4; D = 3; λ = 0.5; L = 5.0
        β = β_from_λT_ratio(0.5, L, λ)
        E_exact_F = E_N_exact_Fermi(N, β, L, λ)

        sys = System(50, N; D=D, β=β, λ=λ, L=L,
                     V=HarmonicPotential(k=0.0), U=NullPairPotential(),
                     statistics=Fermions)
        params = default_worm_params(sys)
        cfg = WormConfiguration(sys)

        for _ in 1:150_000
            worm_step!(cfg, sys, params)
        end

        ES_samples = Float64[]
        S_samples  = Float64[]
        for t in 1:300_000
            worm_step!(cfg, sys, params)
            if t % 5 == 0 && cfg.sector == Z_SECTOR
                _, Evir = energy_estimators(cfg, sys)
                S = Float64(permutation_sign(cfg, sys))
                push!(ES_samples, Evir * S)
                push!(S_samples, S)
            end
        end

        mean_ES = mean(ES_samples)
        mean_S  = mean(S_samples)
        E_vir_F = mean_ES / mean_S
        n_samp  = length(S_samples)

        @printf("\n--- Audit: IdealFermi-N4  N=%d D=%d M=50 β=%.2f L=%.1f ---\n", N, D, β, L)
        @printf("Exact E_F: %.6f\n", E_exact_F)
        @printf("⟨S⟩ = %.4f  (n=%d)\n", mean_S, n_samp)
        @printf("Virial (reweighted): %.6f  (err %+.2f%%)\n",
                E_vir_F, 100 * (E_vir_F - E_exact_F) / E_exact_F)

        @test mean_S > 0.05   # sign problem not catastrophic
        @test isapprox(E_vir_F, E_exact_F, atol=0.5)
    end

    @testset "Hookium (N=2, k=1/4, g=1, E₀=2)" begin
        # Taut 1993: 2 electrons in HO + Coulomb, k=1/4 ⟹ E₀ = 2.0 Eₕ exactly.
        # Spatial ground state is symmetric (singlet) so bosonic PIMC suffices.
        N = 2; D = 3; λ = 0.5; L = 20.0
        β = 20.0; M = 200
        k_ho = 0.25
        res = run_estimator_audit(N=N, D=D, β=β, L=L, M=M, steps=500_000,
                                  equilibration=200_000, label="Hookium",
                                  ext=HarmonicPotential(k=k_ho),
                                  pair=CoulombPotential(g=1.0),
                                  E_exact=2.0)
        @test isapprox(res.mean_th, 2.0, atol=0.1)
        @test isapprox(res.mean_vir, 2.0, atol=0.15)
    end
end
