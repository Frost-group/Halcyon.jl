# Dornheim2019.jl
# Reproduce Table I/III from Dornheim et al., J. Chem. Phys. 151, 014108 (2019)
# "Fermion sign problem in path integral Monte Carlo simulations..."
# System: 2D quantum dot with harmonic confinement + repulsive Coulomb

# NB: Frozen Boson currently (MC simulation assumes Boson, Fermion reweighting at energy estimator levle)

using Halcyon
using Statistics
using Printf

# ═══════════════════════════════════════════════════════════════════════════════
# Physical Parameters (Oscillator Units: ℏ=m=Ω=1)
# ═══════════════════════════════════════════════════════════════════════════════

const N_DEFAULT = 6                 # Number of particles
const D = 2                 # Dimensions
const β_DEFAULT = 1.0               # Inverse temperature
const M_DEFAULT = 100               # Number of beads
const λ = 0.5               # ℏ²/(2m) = 0.5 in oscillator units

# Potential parameters
const k_trap = 0.5               # V(r) = 1/2 k r²
const λ_coulomb = 0.5               # U(r) = λ_c / r (Coulomb strength)

# Physical Box (currently crudy flat initialisation of forms, needs this to be sma;ll;)
const L = 5.0

# ═══════════════════════════════════════════════════════════════════════════════
# Monte Carlo Parameters
# ═══════════════════════════════════════════════════════════════════════════════

const EQUILIBRATION_STEPS = 100_000
const MEASUREMENT_STEPS = 5_000_000
const MEASURE_INTERVAL = 100

# Worm parameters
const C = 1.0                  # Worm ratio
const j_max = 50                   # Max segment length (fixed default)
const r_max = 8.0                  # Max displacement (for confined fermions)

# ═══════════════════════════════════════════════════════════════════════════════
# Simulation Engine
# ═══════════════════════════════════════════════════════════════════════════════

function run_simulation(M, N, β; verbose=true)
    sys = System(
        M, N;
        D=D, β=β, λ=λ,
        V=HarmonicPotential(k=k_trap),
        U=CoulombPotential(g=-λ_coulomb),
        L=L,
        statistics=Fermions
    )

    params = WormParams(C=C, j_max=M ÷ 2, r_max=r_max)
    cfg = WormConfiguration(sys)

    verbose && println("-"^70)
    verbose && @printf("Simulating: N=%d, β=%.2f, M=%d (τ=%.4f)\n", N, β, M, sys.τ)
    verbose && println("-"^70)

    # Equilibration
    for step in 1:EQUILIBRATION_STEPS
        worm_step!(cfg, sys, params)
    end

    # Measurement
    signs = Int[]
    E_thermo_samples = Float64[]
    E_virial_samples = Float64[]

    for step in 1:MEASUREMENT_STEPS
        worm_step!(cfg, sys, params)
        if step % MEASURE_INTERVAL == 0 && cfg.sector == Z_SECTOR
            push!(signs, permutation_sign(cfg, sys))
            Et, Ev = energy_estimators(cfg, sys)
            push!(E_thermo_samples, Et)
            push!(E_virial_samples, Ev)
        end
    end

    if isempty(signs)
        return (avg_S=0.0, err_S=0.0, E_F_thermo=0.0, err_EF_thermo=0.0, E_B_thermo=0.0, err_EB_thermo=0.0, E_F_virial=0.0, err_EF_virial=0.0, E_B_virial=0.0, err_EB_virial=0.0)
    end

    avg_S = mean(signs)
    err_S = std(signs) / sqrt(length(signs))

    # Helper for ratio estimator error
    function ratio_stats(vals, signs, avg_S)
        sum_S = sum(signs)
        E_F = sum(vals .* signs) / sum_S
        var_EF = var((vals .* signs .- E_F .* signs) ./ avg_S)
        err_EF = sqrt(var_EF / length(signs))
        E_B = mean(vals)
        err_EB = std(vals) / sqrt(length(vals))
        return E_F, err_EF, E_B, err_EB
    end

    E_F_thermo, err_EF_thermo, E_B_thermo, err_EB_thermo = ratio_stats(E_thermo_samples, signs, avg_S)
    E_F_virial, err_EF_virial, E_B_virial, err_EB_virial = ratio_stats(E_virial_samples, signs, avg_S)

    return (
        avg_S=avg_S, err_S=err_S,
        E_F_thermo=E_F_thermo, err_EF_thermo=err_EF_thermo, E_B_thermo=E_B_thermo, err_EB_thermo=err_EB_thermo,
        E_F_virial=E_F_virial, err_EF_virial=err_EF_virial, E_B_virial=E_B_virial, err_EB_virial=err_EB_virial
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Table I: System-size dependence (β=1, λ_c=0.5)
# ═══════════════════════════════════════════════════════════════════════════════

function table_I()
    println("\n" * "="^110)
    println("REPRODUCING TABLE I: N-dependence (β=1, λ_c=0.5)")
    println("="^110)
    @printf("%-5s %-18s %-40s %-40s\n", "N", "Sign ⟨S⟩", "E_F: Virial (Thermo)", "E_B: Virial (Thermo)")
    println("-"^110)

    for n in [3, 4, 6, 10]
        res = run_simulation(M_DEFAULT, n, 1.0; verbose=false)
        @printf("%-5d %6.4f ± %.4f    %7.3f ± %.3f (%7.3f ± %.3f)    %7.3f ± %.3f (%7.3f ± %.3f)\n",
            n, res.avg_S, res.err_S,
            res.E_F_virial, res.err_EF_virial, res.E_F_thermo, res.err_EF_thermo,
            res.E_B_virial, res.err_EB_virial, res.E_B_thermo, res.err_EB_thermo)
    end
    println("="^110)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Table III: Temperature dependence (N=6, λ_c=0.5)
# ═══════════════════════════════════════════════════════════════════════════════

function table_III()
    println("\n" * "="^110)
    println("REPRODUCING TABLE III: β-dependence (N=6, λ_c=0.5)")
    println("="^110)
    @printf("%-5s %-18s %-40s %-40s\n", "β", "Sign ⟨S⟩", "E_F: Virial (Thermo)", "E_B: Virial (Thermo)")
    println("-"^110)

    for b in [0.3, 0.5, 1.0, 2.0]
        res = run_simulation(M_DEFAULT, N_DEFAULT, b; verbose=false)
        @printf("%-5.1f %6.4f ± %.4f    %7.3f ± %.3f (%7.3f ± %.3f)    %7.3f ± %.3f (%7.3f ± %.3f)\n",
            b, res.avg_S, res.err_S,
            res.E_F_virial, res.err_EF_virial, res.E_F_thermo, res.err_EF_thermo,
            res.E_B_virial, res.err_EB_virial, res.E_B_thermo, res.err_EB_thermo)
    end
    println("="^110)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

if abspath(PROGRAM_FILE) == @__FILE__
    println("\nSingle point (M=$M_DEFAULT, N=$N_DEFAULT, β=$β_DEFAULT):")
    res = run_simulation(M_DEFAULT, N_DEFAULT, β_DEFAULT)
    @printf("Sign ⟨S⟩:      %.5f ± %.5f\n", res.avg_S, res.err_S)
    @printf("Energy E_F:    %.4f ± %.4f (Thermo: %.4f ± %.4f)\n", res.E_F_virial, res.err_EF_virial, res.E_F_thermo, res.err_EF_thermo)
    @printf("Energy E_B:    %.4f ± %.4f (Thermo: %.4f ± %.4f)\n", res.E_B_virial, res.err_EB_virial, res.E_B_thermo, res.err_EB_thermo)

    # Uncomment these to run the full table sweeps
    table_I()
    table_III()
end
