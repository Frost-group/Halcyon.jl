# Dornheim2023-Fictitious.jl
# Reproduce Figures 1-3 from Dornheim et al. (2023)
# "Fermionic physics from ab initio PIMC simulations of fictitious identical particles"
# System: N=6 spin-polarized electrons in 2D harmonic trap with Coulomb repulsion

using Halcyon
using Statistics
using Printf
using ProgressMeter

# ═══════════════════════════════════════════════════════════════════════════════
# Physical Parameters (Oscillator Units: ℏ=m=Ω=1)
# ═══════════════════════════════════════════════════════════════════════════════

const N_PARTICLES = 6
const D = 2
const β_DEFAULT = 1.0
const M_DEFAULT = 20
const λ = 0.5               # ℏ²/(2m) = 0.5 in oscillator units
const k_trap = 0.5          # V(r) = 1/2 k r²
const L = 5.0              # Box size (large for trapped system)

# ═══════════════════════════════════════════════════════════════════════════════
# Monte Carlo Parameters
# ═══════════════════════════════════════════════════════════════════════════════

const EQUILIBRATION_STEPS = 1_000_000
const MEASUREMENT_STEPS = 5_000_000
const MEASURE_INTERVAL = 100

const C = 0.7
const j_max = 50 # For redrawing: uses Levy flight (diffusion), so in harmonic trap keep these small?
const r_max = 8.0 # max move...

# ═══════════════════════════════════════════════════════════════════════════════
# Reweighting Analysis with Jackknife Errors
# ═══════════════════════════════════════════════════════════════════════════════

"""
    jackknife_reweight(E_samples, N_pp_samples, ξ; n_blocks=20)

Compute reweighted energy and jackknife error at given ξ.
Weight: w_i = ξ^{N_pp_i}
⟨E⟩_ξ = Σ(E_i * w_i) / Σ(w_i)
"""
function jackknife_reweight(E_samples::Vector{Float64}, N_pp_samples::Vector{Int}, ξ::Float64; n_blocks::Int=20)
    n = length(E_samples)
    block_size = n ÷ n_blocks

    # Full estimate
    weights = [ξ^N_pp for N_pp in N_pp_samples]
    sum_Ew = sum(E_samples .* weights)
    sum_w = sum(weights)
    E_full = sum_Ew / sum_w

    # Jackknife: leave-one-block-out
    E_jack = Float64[]
    for b in 1:n_blocks
        i_start = (b - 1) * block_size + 1
        i_end = b * block_size

        # Compute without block b
        sum_Ew_b = sum_Ew - sum(E_samples[i_start:i_end] .* weights[i_start:i_end])
        sum_w_b = sum_w - sum(weights[i_start:i_end])

        if abs(sum_w_b) > 1e-15
            push!(E_jack, sum_Ew_b / sum_w_b)
        end
    end

    if length(E_jack) < 2
        return E_full, NaN
    end

    E_jack_mean = mean(E_jack)
    var_jack = (n_blocks - 1) / n_blocks * sum((E_jack .- E_jack_mean) .^ 2)
    err = sqrt(var_jack)

    return E_full, err
end

"""
    compute_sign(N_pp_samples, ξ; n_blocks=20)

Compute average sign S(ξ) = ⟨ξ^{N_pp}⟩_{ξ=1} / ⟨|ξ|^{N_pp}⟩_{ξ=1} with jackknife error.
For ξ > 0, S = 1 by definition.
"""
function compute_sign(N_pp_samples::Vector{Int}, ξ::Float64; n_blocks::Int=20)
    if ξ >= 0
        return 1.0, 0.0
    end

    n = length(N_pp_samples)
    block_size = n ÷ n_blocks

    # S = ⟨ξ^{N_pp}⟩ / ⟨|ξ|^{N_pp}⟩
    weights_signed = [ξ^N_pp for N_pp in N_pp_samples]
    weights_abs = [abs(ξ)^N_pp for N_pp in N_pp_samples]

    sum_signed = sum(weights_signed)
    sum_abs = sum(weights_abs)
    S_full = sum_signed / sum_abs

    S_jack = Float64[]
    for b in 1:n_blocks
        i_start = (b - 1) * block_size + 1
        i_end = b * block_size

        sum_s_b = sum_signed - sum(weights_signed[i_start:i_end])
        sum_a_b = sum_abs - sum(weights_abs[i_start:i_end])

        if abs(sum_a_b) > 1e-15
            push!(S_jack, sum_s_b / sum_a_b)
        end
    end

    if length(S_jack) < 2
        return S_full, NaN
    end

    S_jack_mean = mean(S_jack)
    var_jack = (n_blocks - 1) / n_blocks * sum((S_jack .- S_jack_mean) .^ 2)
    err = sqrt(var_jack)

    return S_full, err
end

# ═══════════════════════════════════════════════════════════════════════════════
# Simulation Engine
# ═══════════════════════════════════════════════════════════════════════════════

function run_simulation(; N::Int=N_PARTICLES, β::Float64=β_DEFAULT, M::Int=M_DEFAULT,
    λ_coulomb::Float64=0.5, verbose::Bool=true)
    sys = System(
        M, N;
        D=D, β=β, λ=λ,
        V=HarmonicPotential(k=k_trap),
        U=CoulombPotential(g=-λ_coulomb),
        L=L,
        statistics=Bosons  # Sample in Bosonic ensemble for reweighting
    )

    params = WormParams(C=C, j_max=M ÷ 2, r_max=r_max)
    cfg = WormConfiguration(sys)

    verbose && println("-"^70)
    verbose && @printf("Simulating: N=%d, β=%.2f, M=%d, λ_c=%.2f\n", N, β, M, λ_coulomb)
    verbose && println("-"^70)

    # Equilibration
    verbose && println("Equilibrating... $(EQUILIBRATION_STEPS) steps")
    @showprogress dt = 1 desc = "Equilibration" for step in 1:EQUILIBRATION_STEPS
        worm_step!(cfg, sys, params)
    end

    # Measurement arrays (compact: Float64 + Int per sample)
    E_virial_samples = Float64[]
    E_thermo_samples = Float64[]
    N_pp_samples = Int[]
    cycle_counts = zeros(Int, N)  # Histogram of cycle lengths

    verbose && println("Measuring... $(MEASUREMENT_STEPS) steps")
    @showprogress dt = 1 desc = "Measurement" for step in 1:MEASUREMENT_STEPS
        worm_step!(cfg, sys, params)

        if step % MEASURE_INTERVAL == 0 && cfg.sector == Z_SECTOR
            push!(E_virial_samples, energy_virial(cfg, sys))
            push!(E_thermo_samples, energy_thermodynamic(cfg, sys))

            n_cycles = count_cycles(cfg)
            N_pp = N - n_cycles
            push!(N_pp_samples, N_pp)

            # Accumulate cycle histogram
            visited = falses(N)
            for i in 1:N
                if !visited[i]
                    cycle = get_cycle(cfg, i)
                    m = length(cycle)
                    cycle_counts[m] += 1
                    for p in cycle
                        visited[p] = true
                    end
                end
            end
        end
    end

    n_samples = length(E_virial_samples)
    mem_bytes = sizeof(E_virial_samples) + sizeof(E_thermo_samples) + sizeof(N_pp_samples)
    verbose && println("-"^70)
    verbose && @printf("Collected %d Z-sector samples (%.2f MB in RAM)\n", n_samples, mem_bytes / 1e6)
    verbose && println("Entering observable calculation phase...")
    verbose && println("-"^70)

    return (
        E_virial_samples=E_virial_samples,
        E_thermo_samples=E_thermo_samples,
        N_pp_samples=N_pp_samples,
        cycle_counts=cycle_counts,
        N=N, β=β, λ_coulomb=λ_coulomb
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# Analysis & Output
# ═══════════════════════════════════════════════════════════════════════════════

function analyze_and_save(result; output_prefix::String="Dornheim2023", xis::AbstractVector{Float64}=-1.0:0.05:1.0)
    E_vir = result.E_virial_samples
    E_thm = result.E_thermo_samples
    N_pp = result.N_pp_samples

    println("="^70)
    println("Reweighting Analysis")
    println("="^70)

    # Output TSV
    fname = "$(output_prefix)_lambda$(result.λ_coulomb).tsv"
    open(fname, "w") do io
        write(io, "# xi\tE_virial\tE_virial_err\tE_thermo\tE_thermo_err\tSign\tSign_err\n")
        for ξ in xis
            E_v, E_v_err = jackknife_reweight(E_vir, N_pp, ξ)
            E_t, E_t_err = jackknife_reweight(E_thm, N_pp, ξ)
            S, S_err = compute_sign(N_pp, ξ)

            @printf(io, "%.4f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
                ξ, E_v, E_v_err, E_t, E_t_err, S, S_err)
        end
    end
    println("Saved: $fname")

    # Cycle probabilities (at ξ=1, Bosonic)
    total_cycles = sum(result.cycle_counts)
    fname_cycles = "$(output_prefix)_lambda$(result.λ_coulomb)_cycles.tsv"
    open(fname_cycles, "w") do io
        write(io, "# m\tP_m\n")
        for m in 1:result.N
            P_m = (m * result.cycle_counts[m] / total_cycles) / result.N
            @printf(io, "%d\t%.6f\n", m, P_m)
        end
    end
    println("Saved: $fname_cycles")

    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

function main(; λ_coulomb::Float64=0.5)
    result = run_simulation(λ_coulomb=λ_coulomb)
    analyze_and_save(result)
end

# CLI support
if abspath(PROGRAM_FILE) == @__FILE__
    λ_c = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 0.5
    main(λ_coulomb=λ_c)
end
