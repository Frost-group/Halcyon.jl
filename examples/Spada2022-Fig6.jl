# Spada2022-Fig6.jl
# Reproduce Figure 6 from Spada et al. 2022
# "Path-integral Monte Carlo worm algorithm for Bose systems with PBC"
#
# Figure 6: Thermodynamic limit of Hard-sphere gas (na³=0.1)
# E/N vs 1/N for M=64 beads.
#
# Ref: Spada et al. 2022, Section IV

using Halcyon
using Statistics
using Gnuplot
using Printf

# ═══════════════════════════════════════════════════════════════════════════════
# Physical Parameters
# ═══════════════════════════════════════════════════════════════════════════════

const D = 3
const λ = 0.5
const gas_parameter = 0.1  # na³
const a_hs = 1.0           # Hard sphere diameter
const M = 64               # Beads fixed at 64

# Density n = 0.1/a³ = 0.1
const n_target = gas_parameter / a_hs^3

# T/T_c⁰ ratios to test
const T_ratio_values = [0.5, 1.0]
const colors = ["#377eb8", "#e41a1c"]  # Blue, Red

# N values to test
const N_values = [32, 64, 128]

# MC parameters (increased for extrapolation precision)
const EQUILIBRATION_STEPS = 500_000
const MEASUREMENT_STEPS = 10_000_000
const MEASURE_INTERVAL = 1000

# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

function run_simulation(N::Int, β::Float64, L_box::Float64; verbose::Bool=false)
    sys = Halcyon.System(
        M, N,
        m=1.0, D=D, β=β,
        V=Halcyon.HarmonicPotential(k=0.0),
        U=Halcyon.HardSpherePotential(a=a_hs),
        λ=λ, L=L_box
    )

    cfg = Halcyon.WormConfiguration(sys)
    params = Halcyon.WormParams(C=1.0, j_max=M, j_max_open=M, j_max_swap=M, r_max=min(0.5, L_box / 2))

    # Steps adjusted for N
    eq_steps = max(EQUILIBRATION_STEPS ÷ (N ÷ 32), 200_000)
    meas_steps = max(MEASUREMENT_STEPS ÷ (N ÷ 32), 2_000_000)

    if verbose
        println("  Equilibrating ($eq_steps steps)...")
    end
    for _ in 1:eq_steps
        Halcyon.worm_step!(cfg, sys, params)
    end

    energies = Float64[]
    n_Z = 0

    for step in 1:meas_steps
        Halcyon.worm_step!(cfg, sys, params)

        if step % MEASURE_INTERVAL == 0 && cfg.sector == Halcyon.Z_SECTOR
            _, E = Halcyon.energy_estimators(cfg, sys)
            push!(energies, E / N)
        end

        if cfg.sector == Halcyon.Z_SECTOR
            n_Z += 1
        end
    end

    if verbose
        Z_frac = n_Z / meas_steps
        @printf("  Z-sector: %.1f%%, Measurements: %d\n", 100 * Z_frac, length(energies))
    end

    return mean(energies), std(energies) / sqrt(length(energies))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    println("="^70)
    println("Spada et al. 2022 - Figure 6 Reproduction")
    println("Hard-sphere gas (na³=0.1) Thermodynamic Limit")
    println("="^70)

    T_c0 = Halcyon.critical_temperature(n_target, λ)
    println("Target density n = $n_target")
    println("T_c0 = $(round(T_c0, digits=6))")

    # Results: T_ratio => [(N, E/N, err), ...]
    results = Dict{Float64,Vector{Tuple{Int,Float64,Float64}}}()

    for T_ratio in T_ratio_values
        β = 1.0 / (T_ratio * T_c0)
        println("\n" * "-"^50)
        println("T/T_c0 = $T_ratio")
        println("-"^50)

        res_vec = Tuple{Int,Float64,Float64}[]
        for N in N_values
            L_box = (N / n_target)^(1 / D)
            println("\n  N = $N, L = $(round(L_box, digits=4)) (M=$M):")
            E_mean, E_err = run_simulation(N, β, L_box; verbose=true)
            push!(res_vec, (N, E_mean, E_err))
            @printf("  E/N = %.6f ± %.6f\n", E_mean, E_err)
        end
        results[T_ratio] = res_vec
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # Plotting
    # ═══════════════════════════════════════════════════════════════════════════

    println("\n" * "="^70)
    println("Generating plots...")
    println("="^70)

    E_scale = T_c0

    # Figure 6a and 6b in paper are two panels for the two temperatures.
    for (t_idx, T_ratio) in enumerate(T_ratio_values)
        data = results[T_ratio]
        inv_N = [1.0 / d[1] for d in data]
        E_arr = [d[2] / E_scale for d in data]
        err_arr = [d[3] / E_scale for d in data]

        color = colors[t_idx]
        title = @sprintf("Hard-Sphere Gas (na³=0.1) T/T_c^0 = %.1f", T_ratio)
        fname = @sprintf("Spada2022-Fig6_%d", Int(10 * T_ratio))

        @gp "reset"
        @gp :- "set title '$title' font ',14'"
        @gp :- "set xlabel '1/N'"
        @gp :- "set ylabel 'E / (N k_B T_c^0)'"
        @gp :- "set grid"
        @gp :- "set xrange [0:0.04]"

        @gp :- inv_N E_arr err_arr "with yerrorbars pt 7 ps 1.2 lc rgb '$color' notitle"

        # Linear fit? For now just points.

        Gnuplot.save("$(fname).png", term="pngcairo size 800,600 enhanced font 'Helvetica,14'")
        Gnuplot.save("$(fname).pdf", term="pdfcairo size 4in,3in enhanced font 'Helvetica,10'")
    end

    println("\nPlots saved: Spada2022-Fig6_*.{png,pdf}")
end

main()

