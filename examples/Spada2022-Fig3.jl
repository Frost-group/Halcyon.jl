# Spada2022-Fig3.jl
# Reproduce Figure 3 from Spada et al. 2022
# "Path-integral Monte Carlo worm algorithm for Bose systems with PBC"
#
# Figure 3: N=2 system at different λ_T/L values
# Internal energy E vs number of beads M
# Tests swap move correctness with two particles
#
# Ref: Spada et al. 2022, Section III

using Halcyon
using Statistics
using Gnuplot
using Printf

# ═══════════════════════════════════════════════════════════════════════════════
# Physical Parameters
# ═══════════════════════════════════════════════════════════════════════════════

const N = 2          # Two particles
const D = 3          # 3D
const L = 1.0        # Box size
const λ = 0.5        # ℏ²/(2m)

# λ_T/L ratios (same as Fig 1)
const λT_L_values = [0.795, 0.974, 1.257, 1.646]
const colors = ["#e41a1c", "#4daf4a", "#ff7f00", "#377eb8"]

# Number of beads to test
const M_values = [1, 2, 4, 8, 16, 32]

# MC parameters
const EQUILIBRATION_STEPS = 500_000
const MEASUREMENT_STEPS = 10_000_000
const MEASURE_INTERVAL = 1000

# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

"""
Run PIMC simulation for N=2 and return mean energy with error.
"""
function run_simulation(M::Int, β::Float64; verbose::Bool=false)
    sys = Halcyon.System(
        M, N,
        m=1.0, D=D, β=β,
        V=Halcyon.HarmonicPotential(k=0.0),
        U=Halcyon.NullPairPotential(),
        λ=λ, L=L
    )

    cfg = Halcyon.WormConfiguration(sys)
    params = Halcyon.WormParams(C=1.0, j_max=M, j_max_open=M, j_max_swap=M, r_max=0.5)

    # Equilibration
    if verbose
        println("  Equilibrating...")
    end
    for _ in 1:EQUILIBRATION_STEPS
        Halcyon.worm_step!(cfg, sys, params)
    end

    # Measurement
    energies = Float64[]
    n_swaps = 0
    swap_accepts = 0
    n_Z = 0

    for step in 1:MEASUREMENT_STEPS
        move = Halcyon.worm_step!(cfg, sys, params)

        if move == :swap
            n_swaps += 1
        end

        if step % MEASURE_INTERVAL == 0 && cfg.sector == Halcyon.Z_SECTOR
            _, E = Halcyon.energy_estimators(cfg, sys) # seems better SNR particularly at low T
            push!(energies, E)
        end

        if cfg.sector == Halcyon.Z_SECTOR
            n_Z += 1
        end
    end

    if verbose
        Z_frac = n_Z / MEASUREMENT_STEPS
        @printf("  Z-sector: %.1f%%, Measurements: %d\n", 100 * Z_frac, length(energies))
    end

    if length(energies) < 10
        @warn "Too few measurements" n = length(energies)
        return NaN, NaN
    end

    return mean(energies), std(energies) / sqrt(length(energies))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    println("="^70)
    println("Spada et al. 2022 - Figure 3 Reproduction")
    println("N=2 internal energy vs number of beads M")
    println("="^70)

    results = Dict{Float64,Vector{Tuple{Float64,Float64}}}()
    exact_energies = Dict{Float64,Float64}()

    for λT_L in λT_L_values
        println("\n" * "-"^50)
        println("λ_T/L = $λT_L")
        println("-"^50)

        β = Halcyon.β_from_λT_ratio(λT_L, L, λ)
        println("β = $(round(β, digits=4))")

        # Exact N=2 energy
        E_exact = Halcyon.E_N_exact(N, β, L, λ)
        exact_energies[λT_L] = E_exact
        println("Exact E_2 = $(round(E_exact, digits=6))")

        results[λT_L] = Tuple{Float64,Float64}[]

        for M in M_values
            println("\n  M = $M beads:")
            E_mean, E_std = run_simulation(M, β; verbose=true)
            push!(results[λT_L], (E_mean, E_std))

            if !isnan(E_mean)
                diff = E_mean - E_exact
                @printf("  E = %.6f ± %.6f (diff = %.6f)\n", E_mean, E_std, diff)
            end
        end
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # Plotting
    # ═══════════════════════════════════════════════════════════════════════════

    println("\n" * "="^70)
    println("Generating plots...")
    println("="^70)

    n = N / L^D
    T_c0 = Halcyon.critical_temperature(n, λ)
    E_scale = N * T_c0

    M_plot = Float64.(M_values)

    # Figure 3a: Energy vs M
    @gp "reset"
    @gp :- "set title 'Internal Energy E (N=2)' font ',14'"
    @gp :- "set xlabel 'Number of beads M'"
    @gp :- "set ylabel 'E / (N k_B T_c^0)'"
    @gp :- "set logscale y"
    @gp :- "set xtics (1,2,4,8,16,32)"
    @gp :- "set xrange [0:35]"
    @gp :- "set key top right"
    @gp :- "set grid"

    for (idx, λT_L) in enumerate(λT_L_values)
        E_exact = exact_energies[λT_L]
        data = results[λT_L]

        E_means = [d[1] / E_scale for d in data]
        E_errs = [d[2] / E_scale for d in data]
        E_exact_scaled = E_exact / E_scale

        label = @sprintf("λ_T/L = %.3f", λT_L)
        color = colors[idx]

        @gp :- M_plot E_means E_errs "with yerrorbars pt 7 ps 1.2 lc rgb '$color' title '$label'"
        @gp :- [M_plot[1], M_plot[end]] [E_exact_scaled, E_exact_scaled] "with lines lt 2 lw 1.5 lc rgb '$color' notitle"
    end

    Gnuplot.save("Spada2022-Fig3a.png", term="pngcairo size 800,600 enhanced font 'Helvetica,14'")
    Gnuplot.save("Spada2022-Fig3a.pdf", term="pdfcairo size 4in,3in enhanced font 'Helvetica,10'")

    # Figure 3b: Difference from exact
    @gp "reset"
    @gp :- "set title 'Difference from Exact (N=2)' font ',14'"
    @gp :- "set xlabel 'Number of beads M'"
    @gp :- "set ylabel '(E - E_{exact}) / (N k_B T_c^0)'"
    @gp :- "set xtics (1,2,4,8,16,32)"
    @gp :- "set xrange [0:35]"
    @gp :- "set key top right"
    @gp :- "set grid"

    @gp :- [M_plot[1], M_plot[end]] [0.0, 0.0] "with lines lt 2 lw 1 lc rgb 'gray' notitle"

    for (idx, λT_L) in enumerate(λT_L_values)
        E_exact = exact_energies[λT_L]
        data = results[λT_L]

        diffs = [(d[1] - E_exact) / E_scale for d in data]
        errs = [d[2] / E_scale for d in data]

        label = @sprintf("λ_T/L = %.3f", λT_L)
        color = colors[idx]

        @gp :- M_plot diffs errs "with yerrorbars pt 7 ps 1.2 lc rgb '$color' title '$label'"
    end

    Gnuplot.save("Spada2022-Fig3b.png", term="pngcairo size 800,600 enhanced font 'Helvetica,14'")
    Gnuplot.save("Spada2022-Fig3b.pdf", term="pdfcairo size 4in,3in enhanced font 'Helvetica,10'")

    println("\nPlots saved: Spada2022-Fig3a.{png,pdf}, Spada2022-Fig3b.{png,pdf}")
end

main()


