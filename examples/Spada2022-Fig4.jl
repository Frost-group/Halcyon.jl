# Spada2022-Fig4.jl
# Reproduce Figure 4 from Spada et al. 2022
# "Path-integral Monte Carlo worm algorithm for Bose systems with PBC"
#
# Figure 4: N-particle thermodynamic limit
# E/N vs N at different T/T_c^0 ratios
# Shows convergence to thermodynamic limit as 1/N → 0
#
# Ref: Spada et al. 2022, Section III

using Halcyon
using Statistics
using Gnuplot
using Printf

# ═══════════════════════════════════════════════════════════════════════════════
# Physical Parameters
# ═══════════════════════════════════════════════════════════════════════════════

const D = 3          # 3D
const L = 1.0        # Box size (will be adjusted to keep density fixed)
const λ = 0.5        # ℏ²/(2m)
const M = 32         # Number of beads (fixed, sufficiently large)

# T/T_c^0 ratios to test
const T_ratio_values = [0.5, 1.0, 1.5]
const colors = ["#377eb8", "#4daf4a", "#e41a1c"]  # Blue, Green, Red

# N values to test (up to 64 as in paper)
const N_values = [1, 2, 4, 8, 16, 32, 64]

# Fixed density n = N/V = 1.0 (adjust L for each N)
const n_target = 1.0

# MC parameters (reduced for large N)
const EQUILIBRATION_STEPS = 200_000
const MEASUREMENT_STEPS = 2_000_000
const MEASURE_INTERVAL = 500

# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

"""
Run PIMC simulation for N particles and return mean E/N with error.
"""
function run_simulation(N::Int, β::Float64, L_box::Float64; verbose::Bool=false)
    sys = Halcyon.System(
        M, N,
        m=1.0, D=D, β=β,
        V=Halcyon.HarmonicPotential(k=0.0),
        U=Halcyon.NullPairPotential(),
        λ=λ, L=L_box
    )

    cfg = Halcyon.WormConfiguration(sys)
    params = Halcyon.WormParams(C=1.0, j_max=M, j_max_open=M, j_max_swap=M, r_max=min(0.5, L_box / 2))

    # Adjust steps for large N
    eq_steps = max(EQUILIBRATION_STEPS ÷ N, 50_000)
    meas_steps = max(MEASUREMENT_STEPS ÷ N, 500_000)

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
            push!(energies, E / N)  # Store E/N
        end

        if cfg.sector == Halcyon.Z_SECTOR
            n_Z += 1
        end
    end

    if verbose
        Z_frac = n_Z / meas_steps
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
    println("Spada et al. 2022 - Figure 4 Reproduction")
    println("E/N vs N (thermodynamic limit)")
    println("="^70)

    # Compute T_c^0 for the target density
    T_c0 = Halcyon.critical_temperature(n_target, λ)
    println("Target density n = $n_target")
    println("Critical temperature T_c^0 = $(round(T_c0, digits=6))")

    # Results: T_ratio => [(N, E/N, err), ...]
    results = Dict{Float64,Vector{Tuple{Int,Float64,Float64}}}()
    exact_E_per_N = Dict{Float64,Dict{Int,Float64}}()

    for T_ratio in T_ratio_values
        println("\n" * "-"^50)
        println("T/T_c^0 = $T_ratio")
        println("-"^50)

        T = T_ratio * T_c0
        β = 1.0 / T
        println("T = $(round(T, digits=6)), β = $(round(β, digits=4))")

        results[T_ratio] = Tuple{Int,Float64,Float64}[]
        exact_E_per_N[T_ratio] = Dict{Int,Float64}()

        for N in N_values
            # Adjust box size to keep density fixed
            L_box = (N / n_target)^(1 / D)

            println("\n  N = $N, L = $(round(L_box, digits=4)):")

            # Exact energy (per particle)
            E_exact = Halcyon.E_N_exact(N, β, L_box, λ)
            E_per_N_exact = E_exact / N
            exact_E_per_N[T_ratio][N] = E_per_N_exact

            # Simulation
            E_per_N_mean, E_per_N_err = run_simulation(N, β, L_box; verbose=true)
            push!(results[T_ratio], (N, E_per_N_mean, E_per_N_err))

            if !isnan(E_per_N_mean)
                diff = E_per_N_mean - E_per_N_exact
                @printf("  E/N = %.6f ± %.6f (exact: %.6f, diff: %.6f)\n",
                    E_per_N_mean, E_per_N_err, E_per_N_exact, diff)
            end
        end
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # Plotting
    # ═══════════════════════════════════════════════════════════════════════════

    println("\n" * "="^70)
    println("Generating plots...")
    println("="^70)

    E_scale = T_c0  # Energy in units of k_B T_c^0

    # Figure 4a: E/N vs N
    @gp "reset"
    @gp :- "set title 'Internal Energy per Particle' font ',14'"
    @gp :- "set xlabel 'N'"
    @gp :- "set ylabel 'E / (N k_B T_c^0)'"
    @gp :- "set logscale x 2"
    @gp :- "set key top right"
    @gp :- "set grid"

    for (idx, T_ratio) in enumerate(T_ratio_values)
        data = results[T_ratio]
        exact = exact_E_per_N[T_ratio]

        N_arr = [d[1] for d in data]
        E_arr = [d[2] / E_scale for d in data]
        err_arr = [d[3] / E_scale for d in data]
        E_exact_arr = [exact[N] / E_scale for N in N_arr]

        label = @sprintf("T/T_c^0 = %.1f", T_ratio)
        color = colors[idx]

        @gp :- Float64.(N_arr) E_arr err_arr "with yerrorbars pt 7 ps 1.2 lc rgb '$color' title '$label'"
        @gp :- Float64.(N_arr) E_exact_arr "with lines lt 2 lw 1.5 lc rgb '$color' notitle"
    end

    Gnuplot.save("Spada2022-Fig4a.png", term="pngcairo size 800,600 enhanced font 'Helvetica,14'")
    Gnuplot.save("Spada2022-Fig4a.pdf", term="pdfcairo size 4in,3in enhanced font 'Helvetica,10'")

    # Figure 4b: E/N vs 1/N (for extrapolation)
    @gp "reset"
    @gp :- "set title 'Thermodynamic Limit Extrapolation' font ',14'"
    @gp :- "set xlabel '1/N'"
    @gp :- "set ylabel 'E / (N k_B T_c^0)'"
    @gp :- "set key top right"
    @gp :- "set grid"
    @gp :- "set xrange [-0.05:1.1]"

    for (idx, T_ratio) in enumerate(T_ratio_values)
        data = results[T_ratio]
        exact = exact_E_per_N[T_ratio]

        inv_N_arr = [1.0 / d[1] for d in data]
        E_arr = [d[2] / E_scale for d in data]
        err_arr = [d[3] / E_scale for d in data]
        E_exact_arr = [exact[Int(1 / x)] / E_scale for x in inv_N_arr]

        label = @sprintf("T/T_c^0 = %.1f", T_ratio)
        color = colors[idx]

        @gp :- inv_N_arr E_arr err_arr "with yerrorbars pt 7 ps 1.2 lc rgb '$color' title '$label'"
        @gp :- inv_N_arr E_exact_arr "with lines lt 2 lw 1.5 lc rgb '$color' notitle"
    end

    Gnuplot.save("Spada2022-Fig4b.png", term="pngcairo size 800,600 enhanced font 'Helvetica,14'")
    Gnuplot.save("Spada2022-Fig4b.pdf", term="pdfcairo size 4in,3in enhanced font 'Helvetica,10'")

    println("\nPlots saved: Spada2022-Fig4a.{png,pdf}, Spada2022-Fig4b.{png,pdf}")
end

main()


