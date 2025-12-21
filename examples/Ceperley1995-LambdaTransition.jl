# examples/Ceperley1995-LambdaTransition.jl
# Replicate Ceperley 1995 Figure 13 (Energy) and Phase Transition signature
#
# Sweeps temperature from 1.0K to 3.5K.
# Measures:
# 1. Kinetic Energy (K) - Expect drop at Tc
# 2. Potential Energy (V) - Expect smooth variation
# 3. Superfluid Fraction (rho_s/rho) - Expect onset at Tc ~ 2.17K

using Halcyon, Printf, Gnuplot, Statistics

# ═══════════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════════

const N = 64
const D = 3
const rho = 0.02182 # A^-3 (Liquid He density)
const L = (N / rho)^(1 / 3)
const λ = 6.0596 # Å² K

# Temperature sweep
const temperatures = [0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.1, 2.17, 2.2, 2.3, 2.5, 3.0, 3.5, 4.0]

# MC parameters (increased for better Cv convergence)
const EQUILIBRATION_STEPS = 100_000
const MEASUREMENT_STEPS = 5_000_000
const MEASURE_INTERVAL = 5
# Timing: ~6 hours on M1 Mac (21 Dec 2024) with 5M steps × 14 temps, N=64

# ═══════════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════════

function run_point(T::Float64)
    β = 1.0 / T
    M = max(20, round(Int, β / 0.04))

    println("\n" * "-"^60)
    @printf(" T = %.2f K (M=%d)\n", T, M)
    println("-"^60)

    sys = System(M, N; m=4.0026, D=D, β=β, λ=λ, L=L, U=AzizPotential())
    cfg = WormConfiguration(sys)
    params = WormParams(C=0.5, j_max=20, j_max_open=20, j_max_swap=15, r_max=L / 2)

    # Equilibrating
    for step in 1:EQUILIBRATION_STEPS
        worm_step!(cfg, sys, params)
    end

    # Measuring
    K_samples = Float64[]
    V_samples = Float64[]
    E_samples = Float64[]
    rhos_samples = Float64[]

    for step in 1:MEASUREMENT_STEPS
        worm_step!(cfg, sys, params)

        if step % MEASURE_INTERVAL == 0
            if cfg.sector == Z_SECTOR
                # Energy components
                K, V, E = energy_components(cfg, sys)
                push!(K_samples, K)
                push!(V_samples, V)
                push!(E_samples, E)

                # Superfluidity
                rhos = superfluid_fraction(cfg, sys)
                push!(rhos_samples, rhos)
            end
        end
    end

    # Statistics
    results = Dict()
    if !isempty(E_samples)
        # Average Energies per atom
        results[:K] = mean(K_samples) / N
        results[:V] = mean(V_samples) / N
        results[:E] = mean(E_samples) / N
        results[:E_err] = std(E_samples) / sqrt(length(E_samples)) / N

        # Superfluid fraction
        results[:rhos] = mean(rhos_samples)
        results[:rhos_err] = std(rhos_samples) / sqrt(length(rhos_samples))

        # Specific Heat Cv = beta^2 * var(E) / N^2 (dimensionless, per atom)
        e_var = var(E_samples)
        results[:Cv] = (β^2 * e_var) / (N^2)
    else
        results[:K] = NaN
        results[:V] = NaN
        results[:E] = NaN
        results[:E_err] = NaN
        results[:rhos] = NaN
        results[:rhos_err] = NaN
        results[:Cv] = NaN
    end

    @printf("  E/N = %.2f, rho_s = %.3f, Cv = %.3f (Z-samples: %d)\n",
        results[:E], results[:rhos], results[:Cv], length(E_samples))

    return results
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════════════════════

data_K = Float64[]
data_V = Float64[]
data_E = Float64[]
data_E_err = Float64[]
data_rhos = Float64[]
data_rhos_err = Float64[]
data_Cv = Float64[]
data_T = Float64[]

for T in temperatures
    res = run_point(T)
    push!(data_T, T)
    push!(data_K, res[:K])
    push!(data_V, res[:V])
    push!(data_E, res[:E])
    push!(data_E_err, res[:E_err])
    push!(data_rhos, res[:rhos])
    push!(data_rhos_err, res[:rhos_err])
    push!(data_Cv, res[:Cv])
end

# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

println("\nGenerating plots...")

# Figure 1: Energy (Ceperley Fig 13)
@gp "reset"
@gp :- "set title 'Energy per Atom vs Temperature (Ceperley Fig 13)'"
@gp :- "set xlabel 'Temperature (K)'"
@gp :- "set ylabel 'Energy (K/atom)'"
@gp :- "set grid"
@gp :- "set xrange [0:4]"
# Ceperley Energy is negative (~ -7 to 0). 
# If our energy is raw, we might need to subtract ideal part or reference.
@gp :- data_T data_E "w lp lw 2 pt 7 lc rgb 'black' title 'Total E'"
@gp :- data_T data_K "w lp lw 2 pt 5 lc rgb 'blue' title 'Kinetic'"
@gp :- data_T data_V "w lp lw 2 pt 9 lc rgb 'red' title 'Potential'"
Gnuplot.save("Ceperley1995_Lambda_Energies.png", term="pngcairo size 800,600 enhanced font 'Helvetica,14'")

# Figure 2: Superfluid Fraction (via winding number estimator)
@gp "reset"
@gp :- "set title 'Superfluid Fraction vs Temperature'"
@gp :- "set xlabel 'Temperature (K)'"
@gp :- "set ylabel 'Superfluid Fraction \\rho_s/\\rho'"
@gp :- "set xrange [0:4.5]"
@gp :- "set yrange [0:1.1]"
@gp :- "set grid"
@gp :- data_T data_rhos "w lp lw 2 pt 7 lc rgb 'purple' title 'PIMC'"
Gnuplot.save("Ceperley1995_Lambda_Superfluid.png", term="pngcairo size 800,600 enhanced font 'Helvetica,14'")

# Figure 3: Specific Heat (Ceperley Fig 11)
@gp "reset"
@gp :- "set title 'Specific Heat vs Temperature (Ceperley Fig 11)'"
@gp :- "set xlabel 'Temperature (K)'"
@gp :- "set ylabel 'C (K⁻¹/atom)'"
@gp :- "set yrange [0:5]"
@gp :- "set xrange [0:4]"
@gp :- "set grid"
@gp :- data_T data_Cv "w lp lw 3 pt 7 lc rgb 'red' title 'Specific Heat (Fluctuation)'"
Gnuplot.save("Ceperley1995_LambdaTransition.png", term="pngcairo size 800,600 enhanced font 'Helvetica,14'")

println("Done. Saved 3 aligned plots.")
