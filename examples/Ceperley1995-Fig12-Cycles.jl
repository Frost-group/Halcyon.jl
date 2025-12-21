# examples/Ceperley1995-Fig12-Cycles.jl
# Replicate Ceperley 1995 Figure 12: Cycle Length Distribution P(m)
#
# P(m) is the probability that an atom belongs to a permutation cycle of length m.
# In the Normal phase (T > T_lambda), P(m) decays exponentially.
# In the Superfluid phase (T < T_lambda), P(m) becomes broad (delocalization).
#
# We use N=64, M=80 (approx) for better statistics than the N=6 visualizer.

using Halcyon, Printf, Gnuplot

# ═══════════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════════

const N = 64
const D = 3
const rho = 0.02182
const L = (N / rho)^(1 / 3)
const λ = 6.0596 # Å² K

# Temperatures to compare (Lambda transition is ~2.17K)
const temperatures = [1.0, 1.25, 1.5, 1.75, 2.0, 2.17, 2.4, 2.8, 3.2, 3.6, 4.0]

# MC parameters
const EQUILIBRATION_STEPS = 100_000
const MEASUREMENT_STEPS = 10_000_000
const MEASURE_INTERVAL = 5
# Timing: ~3 hours on M1 Mac (21 Dec 2024) with 10M steps × 11 temps, N=64

# ═══════════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════════

function run_point(T::Float64)
    β = 1.0 / T
    M = max(20, round(Int, β / 0.04))

    println("\n" * "="^60)
    @printf(" T = %.2f K (M=%d)\n", T, M)
    println("="^60)

    sys = System(M, N; m=4.0026, D=D, β=β, λ=λ, L=L, U=AzizPotential())
    cfg = WormConfiguration(sys)
    params = WormParams(C=1.0, j_max=20, j_max_open=20, j_max_swap=15, r_max=L / 2)

    # Equilibrating
    for step in 1:EQUILIBRATION_STEPS
        worm_step!(cfg, sys, params)
    end

    # Measurement counts: total_counts[m]
    total_counts = zeros(Int, N)
    samples_count = 0

    for step in 1:MEASUREMENT_STEPS
        worm_step!(cfg, sys, params)

        if step % MEASURE_INTERVAL == 0 && cfg.sector == Z_SECTOR
            visited = zeros(Bool, N)
            for i in 1:N
                if !visited[i]
                    cycle = get_cycle(cfg, i)
                    m = length(cycle)
                    total_counts[m] += 1
                    for p in cycle
                        visited[p] = true
                    end
                end
            end
            samples_count += 1
        end
    end

    # P(m) = m * <N_m> / N
    probs = Float64[]
    for m in 1:5 # Plot m=1 to 5 as in Fig 12
        avg_Nm = total_counts[m] / samples_count
        pm = (m * avg_Nm) / N
        push!(probs, pm)
    end

    return probs
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ═══════════════════════════════════════════════════════════════════════════════

data_T = Float64[]
data_Pm = [Float64[] for _ in 1:5]

for T in temperatures
    pm_T = run_point(T)
    push!(data_T, T)
    for m in 1:5
        push!(data_Pm[m], pm_T[m])
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

println("\nGenerating plot...")
@gp "reset"
@gp :- "set title 'Cycle Length Probability P_m(T) (Ceperley Fig 12)'"
@gp :- "set xlabel 'T (K)'"
@gp :- "set ylabel 'P_m'"
@gp :- "set logscale y"
@gp :- "set yrange [1e-4:1.1]"
@gp :- "set xrange [1:4]"
@gp :- "set grid"

# Plot P_1 (Monomers) - Solid Line
@gp :- data_T data_Pm[1] "w lp lw 3 pt 7 lc rgb 'black' title 'm=1'"

# Plot P_m for m=2,3,4,5 - Dashed Lines
colors = ["red", "blue", "green", "orange"]
for m in 2:5
    @gp :- data_T data_Pm[m] "w lp lw 2 pt 6 dt $(m-1) lc rgb '$(colors[m-1])' title 'm=$m'"
end

Gnuplot.save("Ceperley1995_Fig12_CycleLengths.png", term="pngcairo size 800,600 enhanced font 'Helvetica,14'")
println("Done. Saved to Ceperley1995_Fig12_CycleLengths.png")
