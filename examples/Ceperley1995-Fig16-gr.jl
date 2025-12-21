# examples/Ceperley1995-Fig16-gr.jl
# Replicate Ceperley 1995 Figure 16: Radial Distribution Function g(r)
#
# Validates the Aziz potential and liquid structure.
# At T=1.21K (Superfluid), g(r) shows a correlation hole at short distances
# and a peak around 3.6 Angstroms.

using Halcyon, Printf, Gnuplot

# ═══════════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════════

const N = 64
const D = 3
# Density rho = 0.0218 A^-3 (SVP at low T)
const rho = 0.02182 # More precise SVP density
const L = (N / rho)^(1 / 3)
const λ = 6.0596 # Å² K

const T = 1.21 # K (Superfluid)
const β = 1.0 / T
# M for tau ~ 0.05 (reduced for speed given N increase, but pair action would be better)
const M = 20

const EQUILIBRATION_STEPS = 100_000
const MEASUREMENT_STEPS = 20_000_000
const MEASURE_INTERVAL = 5
# Timing: ~4-6 hours on M1 Mac (21 Dec 2024) with 20M steps, N=64

# ═══════════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════════

function run_simulation()
    println("\n" * "="^60)
    @printf("Simulation: T = %.2f K, N = %d, L = %.2f Å\n", T, N, L)
    println("="^60)

    # Note: Using AzizPotential()
    sys = System(M, N; m=4.0026, D=D, β=β, λ=λ, L=L, U=AzizPotential())
    cfg = WormConfiguration(sys)
    params = WormParams(C=1.0, j_max=20, j_max_open=20, j_max_swap=15, r_max=L / 2)

    println("Equilibrating ($EQUILIBRATION_STEPS steps)...")
    for step in 1:EQUILIBRATION_STEPS
        worm_step!(cfg, sys, params)
    end

    println("Sampling ($MEASUREMENT_STEPS steps)...")

    nbins = 200
    r_max = 8.5 # Ceperley plots to 8.0, so 8.5 is safe
    gr_sum = zeros(nbins)
    samples = 0

    for step in 1:MEASUREMENT_STEPS
        worm_step!(cfg, sys, params)

        if step % MEASURE_INTERVAL == 0
            if cfg.sector == Z_SECTOR
                r_centers, gr = radial_distribution(cfg, sys; nbins=nbins, r_max=r_max)
                gr_sum .+= gr
                samples += 1
            end
        end
    end

    @printf("  Samples: %d (%.1f%% in Z-sector)\n", samples, 100 * samples / (MEASUREMENT_STEPS / MEASURE_INTERVAL))

    gr_mean = samples > 0 ? gr_sum ./ samples : zeros(nbins)
    dr = r_max / nbins
    r_centers = [(i - 0.5) * dr for i in 1:nbins]

    return r_centers, gr_mean
end

# ═══════════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════════

r, gr = run_simulation()

# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

println("\nGenerating plot...")
@gp "reset"
@gp :- "set title 'Radial Distribution Function g(r) (Ceperley Fig 16) at T=$(T)K'"
@gp :- "set xlabel 'r (Å)'"
@gp :- "set ylabel 'g(r)'"
@gp :- "set xrange [0:8]"
@gp :- "set yrange [0:1.5]"
@gp :- "set grid"
@gp :- "set style data lines"

@gp :- r gr "lw 3 lc rgb 'blue' title 'PIMC (Halcyon.jl)'"

# Save using Gnuplot.save
Gnuplot.save("Ceperley1995_Fig16_gr.png", term="pngcairo size 800,600 enhanced font 'Helvetica,14'")
println("Done. Saved to Ceperley1995_Fig16_gr.png")
