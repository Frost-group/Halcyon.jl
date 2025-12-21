# examples/Ceperley1995-Fig22-MomentumParams.jl
# Replicate Ceperley 1995 Figure 22 (Density Matrix) and Figure 24 (Momentum Distribution)
#
# Figure 22: One-body density matrix rho_1(r).
#   - Normal phase (4K): Decays to zero (Gaussian-like).
#   - Superfluid phase (1.18K): Saturates to a finite value n_0 (Condensate Fraction).
#
# Figure 24: Momentum distribution n(k).
#   - Superfluid phase: Divergence at k=0 (Macroscopic occupation).

using Halcyon, Printf, Gnuplot

# ═══════════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════════

const N = 64
const D = 3
const rho = 0.02182 # A^-3
const L = (N / rho)^(1 / 3) # ~18 Å
const λ = 6.0596

# Temperatures
const temperatures = [4.0, 1.18]

# MC parameters
const EQUILIBRATION_STEPS = 100_000
const MEASUREMENT_STEPS = 10_000_000
const MEASURE_INTERVAL = 10
# Timing: ~1.5 hours on M1 Mac (21 Dec 2024) with 10M steps × 2 temps, N=64

const R_BINS = 200
# Grid for density matrix histogram (3D)
const GRID_BINS = 256 # 256^3 grid for FFT (finer resolution)

# ═══════════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════════

function run_simulation(T::Float64)
    β = 1.0 / T
    M = max(20, round(Int, β / 0.04))

    println("\n" * "="^60)
    @printf("Simulation: T = %.2f K (M=%d)\n", T, M)
    println("="^60)

    sys = System(M, N; m=4.0026, D=D, β=β, λ=λ, L=L, U=AzizPotential())
    cfg = WormConfiguration(sys)
    # Increase C to encourage open worms (G-sector)
    params = WormParams(C=2.0, j_max=20, j_max_open=20, j_max_swap=15, r_max=L / 2)

    println("Equilibrating...")
    for step in 1:EQUILIBRATION_STEPS
        worm_step!(cfg, sys, params)
    end

    println("Sampling...")

    # 3D Histogram for rho_1(r)
    rho1_grid = zeros(GRID_BINS, GRID_BINS, GRID_BINS)
    G_samples = 0

    for step in 1:MEASUREMENT_STEPS
        status = worm_step!(cfg, sys, params)

        if step % MEASURE_INTERVAL == 0
            if cfg.sector == G_SECTOR
                accumulate_density_matrix!(rho1_grid, cfg, sys)
                G_samples += 1
            end
        end
    end

    if G_samples > 0
        rho1_grid ./= G_samples
    end

    # Compute n(k)
    k_centers, nk_radial = momentum_distribution(rho1_grid, sys)

    # Compute rho_1(r) radial average for plotting Fig 22
    indices = CartesianIndices(rho1_grid)
    center = GRID_BINS ÷ 2 + 1
    dr = L / GRID_BINS
    r_max_plot = L / 2

    rho1_radial = zeros(R_BINS)
    count_radial = zeros(R_BINS)

    for I in indices
        vec_idx = [I[d] - center for d in 1:3]
        r_vec = vec_idx .* dr
        r_mag = sqrt(sum(abs2, r_vec))

        bin = floor(Int, r_mag / (r_max_plot / R_BINS)) + 1
        if bin >= 1 && bin <= R_BINS
            rho1_radial[bin] += rho1_grid[I]
            count_radial[bin] += 1
        end
    end

    for i in 1:R_BINS
        if count_radial[i] > 0
            rho1_radial[i] /= count_radial[i]
        end
    end

    # Normalize rho1(r) such that rho1(0) is normalized correctly. 
    # Fig 22 plots n(r) / density. 
    # In PIMC, rho_1(0) = rho.
    scale = rho1_radial[1] > 0 ? 1.0 / rho1_radial[1] : 1.0
    rho1_radial .*= scale

    r_gen = [(i - 0.5) * (r_max_plot / R_BINS) for i in 1:R_BINS]

    return r_gen, rho1_radial, k_centers, nk_radial
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ═══════════════════════════════════════════════════════════════════════════════

results = Dict()

for T in temperatures
    r, rho1, k, nk = run_simulation(T)
    results[T] = (r, rho1, k, nk)
end

# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

println("\nGenerating plots...")

# Fig 22: Density Matrix
@gp "reset"
@gp :- "set title 'Single-Particle Density Matrix \\rho_1(r) (Ceperley Fig 22)'"
@gp :- "set xlabel 'r (Å)'"
@gp :- "set ylabel '\\rho_1(r)/\\rho'"
@gp :- "set xrange [0:8]" # Ceperley Fig 22 shows up to approx 8Å
@gp :- "set yrange [0:1.1]"
@gp :- "set grid"

T_normal = temperatures[1]
T_super = temperatures[2]

r_n, rho_n, _, _ = results[T_normal]
r_s, rho_s, _, _ = results[T_super]

@gp :- r_n rho_n "w l lw 3 lc rgb 'red' title 'T=$(T_normal)K (Normal)'"
@gp :- r_s rho_s "w l lw 3 lc rgb 'blue' title 'T=$(T_super)K (Superfluid)'"

Gnuplot.save("Ceperley1995_Fig22_DensityMatrix.png", term="pngcairo size 800,600 enhanced font 'Helvetica,14'")

# Fig 24: Momentum Distribution
@gp "reset"
@gp :- "set title 'Momentum Distribution n(k) (Ceperley Fig 24)'"
@gp :- "set xlabel 'k (Å⁻¹)'"
@gp :- "set ylabel 'n(k)'"
@gp :- "set xrange [0:4]"
@gp :- "set yrange [0:0.12]"
@gp :- "set grid"

_, _, k_n, nk_n = results[T_normal]
_, _, k_s, nk_s = results[T_super]

# Matching normalization roughly to Fig 24 (n(0) ~ 0.1)
nk_n_scaled = nk_n ./ sum(nk_n) .* 0.5
nk_s_scaled = nk_s ./ sum(nk_s) .* 0.5

@gp :- k_n nk_n_scaled "w l lw 3 lc rgb 'red' title 'T=$(T_normal)K'"
@gp :- k_s nk_s_scaled "w l lw 3 lc rgb 'blue' title 'T=$(T_super)K'"

Gnuplot.save("Ceperley1995_Fig24_Momentum.png", term="pngcairo size 800,600 enhanced font 'Helvetica,14'")

println("Done. Saved Ceperley1995_Fig22_DensityMatrix.png and Ceperley1995_Fig24_Momentum.png")
