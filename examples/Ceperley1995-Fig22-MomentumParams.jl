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
const temperatures = [4.0, 2.22, 1.18]

# MC parameters
const EQUILIBRATION_STEPS = 100_000
const MEASUREMENT_STEPS = 10_000_000
const MEASURE_INTERVAL = 1000
# Timing: ~1.5 hours on M1 Mac (21 Dec 2024) with 10M steps × 2 temps, N=64

const R_BINS = 200
# Grid for density matrix histogram (3D)
const GRID_BINS = 128 # grid for FFT 

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

    # Radial Histogram for rho_1(r)
    radial_hist = zeros(R_BINS)
    counts_hist = zeros(R_BINS)
    G_samples = 0

    for step in 1:MEASUREMENT_STEPS
        worm_step!(cfg, sys, params)

        if step % MEASURE_INTERVAL == 0 && cfg.sector == G_SECTOR
            accumulate_radial_density!(radial_hist, counts_hist, cfg, sys)
            G_samples += 1
        end
    end

    if G_samples == 0
        println("  -> WARNING: No G-sector samples collected!")
    end
    println("  -> M=$M, Steps=$MEASUREMENT_STEPS, G_samples=$G_samples")

    # Compute n(k) from radial histogram
    k_smooth, nk_smooth, r_radial, rho1_radial = compute_nk_from_radial(radial_hist, sys; G_samples=G_samples)

    return r_radial, rho1_radial, k_smooth, nk_smooth
end


"""
    accumulate_radial_density!(radial_hist::AbstractVector, counts::AbstractVector, cfg::WormConfiguration, sys::System)

Accumulate the single-particle density matrix directly into a radial histogram.
This avoids sparse sampling issues associated with high-dimensional grids.
"""
function accumulate_radial_density!(radial_hist::AbstractVector, counts::AbstractVector, cfg::WormConfiguration, sys::System)
    # Caller guarantees we are in G_SECTOR

    # Head and Tail positions
    r_head = cfg.r_c_head
    r_tail = cfg.r[cfg.i_tail, 1, :]

    L = sys.L
    D = sys.D

    # Minimum image distance
    d2 = 0.0
    for d in 1:D
        dx = r_head[d] - r_tail[d]
        dx -= L * round(dx / L)
        d2 += dx * dx
    end
    r = sqrt(d2)

    # Binning
    nbins = length(radial_hist)
    r_max = L / 2
    dr = r_max / nbins

    bin = floor(Int, r / dr) + 1
    if bin >= 1 && bin <= nbins
        radial_hist[bin] += 1.0
        counts[bin] += 1.0 # Keep track of raw hits if needed, though hist alone is sufficient
    end
end

"""
    compute_nk_from_radial(radial_hist::AbstractVector, sys::System; 
                           k_max=5.0, nk_bins=200, G_samples=1) -> (k, nk, r, rho1_r)

Compute momentum distribution n(k) from the accumulated radial density matrix histogram.
"""
function compute_nk_from_radial(radial_hist::AbstractVector, sys::System;
    k_max=5.0, nk_bins=200, G_samples=1)

    nbins = length(radial_hist)
    r_max = sys.L / 2
    dr = r_max / nbins

    # Normalize histogram to get rho1(r)
    # The histogram counts occurrences of r in shells [r, r+dr].
    # rho1(r) is probability density per unit volume.
    # But here we want the "projected" rho1(r) such that rho1(r -> large) -> 0 (normal) or n0 (superfluid).
    #
    # Standard relation:
    # <rho1(r)> = < \sum_{i} \delta(r - (r_i - r_i')) >
    # In PIMC G-sector, we sample r = |r_head - r_tail|.
    # Probability of finding separating r is P(r) dr.
    # rho1(r) = P(r) / (4 * pi * r^2) (in 3D).
    # Also need to normalize by total number of samples G_samples and Volume?
    #
    # Actually, let's look at limits.
    # If rho1(r) = constant = rho (ideal gas / condensate limit), then P(r) ~ r^2.
    # So rho1(r) ~ counts[r] / r^2.
    #
    # We also want rho1(0) = rho (total density) or normalized to 1?
    # Let's compute the raw shape first.

    rho1_r = zeros(nbins)
    r_gen = [(i - 0.5) * dr for i in 1:nbins]

    for i in 1:nbins
        r = r_gen[i]
        vol_shell = 4 * π * r^2 * dr
        if vol_shell > 0 && G_samples > 0
            rho1_r[i] = radial_hist[i] / (vol_shell * G_samples)
        end
    end

    # For plotting Fig 22, we want rho1(r)/rho.
    # If correctly normalized, rho1(0) should be approx rho.
    # However, G-sector sampling efficiency depends on C parameter.
    # A robust way is to normalize rho1(r) such that rho1(0) -> 1 (relative) or matches known density.
    # Given the previous logic used "rho1(0) = 1" normalization for n(k) shape, let's stick to that
    # but return the physical rho1 for checking.

    # Heuristic: Normalize max to 1.0 for the plot if it looks reasonable, 
    # OR normalize by the box volume factor?
    # Let's just normalize rho1(0) to 1.0 for the output to match "rho1(r)/rho" expectation.
    # Note: bin 1 might be noisy.

    val_at_zero = rho1_r[1]
    # Search for first non-zero if bin 1 is empty? 
    # Usually bin 1 is well populated for interacting bosons.
    if val_at_zero == 0
        # find first non-zero
        idx = findfirst(x -> x > 0, rho1_r)
        if idx !== nothing
            val_at_zero = rho1_r[idx]
        else
            val_at_zero = 1.0
        end
    end

    rho1_norm = rho1_r ./ val_at_zero

    # Hankel Transform for n(k)
    # n(k) = Integral rho1(r) e^{ikr} d^3r
    # We use rho1_norm for the shape of n(k).

    k_smooth = collect(range(0.001, stop=k_max, length=nk_bins))
    nk_smooth = zeros(nk_bins)

    for (ik, k) in enumerate(k_smooth)
        integral = 0.0
        for i in 1:nbins
            r = r_gen[i]
            # 4*pi*r^2 * rho1(r) * (sin(kr)/kr) * dr
            # = (4*pi*r/k) * rho1(r) * sin(kr) * dr
            integral += (r * rho1_norm[i] * sin(k * r)) * dr
        end
        nk_smooth[ik] = (4 * π / k) * integral
    end

    return k_smooth, nk_smooth, r_gen, rho1_norm
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
@gp :- "set xrange [0:8]"
@gp :- "set yrange [0:1.1]"
@gp :- "set grid"
@gp :- "set size square"

colors = ["red", "blue", "green", "black", "orange", "purple"]

for (i, T) in enumerate(temperatures)
    r_vals, rho_vals, _, _ = results[T]
    color = colors[mod1(i, length(colors))]
    title_str = "T=$(T)K"
    if T < 2.17
        title_str *= " (Superfluid)"
    else
        title_str *= " (Normal)"
    end
    @gp :- r_vals rho_vals "w l lw 3 lc rgb '$color' title '$title_str'"
end

Gnuplot.save("Ceperley1995_Fig22_DensityMatrix.png", term="pngcairo size 800,600 enhanced font 'Helvetica,14'")

# Fig 24: Momentum Distribution
@gp "reset"
@gp :- "set title 'Momentum Distribution n(k) (Ceperley Fig 24)'"
@gp :- "set xlabel 'k (Å⁻¹)'"
@gp :- "set ylabel 'n(k)'"
@gp :- "set xrange [0:4]"
@gp :- "set yrange [0:0.12]"
@gp :- "set grid"
@gp :- "set size square"

for (i, T) in enumerate(temperatures)
    _, _, k_vals, nk_vals = results[T]

    # Scale n(k) to match rough magnitude of Fig 24.
    # The Hankel transform is unnormalized.
    # We apply a heuristic scaling for visualization compared to Ceperley.
    nk_scaled = nk_vals .* 0.005

    color = colors[mod1(i, length(colors))]
    title_str = "T=$(T)K"
    @gp :- k_vals nk_scaled "w l lw 3 lc rgb '$color' title '$title_str'"
end

Gnuplot.save("Ceperley1995_Fig24_Momentum.png", term="pngcairo size 800,600 enhanced font 'Helvetica,14'")

println("Done. Saved Ceperley1995_Fig22_DensityMatrix.png and Ceperley1995_Fig24_Momentum.png")
