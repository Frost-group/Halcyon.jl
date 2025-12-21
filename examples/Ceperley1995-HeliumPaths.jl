# examples/Ceperley1995-HeliumPaths.jl
# Replicate Ceperley 1995 Figure 10: Paths of Helium-4 atoms
# Demonstrating the emergence of long-range exchanges at low temperatures.

using Halcyon, Printf, Gnuplot

# ═══════════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════════

const N = 6
const M = 53
const D = 2 # 2D simulation for Figure 10
const λ = 6.0592 # Å² K (for 4He)
const L_box = 12.0 # Å (approximate from Fig 10)
const temperatures = [2.5, 1.25, 0.75] # K

# MC parameters
const EQUILIBRATION_STEPS = 500_000
const MEASUREMENT_STEPS = 1_000_000

# ═══════════════════════════════════════════════════════════════════════════════
# Simulation and Plotting
# ═══════════════════════════════════════════════════════════════════════════════

function run_simulation(T::Float64)
    β = 1.0 / T
    println("\n" * "="^50)
    @printf("Simulation: T = %.2f K, β = %.3f K⁻¹\n", T, β)
    println("="^50)

    # Initialize System with Aziz Potential
    sys = System(M, N; m=1.0, D=D, β=β, λ=λ, L=L_box, U=AzizPotential())
    cfg = WormConfiguration(sys)

    # Worm parameters
    params = WormParams(C=10.0, j_max=M, j_max_open=M, j_max_swap=M, r_max=L_box / 2)

    println("Equilibrating ($EQUILIBRATION_STEPS steps)...")
    for step in 1:EQUILIBRATION_STEPS
        worm_step!(cfg, sys, params)
        if step % 100_000 == 0
            @printf("  Step %d...\n", step)
        end
    end

    println("Sampling ($MEASUREMENT_STEPS steps)...")
    rhos_sum = 0.0
    Z_count = 0
    for step in 1:MEASUREMENT_STEPS
        worm_step!(cfg, sys, params)
        if cfg.sector == Z_SECTOR
            rhos = superfluid_fraction(cfg, sys)
            rhos_sum += rhos
            Z_count += 1
        end
    end

    rhos_mean = Z_count > 0 ? rhos_sum / Z_count : 0.0
    @printf("Superfluid Fraction ρ_s/ρ = %.4f (Samples: %d)\n", rhos_mean, Z_count)

    # Ensure we end in Z-sector for visualization
    println("Closing worm if present...")
    count = 0
    while cfg.sector != Z_SECTOR && count < 1000
        worm_step!(cfg, sys, params)
        count += 1
    end

    println("Generating plot...")
    plot_paths(cfg, sys, T)
end

function plot_paths(cfg::WormConfiguration, sys::System, T::Float64)
    L = sys.L
    half_L = L / 2

    @gp "reset"
    @gp :- "set size ratio 1"
    @gp :- "set xlabel 'x (Å)'"
    @gp :- "set ylabel 'y (Å)'"
    @gp :- "set title '4He Paths (Now FULLY 2D!) at T = $T K (N=6, M=53)'"

    # Range to see winding paths and replicas; follow Fig 10 Cepereley
    @gp :- "set xrange [-10:10]"
    @gp :- "set yrange [-10:10]"
    @gp :- "set grid"

    # Draw fundamental cell boundary (x-y plane)
    @gp :- "set obj 1 rect from -$half_L,-$half_L to $half_L,$half_L dt 2 lw 2 fs empty border lc rgb \"#000000\""

    visited = zeros(Bool, sys.N)

    cycle_lengths = []
    color_id = 1
    for i in 1:sys.N
        visited[i] && continue

        cycle = get_cycle(cfg, i)
        push!(cycle_lengths, length(cycle))
        for p in cycle
            visited[p] = true
        end

        # Extract unwrapped path (3D points)
        # pts_3d[1, :] is bead 0 of the first particle in the cycle
        pts_3d = extract_extended_path(cfg, sys, i)

        # Shift so the entire cycle is relative to the fundamental cell [-L/2, L/2]
        # Calculate shift based on the first bead of the cycle
        shift_x = floor((pts_3d[1, 1] + half_L) / L) * L
        shift_y = floor((pts_3d[1, 2] + half_L) / L) * L

        px = pts_3d[:, 1] .- shift_x
        py = pts_3d[:, 2] .- shift_y

        # Plot path projection and replicas in neighboring unit cells
        for ox in -1:1, oy in -1:1
            off_x = ox * L
            off_y = oy * L

            # Plot the continuous trace with consistent color_id
            @gp :- px .+ off_x py .+ off_y "with lines lw 2 lc $color_id notitle"

            # Mark bead 0 positions for every particle in the cycle
            for k in 0:(length(cycle)-1)
                idx = k * M + 1
                x0 = pts_3d[idx, 1] - shift_x + off_x
                y0 = pts_3d[idx, 2] - shift_y + off_y

                # Only plot dots that are roughly within the viewing range to avoid clutter
                if -15 < x0 < 15 && -15 < y0 < 15
                    @gp :- [x0] [y0] "with points pt 7 ps 1.0 lc \"black\" notitle"
                end
            end
        end
        color_id += 1
    end

    @printf("Cycle lengths: %s\n", join(cycle_lengths, ", "))

    fname = @sprintf("Ceperley1995_HeliumPaths_T%.2f.png", T)
    Gnuplot.save(fname, term="pngcairo size 800,800 enhanced font 'Helvetica,12'")
    println("Saved plot to $fname")
end

# Main loop
for T in temperatures
    run_simulation(T)
end
