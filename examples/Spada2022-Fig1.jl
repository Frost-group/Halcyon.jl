# Spada2022-Fig1.jl
# Reproduce Figure 1 from Spada et al. 2022
# "Path-integral Monte Carlo worm algorithm for Bose systems with PBC"
# Condens. Matter 2022, 7, 30 (arXiv:2203.00010)
#
# Figure 1: N=1 system at different λ_T/L values
# Left panel: Internal energy E vs number of beads M
# Right panel: Difference from exact values
#
# Tests all worm moves EXCEPT swap (only 1 particle)

using Halcyon
using Statistics
using Gnuplot
using Printf

# ═══════════════════════════════════════════════════════════════════════════════
# Physical Parameters
# ═══════════════════════════════════════════════════════════════════════════════

const N = 1          # Single particle
const D = 3          # 3D
const L = 1.0        # Box size (sets length scale)
const λ = 0.5        # ℏ²/(2m), sets energy scale
const m_particle = 1.0

# λ_T/L ratios to test (as in Fig. 1)
const λT_L_values = [0.795, 0.974, 1.257, 1.646]
# Colors and symbols for different λ_T/L
const colors = ["#e41a1c", "#4daf4a", "#ff7f00","#377eb8"]  # Red, Green, Orange, Blue
const symbols = ["circle", "triangledown", "triangleup", "square"]

# Number of beads to test
const M_values = [1, 2, 4, 8, 16, 32]

# MC parameters. 50k/200k/100 is very fast but noisy at large lambda
const EQUILIBRATION_STEPS = 500_000
const MEASUREMENT_STEPS = 10_000_000
const MEASURE_INTERVAL = 1000  # Measure every N steps

# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

"""
Run PIMC simulation for given parameters and return mean energy with error.
"""
function run_simulation(M::Int, β::Float64; verbose::Bool=false)
    # Create system
    sys = Halcyon.System(
        M, N,
        m = m_particle,
        D = D,
        β = β,
        V = Halcyon.HarmonicPotential(k=0.0),  # No external potential (free particle)
        U = Halcyon.NullPairPotential(),       # No interactions
        λ = λ,
        L = L
    )
    
    # Create worm configuration
    cfg = Halcyon.WormConfiguration(sys)
    
    # Worm parameters - set j_max to M for unconstrained moves (as in paper)
    params = Halcyon.WormParams(
        C = 1.0,
        j_max = M,
        j_max_open = max(1, M - 1),
        j_max_swap = M,
        r_max = min(0.5, L / 2)
    )
    
    # Equilibration
    if verbose
        println("  Equilibrating for $EQUILIBRATION_STEPS steps...")
    end
    for _ in 1:EQUILIBRATION_STEPS
        Halcyon.worm_step!(cfg, sys, params)
    end
    
    # Measurement
    if verbose
        println("  Measuring for $MEASUREMENT_STEPS steps...")
    end
    
    energies = Float64[]
    n_measurements = 0
    n_Z_samples = 0
    
    for step in 1:MEASUREMENT_STEPS
        Halcyon.worm_step!(cfg, sys, params)
        
        # Only measure in Z-sector (diagonal configurations)
        if step % MEASURE_INTERVAL == 0 && cfg.sector == Halcyon.Z_SECTOR
            E = Halcyon.energy_thermodynamic(cfg, sys)
            push!(energies, E)
            n_measurements += 1
        end
        
        if cfg.sector == Halcyon.Z_SECTOR
            n_Z_samples += 1
        end
    end
    
    if verbose
        Z_fraction = n_Z_samples / MEASUREMENT_STEPS
        println("  Z-sector fraction: $(round(Z_fraction, digits=3))")
        println("  Measurements collected: $n_measurements")
    end
    
    if length(energies) < 10
        @warn "Too few measurements in Z-sector" n=length(energies)
        return NaN, NaN
    end
    
    # Compute mean and standard error
    E_mean = mean(energies)
    E_std = std(energies) / sqrt(length(energies))
    
    return E_mean, E_std
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main Simulation
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    println("=" ^ 70)
    println("Spada et al. 2022 - Figure 1 Reproduction")
    println("N=1 internal energy vs number of beads M")
    println("=" ^ 70)
    
    # Storage for results
    # results[i][j] = (E_mean, E_std) for λT_L_values[i] and M_values[j]
    results = Dict{Float64, Vector{Tuple{Float64,Float64}}}()
    exact_energies = Dict{Float64, Float64}()
    
    for λT_L in λT_L_values
        println("\n" * "-" ^ 50)
        println("λ_T/L = $λT_L")
        println("-" ^ 50)
        
        # Compute β from λ_T/L ratio
        β = Halcyon.β_from_λT_ratio(λT_L, L, λ)
        println("β = $(β)")
        
        # Exact energy
        E_exact = Halcyon.E1_exact(β, L, λ)
        exact_energies[λT_L] = E_exact
        println("Exact E = $(round(E_exact, digits=6))")
        
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
    # Plotting with Gnuplot
    # ═══════════════════════════════════════════════════════════════════════════
    
    println("\n" * "=" ^ 70)
    println("Generating plots...")
    println("=" ^ 70)
    
    # Energy scale: use k_B T_c^0 as in paper
    # For N=1 in a box, we just use the thermal energy 1/β as scale
    # But paper uses T_c^0 which requires density n = N/V
    n = N / L^D
    T_c0 = Halcyon.critical_temperature(n, λ)
    E_scale = T_c0  # Energy in units of k_B T_c^0
    
    println("Energy scale: k_B T_c^0 = $(round(E_scale, digits=6))")
    
    # Prepare data for plotting
    M_plot = Float64.(M_values)
    

    
    # ─────────────────────────────────────────────────────────────────────────
    # Figure 1a: Energy vs M
    # ─────────────────────────────────────────────────────────────────────────
    
    @gp "reset"
    @gp :- "set title 'Internal Energy E (N=1)' font ',14'"
    @gp :- "set xlabel 'Number of beads M'"
    @gp :- "set ylabel 'E / (N k_B T_c^0)'"
 #   @gp :- "set logscale x 2" # it's nicer like this, but Spada use a linear scale
    @gp :- "set logscale y" 
    @gp :- "set xtics (1,2,4,8,16,32)"
    @gp :- "set xrange [0:35]" # again, following Spada
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
    
    Gnuplot.save("Spada2022-Fig1a.png", term="pngcairo size 800,600 enhanced font 'Helvetica,14'")
    Gnuplot.save("Spada2022-Fig1a.pdf", term="pdfcairo size 4in,3in enhanced font 'Helvetica,10'")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Figure 1b: Difference from exact
    # ─────────────────────────────────────────────────────────────────────────

    @gp "reset"


    @gp :- "set title 'Difference from Exact (N=1)' font ',14'"
    @gp :- "set xlabel 'Number of beads M'"
    @gp :- "set ylabel '(E - E_{exact}) / (N k_B T_c^0)'"
 #   @gp :- "set logscale x 2" # it's nicer like this, but Spada use a linear scale 
    @gp :- "set xtics (1,2,4,8,16,32)"
    @gp :- "set xrange [0:35]" # again, following Spada
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
    
    Gnuplot.save("Spada2022-Fig1b.png", term="pngcairo size 800,600 enhanced font 'Helvetica,14'")
    Gnuplot.save("Spada2022-Fig1b.pdf", term="pdfcairo size 4in,3in enhanced font 'Helvetica,10'")
end

# Run
main()

