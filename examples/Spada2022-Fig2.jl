# Spada2022-Fig2.jl
# Reproduce Figure 2 from Spada et al. 2022
# "Path-integral Monte Carlo worm algorithm for Bose systems with PBC"
#
# Figure 2: N_G/N_Z ratio vs C parameter
# Tests that open/close moves satisfy detailed balance:
# N_G/N_Z = C × G_1/Z_1
#
# Ref: Spada et al. 2022, Section III, Eq. 35

using Halcyon
using Statistics
using Gnuplot
using Printf

# ═══════════════════════════════════════════════════════════════════════════════
# Physical Parameters
# ═══════════════════════════════════════════════════════════════════════════════

const N = 1          # Single particle
const D = 3          # 3D
const L = 1.0        # Box size
const λ = 0.5        # ℏ²/(2m)
const M = 16         # Number of beads (fixed for this test)

# λ_T/L ratios (same as Fig 1)
const λT_L_values = [0.795, 0.974, 1.257, 1.646]
const colors = ["#e41a1c", "#4daf4a", "#ff7f00", "#377eb8"]

# C values to test
const C_values = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

# MC parameters
const EQUILIBRATION_STEPS = 100_000
const MEASUREMENT_STEPS = 5_000_000

# ═══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════

"""
Run simulation and return N_G/N_Z ratio.
"""
function run_simulation(β::Float64, C::Float64; verbose::Bool=false)
    sys = Halcyon.System(
        M, N,
        m = 1.0, D = D, β = β,
        V = Halcyon.HarmonicPotential(k=0.0),
        U = Halcyon.NullPairPotential(),
        λ = λ, L = L
    )
    
    cfg = Halcyon.WormConfiguration(sys)
    params = Halcyon.WormParams(C=C, j_max=M, j_max_open=M, j_max_swap=M, r_max=0.5)
    
    # Equilibration
    for _ in 1:EQUILIBRATION_STEPS
        Halcyon.worm_step!(cfg, sys, params)
    end
    
    # Count time in each sector
    n_Z = 0
    n_G = 0
    
    for _ in 1:MEASUREMENT_STEPS
        Halcyon.worm_step!(cfg, sys, params)
        if cfg.sector == Halcyon.Z_SECTOR
            n_Z += 1
        else
            n_G += 1
        end
    end
    
    if verbose
        @printf("  C=%.2f: N_Z=%d, N_G=%d, ratio=%.4f\n", C, n_Z, n_G, n_G/n_Z)
    end
    
    return n_G / n_Z
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    println("=" ^ 70)
    println("Spada et al. 2022 - Figure 2 Reproduction")
    println("N_G/N_Z ratio vs C (detailed balance test)")
    println("=" ^ 70)
    
    # Results: λT_L => [(C, N_G/N_Z), ...]
    results = Dict{Float64, Vector{Tuple{Float64,Float64}}}()
    exact_ratios = Dict{Float64, Float64}()
    
    for λT_L in λT_L_values
        println("\n" * "-" ^ 50)
        println("λ_T/L = $λT_L")
        println("-" ^ 50)
        
        β = Halcyon.β_from_λT_ratio(λT_L, L, λ)
        
        # Exact G_1/Z_1 ratio
        G1_Z1 = Halcyon.G1_Z1_ratio(λT_L)
        exact_ratios[λT_L] = G1_Z1
        println("Exact G_1/Z_1 = $(round(G1_Z1, digits=6))")
        
        results[λT_L] = Tuple{Float64,Float64}[]
        
        for C in C_values
            ratio = run_simulation(β, C; verbose=true)
            push!(results[λT_L], (C, ratio))
        end
    end
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Plotting
    # ═══════════════════════════════════════════════════════════════════════════
    
    println("\n" * "=" ^ 70)
    println("Generating plots...")
    println("=" ^ 70)
    
    # Figure 2a: N_G/N_Z vs C
    @gp "reset"
    @gp :- "set title 'N_G/N_Z vs C (N=1, M=$M)' font ',14'"
    @gp :- "set xlabel 'C'"
    @gp :- "set ylabel 'N_G / N_Z'"
#    @gp :- "set logscale x" # following Spada
#    @gp :- "set logscale y"
    @gp :- "set key bottom right"
    @gp :- "set grid"
    
    for (idx, λT_L) in enumerate(λT_L_values)
        G1_Z1 = exact_ratios[λT_L]
        data = results[λT_L]
        
        C_arr = [d[1] for d in data]
        ratio_arr = [d[2] for d in data]
        
        # Exact prediction: N_G/N_Z = C × G_1/Z_1
        exact_line = [C * G1_Z1 for C in C_arr]
        
        label = @sprintf("λ_T/L = %.3f", λT_L)
        color = colors[idx]
        
        @gp :- C_arr ratio_arr "with points pt 7 ps 1.2 lc rgb '$color' title '$label'"
        @gp :- C_arr exact_line "with lines lt 2 lw 1.5 lc rgb '$color' notitle"
    end
    
    Gnuplot.save("Spada2022-Fig2a.png", term="pngcairo size 800,600 enhanced font 'Helvetica,14'")
    Gnuplot.save("Spada2022-Fig2a.pdf", term="pdfcairo size 4in,3in enhanced font 'Helvetica,10'")
    
    # Figure 2b: fitted coefficient vs λ_T/L
    @gp "reset"
    @gp :- "set title 'G_1/Z_1 coefficient' font ',14'"
    @gp :- "set xlabel 'λ_T / L'"
    @gp :- "set ylabel 'G_1 / Z_1'"
    @gp :- "set key top left"
    @gp :- "set grid"
    
    # Compute average slope for each λT_L
    λT_L_arr = Float64[]
    measured_slope_arr = Float64[]
    slope_err_arr = Float64[]
    exact_G1Z1_arr = Float64[]
    
    for λT_L in λT_L_values
        G1_Z1 = exact_ratios[λT_L]
        data = results[λT_L]
        
        slopes = [d[2]/d[1] for d in data]  # (N_G/N_Z)/C
        
        push!(λT_L_arr, λT_L)
        push!(measured_slope_arr, mean(slopes))
        push!(slope_err_arr, std(slopes) / sqrt(length(slopes)))
        push!(exact_G1Z1_arr, G1_Z1)
    end
    
    # Perfect agreement line (exact G1/Z1 vs λT/L)
    λT_L_fine = range(minimum(λT_L_arr)*0.9, maximum(λT_L_arr)*1.1, length=100)
    exact_fine = [Halcyon.G1_Z1_ratio(x) for x in λT_L_fine]
    @gp :- λT_L_fine exact_fine "with lines lt 2 lw 2 lc rgb 'gray' title 'Exact'"
    
    @gp :- λT_L_arr measured_slope_arr slope_err_arr "with yerrorbars pt 7 ps 1.5 lc rgb 'black' title 'Simulation'"
    
    Gnuplot.save("Spada2022-Fig2b.png", term="pngcairo size 800,600 enhanced font 'Helvetica,14'")
    Gnuplot.save("Spada2022-Fig2b.pdf", term="pdfcairo size 4in,3in enhanced font 'Helvetica,10'")
    
    println("\nPlots saved: Spada2022-Fig2a.{png,pdf}, Spada2022-Fig2b.{png,pdf}")
end

main()


