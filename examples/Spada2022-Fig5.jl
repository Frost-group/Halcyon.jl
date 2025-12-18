# Spada2022-Fig5.jl
# Reproduce Figure 5 from Spada et al. 2022
# "Path-integral Monte Carlo worm algorithm for Bose systems with PBC"
#
# Figure 5: Hard-sphere gas at na³=0.1
# Internal energy E vs number of beads M
#
# Ref: Spada et al. 2022, Section IV

using Distributed

# ═══════════════════════════════════════════════════════════════════════════════
# Parallel Setup
# ═══════════════════════════════════════════════════════════════════════════════

if nworkers() < 6
    addprocs(6)
end

@everywhere using Halcyon, Statistics, Printf

@everywhere begin
    # Physical Parameters
    const D = 3
    const λ = 0.5
    const gas_parameter = 0.1  # na³
    const a_hs = 1.0           # Hard sphere diameter
    const n_target = gas_parameter / a_hs^3

    # MC parameters
    const EQUILIBRATION_STEPS = 10_000
    const MEASUREMENT_STEPS = 100_000
    const MEASURE_INTERVAL = 100

    function run_simulation(N::Int, M::Int, β::Float64, L_box::Float64; verbose::Bool=false)
        sys = Halcyon.System(
            M, N,
            m = 1.0, D = D, β = β,
            V = Halcyon.HarmonicPotential(k=0.0),
            U = Halcyon.HardSpherePotential(a=a_hs),
            λ = λ, L = L_box
        )
        
        cfg = Halcyon.WormConfiguration(sys)
        params = Halcyon.WormParams(C=1.0, j_max=M, j_max_open=M, j_max_swap=M, r_max=min(0.5, L_box/2))
            
        if verbose
            @printf("  [Worker %d] N=%d, M=%d: Equilibrating...\n", myid(), N, M)
        end
        for _ in 1:EQUILIBRATION_STEPS
            Halcyon.worm_step!(cfg, sys, params)
        end
        
        energies_th = Float64[]
        energies_vi = Float64[]
        n_Z = 0
        
        measureASAP = false
        for step in 1:MEASUREMENT_STEPS
            Halcyon.worm_step!(cfg, sys, params)
                    
            if step % MEASURE_INTERVAL == 0 
                measureASAP = true
            end
            
            if measureASAP && cfg.sector == Halcyon.Z_SECTOR
                push!(energies_th, Halcyon.energy_thermodynamic(cfg, sys) / N)
                push!(energies_vi, Halcyon.energy_virial(cfg, sys) / N)
                measureASAP = false
            end
            
            if cfg.sector == Halcyon.Z_SECTOR
                n_Z += 1
            end
        end
        
        if isempty(energies_th)
            return (NaN, NaN), (NaN, NaN)
        end
        
        res_th = (mean(energies_th), std(energies_th) / sqrt(length(energies_th)))
        res_vi = (mean(energies_vi), std(energies_vi) / sqrt(length(energies_vi)))
        
        if verbose
            @printf(" N=%d, M=%d: Done (Z-sector %.1f%%) %d measurements\n", 
            N, M, 100*n_Z/MEASUREMENT_STEPS, length(energies_th))
        end
        return res_th, res_vi
    end
end

using Gnuplot

# ═══════════════════════════════════════════════════════════════════════════════
# Plotting Configuration (Master Process)
# ═══════════════════════════════════════════════════════════════════════════════

const T_ratio_values = [1.0, 0.5]  # Fig 5a and 5b
const N_values = [256, 128, 64, 16] 
const colors = ["#377eb8", "#ff7f00", "#4daf4a", "#e41a1c"]  # Blue, Orange, Green, Red
const symbols_filled = [7, 11, 9, 5] # Circle, DownTriangle, UpTriangle, Square
const symbols_open = [6, 10, 8, 4]   # Corresponding open markers

const M_values = [4, 8, 16, 32, 48]

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

function main()
    println("=" ^ 70)
    println("Spada et al. 2022 - Figure 5 Reproduction (PARALLEL)")
    println("Hard-sphere gas (na³=0.1) energy vs beads M")
    println("=" ^ 70)
    
    T_c0 = Halcyon.critical_temperature(n_target, λ)
    println("Target density n = $n_target")
    println("T_c0 = $(round(T_c0, digits=6))")
    
    # 1. Prepare parameter list for pmap
    params_list = []
    for T_ratio in T_ratio_values
        β = 1.0 / (T_ratio * T_c0)
        for N in N_values
            L_box = (N / n_target)^(1/3)
            for M in M_values
                push!(params_list, (T_ratio, N, M, β, L_box))
            end
        end
    end
   
    # Sort by complexity (N^2 * M) ; pmap then iterates through the list in order
    sort!(params_list, by=p -> p[2]^2 * p[3], rev=true)
    # Major leagues first, then back fill small tasks
    #  The opposite to what I was doing on single CPU, as I wanted to see the data come back live...


    println("Starting parallel execution of $(length(params_list)) simulations...")
    
    # 2. Execute in parallel
    results_flat = pmap(p -> (p, run_simulation(p[2], p[3], p[4], p[5]; verbose=true)), params_list)
    
    # 3. Reconstruct results dictionary
    # results: (T_ratio, N) => [(M, thermo_res, virial_res), ...]
    results = Dict{Tuple{Float64, Int}, Vector{Tuple{Int, Tuple{Float64, Float64}, Tuple{Float64, Float64}}}}()
    
    for (p, res) in results_flat
        T_ratio, N, M, _, _ = p
        res_th, res_vi = res
        key = (T_ratio, N)
        if !haskey(results, key)
            results[key] = []
        end
        push!(results[key], (M, res_th, res_vi))
    end
    
    # Sort results by M for each key
    for key in keys(results)
        sort!(results[key], by=x->x[1])
    end
    
    # 4. Plotting
    println("\n" * "=" ^ 70)
    println("Generating plots...")
    println("=" ^ 70)
    
    E_scale = T_c0
    
    for T_ratio in T_ratio_values
        suffix = T_ratio == 1.0 ? "a" : "b"
        title = @sprintf("T = %.1f T_c^0", T_ratio)
        
        @gp "reset"
        @gp :- "set title '$title' font ',14'"
        @gp :- "set xlabel 'M'"
        @gp :- "set ylabel 'E / (N k_B T_c^0)'"
        @gp :- "set key outside right bottom vertical enhanced"
        @gp :- "set grid"
        @gp :- "set xtics (4,8,16,32,48)"
        
        for (n_idx, N) in enumerate(N_values)
            data = results[(T_ratio, N)]
            M_arr = [d[1] for d in data]
            
            # Thermodynamic (Dashed/Open)
            E_th_arr = [d[2][1] / E_scale for d in data]
            
            # Virial (Solid/Filled)
            E_vi_arr = [d[3][1] / E_scale for d in data]
            err_vi_arr = [d[3][2] / E_scale for d in data]
            
            color = colors[n_idx]
            sym_f = symbols_filled[n_idx]
            sym_o = symbols_open[n_idx]
            
            # Virial (Solid)
            @gp :- Float64.(M_arr) E_vi_arr err_vi_arr "with yerrorbars pt $sym_f ps 1.2 lc rgb '$color' title 'N=$N (Virial)'"
            @gp :- Float64.(M_arr) E_vi_arr "with lines lc rgb '$color' notitle"
            
            # Thermodynamic (Dashed)
            @gp :- Float64.(M_arr) E_th_arr "with lp dt 2 pt $sym_o ps 1.0 lc rgb '$color' title 'N=$N (Thermo)'"
        end
        
        Gnuplot.save("Spada2022-Fig5$suffix.png", term="pngcairo size 800,500 enhanced font 'Helvetica,12'")
        Gnuplot.save("Spada2022-Fig5$suffix.pdf", term="pdfcairo size 4in,3in enhanced font 'Helvetica,10'")
        println("Saved: Spada2022-Fig5$suffix.{png,pdf}")
    end
    
    # Cleanup
    rmprocs(workers())
    println("Workers removed. Process complete.")
end

main()
