# examples/Ceperley1995-TemperatureSweepFigures.jl
# Combined reproduction of Ceperley 1995 results:
# - Energy (Fig 13) - Separate Virial and Thermodynamic estimators
# - Superfluid Fraction (Fig 14/15)
# - Cycle Length Distribution (Fig 12)

using Distributed

# ═══════════════════════════════════════════════════════════════════════════════
# Parallel Setup
# ═══════════════════════════════════════════════════════════════════════════════

if nworkers() < 6
    addprocs(6) # I dunno, use the efficiency cores so that early-finishing work gets backfilled?
end

@everywhere using Halcyon, Statistics, Printf

@everywhere begin
    # ═══════════════════════════════════════════════════════════════════════════════
    # Parameters
    # ═══════════════════════════════════════════════════════════════════════════════

    const N = 64
    const D = 3
    const rho = 0.02182 # Å⁻³
    const L = (N / rho)^(1 / 3)
    const λ = 6.0596 # Å² K

    # MC parameters. About half an hour for 12 sweeps with ~110k total moves
    const EQUILIBRATION_STEPS = 10_000
    const MEASUREMENT_STEPS = 100_000
    const MEASURE_INTERVAL = 100

    function run_T(T::Float64)
        β = 1.0 / T
        M = max(80, round(Int, β / 0.0125))

        @printf("  StartingMC... T = %.2f K (M=%d)\n", T, M)

        sys = System(M, N; m=4.0026, D=D, β=β, λ=λ, L=L, U=AzizPotential())
        cfg = WormConfiguration(sys)
        params = WormParams(C=0.5, j_max=20, j_max_open=20, j_max_swap=15, r_max=L / 2)

        # Equilibrating
        for step in 1:EQUILIBRATION_STEPS
            worm_step!(cfg, sys, params)
        end

        # Accumulators
        E_thermo_samples = Float64[]
        E_virial_samples = Float64[]
        rhos_samples = Float64[]

        cycle_counts = zeros(Int, N)
        cycle_samples = 0

        collect_sample::Bool = false

        for step in 1:MEASUREMENT_STEPS
            worm_step!(cfg, sys, params)

            if step % MEASURE_INTERVAL == 0
                collect_sample = true # i.e. hit interval... now collect sample...
            end
            if step % MEASURE_INTERVAL == 0 # whichever sector...
                #collect_sample && cfg.sector == Z_SECTOR # ...as soon as we hit Z-sector, worm reconnects
                # 1. Thermodynamics, only if in Z...
                if cfg.sector == Z_SECTOR
                    push!(E_thermo_samples, energy_thermodynamic(cfg, sys))
                    push!(E_virial_samples, energy_virial(cfg, sys))
                end

                push!(rhos_samples, superfluid_fraction(cfg, sys))
                # not sure about this one?

                # 2. Permutation Cycles... seem ok in both sectors?
                visited = zeros(Bool, N)
                for i in 1:N
                    if !visited[i]
                        cycle = get_cycle(cfg, i)
                        m = length(cycle)
                        cycle_counts[m] += 1
                        for p in cycle
                            visited[p] = true
                        end
                    end
                end
                cycle_samples += 1
                collect_samepl = false # OK, we got our sample
            end

            if step % (MEASUREMENT_STEPS / 10) == 0
                @printf("T = %.2f  %02d pc  done\n", T, Int(step * 100 / MEASUREMENT_STEPS))
            end
        end

        # RETURN EARLY if no results; generally only happens short testing runs & fail to reconnect to Z
        if isempty(E_thermo_samples)
            return (T=T, samples=0, E_thermo=NaN, E_thermo_err=NaN, E_virial=NaN, E_virial_err=NaN, rhos=NaN, rhos_err=NaN, probs=zeros(40))
        end

        probs = [(m * (cycle_counts[m] / cycle_samples)) / N for m in 1:40]

        @printf("  Completed work package. T = %.2f K (M=%d)\n", T, M)

        return (
            T=T,
            samples=length(E_thermo_samples),
            E_thermo=mean(E_thermo_samples) / N,
            E_thermo_err=std(E_thermo_samples) / sqrt(length(E_thermo_samples)) / N,
            E_virial=mean(E_virial_samples) / N,
            E_virial_err=std(E_virial_samples) / sqrt(length(E_virial_samples)) / N,
            rhos=mean(rhos_samples),
            rhos_err=std(rhos_samples) / sqrt(length(rhos_samples)),
            probs=probs
        )
    end
end

using Gnuplot, DelimitedFiles

function main()
    println("="^70)
    println("Ceperley 1995 Temperature Sweep")
    println("="^70)

    # Temperature sweep... 6 high-speed CPUs on my macbook; so make it a factor
    temperatures = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
        2.17, 2.4, 2.8, 3.2, 3.6, 4.0]

    println("Starting parallel sweep of $(length(temperatures)) temperatures...")
    results = pmap(run_T, temperatures)
    sort!(results, by=r -> r.T)

    # ═══════════════════════════════════════════════════════════════════════════════
    # Data Export
    # ═══════════════════════════════════════════════════════════════════════════════

    open("Ceperley1995-TSweep.tsv", "w") do io
        write(io, "# Temperature\tsamples\tE_thermo\tE_thermo_err\tE_virial\tE_virial_err\trhos\trhos_err\tP1\tP2\tP4\tP8\n")
        for r in results
            @printf(io, "%.4f\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
                r.T, r.samples, r.E_thermo, r.E_thermo_err, r.E_virial, r.E_virial_err, r.rhos, r.rhos_err,
                r.probs[1], r.probs[2], r.probs[4], r.probs[8])
        end
    end
    println("Data saved to Ceperley1995-TSweep.tsv")

    # ═══════════════════════════════════════════════════════════════════════════════
    # Plotting
    # ═══════════════════════════════════════════════════════════════════════════════

    data_T = [r.T for r in results]

    # 1. Energy Comparison
    @gp "reset"
    @gp :- "set title 'Energy Estimator Comparison (Ceperley Fig 13)'" "set grid" "set size square"
    @gp :- "set xlabel 'T (K)'" "set ylabel 'E (K/atom)'"
    @gp :- data_T [r.E_thermo for r in results] [r.E_thermo_err for r in results] "w yerrorlines lw 2 pt 5 lc rgb 'blue' title 'Thermodynamic'"
    @gp :- data_T [r.E_virial for r in results] [r.E_virial_err for r in results] "w yerrorlines lw 2 pt 9 lc rgb 'red' title 'Virial'"
    Gnuplot.save("Ceperley1995-TSweep_Fig22_energies.png", term="pngcairo size 800,600 font 'Helvetica,12'")

    # 2. Superfluid Fraction
    @gp "reset"
    @gp :- "set title 'Superfluid Fraction'" "set grid" "set size square"
    @gp :- "set xlabel 'T (K)'" "set ylabel 'rho_s/rho'" "set yrange [0:1.1]"
    @gp :- data_T [r.rhos for r in results] [r.rhos_err for r in results] "w yerrorlines lw 2 pt 7 lc rgb 'purple' title 'PIMC'"
    Gnuplot.save("Ceperley1995-TSweep_Fig14_superfluid.png", term="pngcairo size 800,600 font 'Helvetica,12'")

    # 3. Cycle Length Distribution
    @gp "reset"
    @gp :- "set title 'Cycle Length Probability P_m(T) (Ceperley Fig 12)'"
    @gp :- "set xlabel 'T (K)'" "set ylabel 'P_m'" "set logscale y" "set yrange [1e-4:1.1]"
    @gp :- "set grid" "set size square"

    for m in 1:40
        pm_data = [r.probs[m] for r in results]
        @gp :- data_T pm_data "w l lw 1 lc rgb 'gray' notitle"
    end
    for m in [1, 2, 4, 8, 20]
        pm_data = [r.probs[m] for r in results]
        @gp :- data_T pm_data "w lp lw 2 pt 6 title 'm=$m'"
    end
    Gnuplot.save("Ceperley1995-TSweep_Fig12_cycles.png", term="pngcairo size 800,600 font 'Helvetica,12'")

    # Cleanup
    rmprocs(workers())
    println("Complete.")
end

main()


