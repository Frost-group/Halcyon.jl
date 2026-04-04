using Halcyon
using BenchmarkTools
using Printf

function run_benchmarks()
    # ═══════════════════════════════════════════════════════════════════════════════
    # Benchmark Configurations - Physically Relevant Setups
    # ═══════════════════════════════════════════════════════════════════════════════

    cases = [
        # Small ideal gas - minimal test
        (name="Case A: Small Ideal Gas", N=2, M=16, D=3, L=1.0,
            V=HarmonicPotential(k=0.0), U=NullPairPotential()),

        # Large ideal gas - scaling test  
        (name="Case B: Large Ideal Gas", N=64, M=32, D=3, L=10.0,
            V=HarmonicPotential(k=0.0), U=NullPairPotential()),

        # Hard-sphere liquid - Cao-Berne test
        (name="Case C: Hard-Sphere Liquid", N=64, M=64, D=3, L=20.0,
            V=HarmonicPotential(k=0.0), U=NullPairPotential()),

        # Dornheim 2019: Harmonic trap + Coulomb (2D quantum dot)
        (name="Case D: Harmonic Trap + Coulomb (Dornheim)", N=6, M=100, D=2, L=5.0,
            V=HarmonicPotential(k=0.5), U=CoulombPotential(g=-0.5)),
    ]

    params = WormParams(C=1.0, j_max=16, j_max_open=16, j_max_swap=16, r_max=0.5)

    for case in cases
        println("\n" * "="^80)
        println("Benchmarking: $(case.name)")
        println("N=$(case.N), M=$(case.M), D=$(case.D)")
        println("="^80)

        sys = System(case.M, case.N; m=1.0, D=case.D, β=1.0,
            V=case.V, U=case.U, λ=0.5, L=case.L)
        cfg = WormConfiguration(sys)

        println("\n--- Monte Carlo Moves ---")
        print("translate!: ")
        @btime translate!($cfg, $sys, $params)
        print("redraw!:    ")
        @btime redraw!($cfg, $sys, $params)
        cfg.sector = Z_SECTOR
        print("open!:      ")
        @btime open!($cfg, $sys, $params)

        println("\n--- Energy Estimator ---")
        cfg.sector = Z_SECTOR
        @btime energy_estimators($cfg, $sys)
    end

    # ═══════════════════════════════════════════════════════════════════════════════
    # Scaling Benchmarks: Hard-Sphere (Cao-Berne)
    # ═══════════════════════════════════════════════════════════════════════════════
    println("\n" * "="^80)
    println("SCALING BENCHMARKS: energy_estimators (Hard-Sphere)")
    println("="^80)

    # Scaling with N (fixed M=32)
    println("\n--- N-scaling (M=32, D=3, HS a=1.0) ---")
    println(rpad("N", 6) * " | " * rpad("Time", 12))
    println("-"^25)

    for N_val in [1, 10, 32, 64, 128]
        L = max(5.0, (N_val)^(1 / 3) * 3.0)  # Scale L with N
        sys = System(32, N_val; m=1.0, D=3, β=1.0,
            V=HarmonicPotential(k=0.0), U=NullPairPotential(), λ=0.5, L=L)
        cfg = WormConfiguration(sys)
        energy_estimators(cfg, sys)  # Warmup
        t = @belapsed energy_estimators($cfg, $sys)
        @printf("%-6d | %s\n", N_val, pretty_time(t * 1e9))
    end

    # Scaling with M (fixed N=10)
    println("\n--- M-scaling (N=10, D=3, HS a=1.0) ---")
    println(rpad("M", 6) * " | " * rpad("Time", 12))
    println("-"^25)

    for M_val in [16, 32, 64, 128, 256]
        sys = System(M_val, 10; m=1.0, D=3, β=1.0,
            V=HarmonicPotential(k=0.0), U=NullPairPotential(), λ=0.5, L=10.0)
        cfg = WormConfiguration(sys)
        energy_estimators(cfg, sys)  # Warmup
        t = @belapsed energy_estimators($cfg, $sys)
        @printf("%-6d | %s\n", M_val, pretty_time(t * 1e9))
    end

    # ═══════════════════════════════════════════════════════════════════════════════
    # Scaling Benchmarks: Harmonic + Coulomb (Dornheim setup)
    # ═══════════════════════════════════════════════════════════════════════════════
    println("\n" * "="^80)
    println("SCALING BENCHMARKS: energy_estimators (Harmonic+Coulomb)")
    println("="^80)

    # Scaling with N (fixed M=50)
    println("\n--- N-scaling (M=50, D=2, Harmonic+Coulomb) ---")
    println(rpad("N", 6) * " | " * rpad("Time", 12))
    println("-"^25)

    for N_val in [2, 4, 6, 10, 20]
        sys = System(50, N_val; m=1.0, D=2, β=1.0,
            V=HarmonicPotential(k=0.5), U=CoulombPotential(g=-0.5), λ=0.5, L=5.0)
        cfg = WormConfiguration(sys)
        energy_estimators(cfg, sys)  # Warmup
        t = @belapsed energy_estimators($cfg, $sys)
        @printf("%-6d | %s\n", N_val, pretty_time(t * 1e9))
    end
end

function pretty_time(ns)
    ns < 1e3 && return @sprintf("%.2f ns", ns)
    ns < 1e6 && return @sprintf("%.2f μs", ns / 1e3)
    ns < 1e9 && return @sprintf("%.2f ms", ns / 1e6)
    return @sprintf("%.2f s", ns / 1e9)
end

run_benchmarks()
