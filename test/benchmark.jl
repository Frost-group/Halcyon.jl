using Halcyon
using BenchmarkTools
using Printf
using LinearAlgebra

function run_benchmarks()
    # ═══════════════════════════════════════════════════════════════════════════════
    # Benchmark Configurations
    # ═══════════════════════════════════════════════════════════════════════════════
    
    cases = [
        (name="Case A (Small Ideal)", N=2, M=16, U=NullPairPotential()),
        (name="Case B (Large Ideal)", N=64, M=32, U=NullPairPotential()),
        (name="Case C (Large Interacting)", N=64, M=64, U=HardSpherePotential(a=1.0))
    ]
    
    params = WormParams(C=1.0, j_max=16, j_max_open=16, j_max_swap=16, r_max=0.5)
    
    for case in cases
        println("\n" * "=" ^ 80)
        println("Benchmarking: $(case.name)")
        println("N=$(case.N), M=$(case.M), Interaction=$(typeof(case.U))")
        println("=" ^ 80)
        
        # Setup system and configuration
        # Fix density n=0.1 for interacting, L=1.0 for small ideal
        L = (case.U isa NullPairPotential) ? 1.0 : (case.N / 0.1)^(1/3)
        sys = System(case.M, case.N, m=1.0, D=3, β=1.0, V=HarmonicPotential(k=0.0), U=case.U, λ=0.5, L=L)
        cfg = WormConfiguration(sys)
        
        # Benchmarking individual moves
        # Note: some moves only work in specific sectors or under certain conditions.
        # We'll force sectors where necessary.
        
        println("\n--- Monte Carlo Moves ---")
        
        # translate! (works in both sectors)
        print("translate!: ")
        @btime translate!($cfg, $sys, $params)
        
        # redraw! (works in both sectors)
        print("redraw!:    ")
        @btime redraw!($cfg, $sys, $params)
        
        # For open/close/swap/head/tail, we need to be in G-sector or handle transitions
        # We'll benchmark them in their "natural" sectors if possible.
        
        # open! (Z-sector)
        cfg.sector = Z_SECTOR
        print("open!:      ")
        @btime open!($cfg, $sys, $params)
        
        # close! (G-sector)
        # Note: close! might transition to Z-sector, we force it back for benchmarking
        print("close!:     ")
        @btime begin
            $cfg.sector = G_SECTOR
            $cfg.i_head = 1
            $cfg.i_tail = $cfg.next[$cfg.i_head]
            $cfg.r_c_head .= $cfg.r[1, 1, :]
            close!($cfg, $sys, $params)
        end
        
        # swap! (G-sector)
        print("swap!:      ")
        @btime begin
            $cfg.sector = G_SECTOR
            $cfg.i_head = 1
            $cfg.i_tail = $cfg.next[$cfg.i_head]
            swap!($cfg, $sys, $params)
        end
        
        # move_head! (G-sector)
        print("move_head!: ")
        @btime begin
            $cfg.sector = G_SECTOR
            $cfg.i_head = 1
            move_head!($cfg, $sys, $params)
        end
        
        # move_tail! (G-sector)
        print("move_tail!: ")
        @btime begin
            $cfg.sector = G_SECTOR
            $cfg.i_head = 1
            $cfg.i_tail = $cfg.next[$cfg.i_head]
            move_tail!($cfg, $sys, $params)
        end
        
        println("\n--- Energy Estimators (Z-sector) ---")
        cfg.sector = Z_SECTOR
        
        print("thermodynamic: ")
        @btime energy_thermodynamic($cfg, $sys)
        
        print("virial:        ")
        @btime energy_virial($cfg, $sys)
        
        println("\n--- Aggregate Step ---")
        print("worm_step!:    ")
        @btime worm_step!($cfg, $sys, $params)
    end
end

run_benchmarks()

