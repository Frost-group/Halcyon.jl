# Spada et al. 2022 — Fig. 1: N=1 internal energy vs M at fixed λ_T/L.
# Sec. II moves only (no swap for N=1). Exact horizontal lines = E1_exact.

using Test
using Halcyon

@testset "Spada2022 Fig. 1 worm (N=1)" begin
    N, D, L, λ = 1, 3, 1.0, 0.5
    λT_L = 0.795
    M = 16
    β = β_from_λT_ratio(λT_L, L, λ)
    E_exact = E1_exact(β, L, λ)

    sys = System(M, N; m=1.0, D=D, β=β, V=HarmonicPotential(0.0), U=NullPairPotential(), λ=λ, L=L)
    cfg = WormConfiguration(sys)
    params = WormParams(C=1.0, j_max=M, j_max_open=max(1, M - 1), j_max_swap=M, r_max=min(0.5, L / 2))

    n_eq =  25_000
    n_meas = 350_000
    interval = 50

    for _ in 1:n_eq
        worm_step!(cfg, sys, params)
    end
    samples = Float64[]
    for step in 1:n_meas
        worm_step!(cfg, sys, params)
        if step % interval == 0 && cfg.sector == Z_SECTOR
            _, E_virial = energy_estimators(cfg, sys)
            push!(samples, E_virial)
        end
    end
    n = length(samples)
    
    E_mean = sum(samples) / n
    E_se = sqrt(sum((x - E_mean)^2 for x in samples) / (n - 1)) / sqrt(n)
    
    println("E1_exact(β=$β, L=$L, λ=$λ) = $E_exact")
    println("MC (n_eq=$n_eq, n_meas=$n_meas): E_mean = $E_mean, E_se = $E_se, n = $n")

    # Statistical gate: within 4σ of exact, or moderate rtol if stderr large
    @test abs(E_mean - E_exact) < 4 * E_se
end


