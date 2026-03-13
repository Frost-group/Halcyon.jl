# Spada et al. 2022 — Fig. 3 (\label{fig:3}): N=2 + swap vs exact E.
# Sec. III: "swap update makes its first appearance".

using Test
using Halcyon

@testset "Spada2022 Fig. 3 worm (N=2 swap)" begin
    N, D, L, λ = 2, 3, 1.0, 0.5
    λT_L = 0.795
    M = 16
    β = β_from_λT_ratio(λT_L, L, λ)
    E_exact = E_N_exact(N, β, L, λ)

    sys = System(M, N; m=1.0, D=D, β=β, V=HarmonicPotential(0.0), U=NullPairPotential(), λ=λ, L=L)
    cfg = WormConfiguration(sys)
    params = WormParams(C=1.0, j_max=M, j_max_open=M, j_max_swap=M, r_max=0.5)

    n_eq = 60_000
    n_meas = 500_000
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
    @test n >= 50 # should be more like 5000
    E_mean = sum(samples) / n
    E_se = sqrt(sum((x - E_mean)^2 for x in samples) / (n - 1)) / sqrt(n)

    println("SpadaFig3_exact(β=$β, L=$L, λ=$λ) = $E_exact")
    println("SpadaFig3 MC (n_eq=$n_eq, n_meas=$n_meas): E_mean = $E_mean, E_se = $E_se, n = $n")
    
    @test abs(E_mean - E_exact) < 4 * E_se
end
