using Halcyon, Printf, Statistics

function default_worm_params(sys::System)
    return WormParams(C=1.0, j_max=sys.M ÷ 2, r_max=sys.L / 4.0)
end

function ideal_fermi_gas_scaling_table()
    println("\n=========================================================================")
    println("Ideal Spin-Polarized Fermi Gas Scaling Table")
    println("=========================================================================")
    @printf("%4s %8s %12s %12s %12s %12s %10s %10s\n", "N", "Theta", "λ_T/L", "Exact E", "Virial E", "Diff", "Err %", "Sign")
    println(repeat("-", 86))

    D = 3
    λ = 0.5
    L = 5.0
    M = 50
    for N in [2, 4, 7, 19, 33]
        @printf("\n")
        for θ in [0.01, 0.05, 0.125, 0.5, 1.0, 2.0, 5.0, 10.0]
            n_density = N / L^3
            k_F = (6π^2 * n_density)^(1 / 3)
            E_F = λ * k_F^2
            β = 1.0 / (θ * E_F)

            ratio_λT_L = sqrt(4π * λ * β) / L
            E_exact_F = E_N_exact_Fermi(N, β, L, λ)

            sys = System(M, N; D=D, β=β, λ=λ, L=L,
                V=HarmonicPotential(k=0.0), U=NullPairPotential(),
                statistics=Fermions)
            params = default_worm_params(sys)
            cfg = WormConfiguration(sys)

            for _ in 1:100_000
                worm_step!(cfg, sys, params)
            end

            ES_samples = Float64[]
            S_samples = Float64[]
            for t in 1:200_000
                worm_step!(cfg, sys, params)
                if t % 5 == 0 && cfg.sector == Z_SECTOR
                    _, Evir = energy_estimators(cfg, sys)
                    S = Float64(permutation_sign(cfg, sys))
                    push!(ES_samples, Evir * S)
                    push!(S_samples, S)
                end
            end

            mean_ES = mean(ES_samples)
            mean_S = mean(S_samples)
            E_vir_F = mean_ES / mean_S

            diff = E_vir_F - E_exact_F
            err_pct = 100 * diff / E_exact_F

            @printf("%4d %8.3f %12.3f %12.6f %12.6f %+12.6f %+10.2f %10.4f\n",
                N, θ, ratio_λT_L, E_exact_F / N, E_vir_F / N, diff, err_pct, mean_S)
        end
    end
    println("=========================================================================\n")
end

if abspath(PROGRAM_FILE) == @__FILE__
    ideal_fermi_gas_scaling_table()
end
