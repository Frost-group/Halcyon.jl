# Interacting UEG-style run at r_s=2, θ=0.5: repulsive 1/r with minimum image (not Yakub–Ronchi).
# NOTE THAT WE ONLY HAVE BARE COULOMB

using Halcyon
using Printf

include(joinpath(@__DIR__, "Dornheim2019Permutation_setup.jl"))

function run_ueg(; N::Int=19, θ::Float64=0.5, r_s::Float64=2.0, M::Int=100,
                 equil::Int=80_000, steps::Int=300_000, measure_every::Int=25)
    λ = 0.5
    (; L, β, E_F) = ueg_theta_parameters(; N, θ, r_s, λ)
    pair = CoulombPotential(g=-1.0)  # +1/r repulsion in atomic units
    sys = make_periodic_fermion_system(; M, N, β, L, λ, pair=pair)
    params = default_worm_params(sys)
    cfg = WormConfiguration(sys)

    @printf("UEG (MI Coulomb)  N=%d  θ=%.3g  r_s=%.3g  M=%d  β=%.5f  L=%.5f  E_F=%.5f\n",
        N, θ, r_s, M, β, L, E_F)
    @printf("WARNING: paper uses Yakub–Ronchi local potential; currently we only have BARE COULOMB.\n\n")

    acc_P = zeros(N)
    n_z = 0
    for _ in 1:equil
        worm_step!(cfg, sys, params)
    end
    for t in 1:steps
        worm_step!(cfg, sys, params)
        if t % measure_every == 0 && cfg.sector == Z_SECTOR
            acc_P .+= permutation_pl_weights(cfg)
            n_z += 1
        end
    end
    n_z == 0 && error("No Z-sector samples")

    P_mc = acc_P ./ n_z
    println("l   P(l) MC        P(l)*l")
    for l in 1:N
        @printf("%3d  %.5e  %.5f\n", l, P_mc[l], P_mc[l] * l)
    end
    println("\nZ-sector samples: ", n_z)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_ueg()
end
