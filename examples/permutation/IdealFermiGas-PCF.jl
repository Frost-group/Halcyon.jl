# Estimates a single column P(l_fixed, k) and uncorrelated P_u(l_fixed, k) for ideal fermions.
# # Ideal spin-polarised Fermi gas, θ=0.5, r_s=2
# Matches the "Column_PCF" style plots in Dornheim2019Permutation (e.g. k = 1, 3, 10).
# Fig 4, all notably monatonic 
# very simple test currently! Then went off to play with 2D quantum dots

using Halcyon
using Printf

include(joinpath(@__DIR__, "Dornheim2019Permutation_setup.jl"))

function run_pcf_column(; N::Int=33, θ::Float64=0.5, r_s::Float64=2.0, M::Int=100,
                        l_fixed::Int=1, equil::Int=60_000, steps::Int=250_000,
                        measure_every::Int=25)
    λ = 0.5
    (; L, β) = ueg_theta_parameters(; N, θ, r_s, λ)
    sys = make_periodic_fermion_system(; M, N, β, L, λ)
    params = default_worm_params(sys)
    cfg = WormConfiguration(sys)

    for _ in 1:equil
        worm_step!(cfg, sys, params)
    end

    acc_pcf = zeros(N)
    acc_pl = zeros(N)
    n_z = 0
    for t in 1:steps
        worm_step!(cfg, sys, params)
        if t % measure_every == 0 && cfg.sector == Z_SECTOR
            W = permutation_pcf_weights(cfg)
            acc_pcf .+= W[l_fixed, :]
            acc_pl .+= permutation_pl_weights(cfg)
            n_z += 1
        end
    end
    n_z == 0 && error("No Z-sector samples")

    P_mc = acc_pl ./ n_z
    col = acc_pcf ./ n_z
    Pu_col = [P_mc[l_fixed] * P_mc[k] for k in 1:N]

    cache = Dict{Tuple{Int,Float64},Float64}()
    println(@sprintf("# Ideal fermions  N=%d  θ=%g  column P(%d,k)  (%d Z-samples)\n", N, θ, l_fixed, n_z))
    println("#k    P(l,k) MC    P_u(l,k)     P(l) exact")
    for k in 1:N
        p_ex = ideal_fermion_permutation_P_l(k, N, β, L, λ; cache=cache)
        @printf("%3d  %.5e  %.5e  %.5e\n", k, col[k], Pu_col[k], p_ex)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_pcf_column(; l_fixed=1)
end
