#!/usr/bin/env julia
# Ideal spin-polarised Fermi gas, θ=0.5, r_s=2 (sets L only; ideal P(l) independent of r_s).
# Compares MC averages of permutation_pl_weights to ideal_fermion_permutation_P_l.

using Halcyon
using Statistics
using Printf

include(joinpath(@__DIR__, "Dornheim2019Permutation_setup.jl"))

function run_ideal(; N::Int=19, θ::Float64=0.5, r_s::Float64=2.0, M::Int=100,
                   equil::Int=100_000, steps::Int=1_000_000, measure_every::Int=20,
                   C::Float64=1.0)
    λ = 0.5
    (; L, β, E_F) = ueg_theta_parameters(; N, θ, r_s, λ)
    sys = make_periodic_fermion_system(; M, N, β, L, λ)
    params = default_worm_params(sys; C=C)
    cfg = WormConfiguration(sys)

    @printf("#Ideal fermions  N=%d  θ=%.3g  r_s=%.3g  M=%d  β=%.5f  L=%.5f  E_F=%.5f\n",
        N, θ, r_s, M, β, L, E_F)

    for _ in 1:equil
        worm_step!(cfg, sys, params)
    end

    acc_P = zeros(N)
    n_z = 0
    for t in 1:steps
        worm_step!(cfg, sys, params)
        if t % measure_every == 0 && cfg.sector == Z_SECTOR
            acc_P .+= permutation_pl_weights(cfg)
            n_z += 1
        end
    end
    n_z == 0 && error("No Z-sector samples; increase steps or measure_every")

    P_mc = acc_P ./ n_z
    cache = Dict{Tuple{Int,Float64},Float64}()
    println("\n# l   P(l) MC        P(l) exact     P(l)*l MC    P(l)*l exact")
    for l in 1:N
        p_ex = ideal_fermion_permutation_P_l(l, N, β, L, λ; cache=cache)
        @printf("%3d  %.5e  %.5e  %.5f      %.5f\n", l, P_mc[l], p_ex, P_mc[l] * l, p_ex * l)
    end
    println("\n# MC samples (Z-sector): ", n_z)
    return (; P_mc, sys, β, L)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_ideal()
end
