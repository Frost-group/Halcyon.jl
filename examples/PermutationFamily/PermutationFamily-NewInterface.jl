# permutation-family model fitting
# Adaptd PermutationFamily-IdealFermiGas.jl but new structuted code now in main source

using Halcyon
using Printf
using ProgressMeter

"""Wigner-Seitz density n = 3/(4π r_s³) in 3D (atomic units)."""
ueg_density_3d(r_s::Float64) = 3.0 / (4π * r_s^3)

"""Cubic box side L for N electrons at given r_s."""
ueg_box_length(N::Int, r_s::Float64) = (N / ueg_density_3d(r_s))^(1 / 3)

"""Fermi wavevector for fully spin-polarised 3D gas."""
fermi_wavenumber_polarised(n::Float64) = (6π^2 * n)^(1 / 3)

"""Returns (L, β, E_F, n, k_F) for given degeneracy temperature θ = T/E_F."""
function ueg_theta_parameters(; N::Int, θ::Float64, r_s::Float64, λ::Float64=0.5)
    L = ueg_box_length(N, r_s)
    n = N / L^3
    kF = fermi_wavenumber_polarised(n)
    EF = λ * kF^2
    β = 1.0 / (θ * EF)
    return (; L, β, E_F=EF, n, k_F=kF)
end

function make_periodic_fermion_system(; M::Int, N::Int, β::Float64, L::Float64,
                                    λ::Float64=0.5, pair::PairPotential=NullPairPotential())
    System(M, N; D=3, β=β, λ=λ, L=L, V=HarmonicPotential(k=0.0), U=pair, statistics=Fermions)
end

default_worm_params(sys::System; C::Float64=1.0) =
    WormParams(C=C, j_max=sys.M ÷ 2, r_max=sys.L / 2)

function MC_and_fit_model(; N::Int=33, θ::Float64=0.5, r_s::Float64=2.0, M::Int=100,
                             equil::Int=100_000, steps::Int=100_000_000, measure_every::Int=5)
    λħ = 0.5
    (; L, β) = ueg_theta_parameters(; N, θ, r_s, λ=λħ)
    sys = make_periodic_fermion_system(; M, N, β, L, λ=λħ)
    params = default_worm_params(sys)
    cfg = WormConfiguration(sys)

    for _ in 1:equil
        worm_step!(cfg, sys, params)
    end

    histo = DensePermutationFamilyStats(N)
    println("Allocated DensePermutationFamilyStats  N=$N  p(N)=$(histo.n_families)  ",
            "size=$(Base.summarysize(histo) ÷ 1024) KiB")
    n_z = 0

    @showprogress desc="MC:$(steps/1E6)M" for t in 1:steps
        worm_step!(cfg, sys, params)
        if t % measure_every == 0 && cfg.sector == Z_SECTOR
            λ_vec = permutation_family_lambda(cfg)
            observe_permutation_family!(histo, λ_vec, 0.0)
            n_z += 1
        end
    end
    n_z == 0 && error("No Z-sector samples")
    println(@sprintf("\n# Ideal fermions  N=%d  θ=%g  p(N)=%d  Z-samples=%d\n", N, θ, histo.n_families, n_z))

    # ---------------------------------------------------------------
    # Fit the model hierarchy with the new interface 
    # ---------------------------------------------------------------
    mod_mult   = fit(MultiplicityModel, histo)
    mod_dubois = fit(DuBoisModel, histo)
    mod_map    = fit(MAPHybridModel, histo; prior=mod_dubois, τ₀=1.0)
    mod_full   = fit(MaxEntModel, histo; l2_reg=1e-4)

    models = [
        ("Infinite-T (degeneracy)", mod_mult),
        ("DuBois 1-param",         mod_dubois),
        ("MAP hybrid",             mod_map),
        ("Full MaxEnt",            mod_full),
    ]

    println("Model fits:")
    for (label, m) in models
        @printf("  KL %-30s = %.5f nats\n", label, kl_divergence(m, histo))
    end
    println()
    println("  DuBois: ", mod_dubois)
    println("  MAP:    ", mod_map)
    println("  MaxEnt: ", mod_full)

    # ---------------------------------------------------------------
    # Write residuals for all sectors
    # ---------------------------------------------------------------
    n_tot = sum(histo.count)
    q_mult   = probabilities(mod_mult,   histo)
    q_dubois = probabilities(mod_dubois, histo)
    q_map    = probabilities(mod_map,    histo)
    q_full   = probabilities(mod_full,   histo)

    open("PermutationFamily_histogram_fit.dat", "w") do io
        println(io, "# count  family_index  P_hat  P_mult  P_dubois  P_map  P_full  λ")
        for k in 1:histo.n_families
            c = histo.count[k]
            p_hat = c > 0 ? c / n_tot : 0.0
            λk = permutation_family_lambda_from_rank(k, N, histo.P)
            r = findfirst(iszero, λk)
            head = isnothing(r) ? λk : λk[1:(r - 1)]
            @printf(io, "%8d  %8d  %.5e  %.5e  %.5e  %.5e  %.5e  %s\n",
                    c, k, p_hat, q_mult[k], q_dubois[k], q_map[k], q_full[k], string(collect(head)))
        end
    end

    open("PermutationFamily_theta_models.dat", "w") do io
        println(io, "# cycle_length  θ_mult  θ_dubois  θ_map  θ_full")
        for ℓ in 1:N
            @printf(io, "%4d  %10.5f  %10.5f  %10.5f  %10.5f\n",
                    ℓ, mod_mult.θ[ℓ], mod_dubois.θ[ℓ], mod_map.θ[ℓ], mod_full.θ[ℓ])
        end
    end

    return histo, models
end

if abspath(PROGRAM_FILE) == @__FILE__
#    MC_and_fit_model(; N=33, θ=0.5, r_s=2.0)
#    MC_and_fit_model(; N=33, θ=0.125, r_s=1.0) # DuBois Table 1, Weak Coupling / High Density
    MC_and_fit_model(; N=33, θ=0.125, r_s=10.0)# DuBois Table 1, Strong Coupling / Low Density
end
