# PermutationFamily-LSTM.jl
# Trains the LSTM autoregressive model on ideal-fermion MC data and compares
# KL divergence against the exponential family models.
# Three LSTM variants are fitted:
#   1. No prior (flat logit bias)
#   2. Multiplicity prior (combinatorial M(λ) factors for free)
#   3. DuBois prior (M(λ) + exchange penalty κ)

using Halcyon
using Printf
using ProgressMeter
using Random

"""Wigner-Seitz density n = 3/(4π r_s³) in 3D (atomic units)."""
ueg_density_3d(r_s::Float64) = 3.0 / (4π * r_s^3)

"""Cubic box side L for N electrons at given r_s."""
ueg_box_length(N::Int, r_s::Float64) = (N / ueg_density_3d(r_s))^(1 / 3)

"""Fermi wavevector for fully spin-polarised 3D gas."""
fermi_wavenumber_polarised(n::Float64) = (6π^2 * n)^(1 / 3)

fermi_wavenumber_unpolarised(n::Float64) = (3π^2 * n)^(1 / 3)

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


"""
    calculate_reweighted_energy(model::AbstractPermutationModel, stats::DensePermutationFamilyStats)

Calculate the expectation value of the energy with our model prob; not hte MC prob.
"""
function calculate_reweighted_energy(model::AbstractPermutationModel, stats::DensePermutationFamilyStats)
    q = probabilities(model, stats)
    N = stats.N

    Etot = 0.0
    Z_sign = 0.0
    for k in 1:stats.n_families
        if stats.count[k] > 0
            # Conditional mean energy in sector k
            E_k = stats.estimator[k] / stats.count[k]

            C_k = C_from_rank(k, N, stats.P)
            n_cycles = sum(C_k)
            sigma = iseven(N - n_cycles) ? 1.0 : -1.0

            Etot += sigma * q[k] * E_k
            Z_sign += sigma * q[k]
        end
    end
    return Etot / Z_sign
end

function calculate_mc_energy(stats::DensePermutationFamilyStats)
    N = stats.N
    n_z=sum(stats.count)

    Etot = 0.0
    Z_sign = 0.0
    for k in 1:stats.n_families
        if stats.count[k] > 0
            # Conditional mean energy in sector k
            E_k = stats.estimator[k] / stats.count[k]

            C_k = C_from_rank(k, N, stats.P)
            n_cycles = sum(C_k)
            sigma = iseven(N - n_cycles) ? 1.0 : -1.0

            p_k = stats.count[k] / n_z
            Etot += sigma * p_k * E_k
            Z_sign += sigma * p_k
        end
    end
    return Etot / Z_sign
end

function calculate_reweighted_sign(model::AbstractPermutationModel, stats::DensePermutationFamilyStats)
    q = probabilities(model, stats)
    N = stats.N

    val = 0.0
    for k in 1:stats.n_families
        C_k = C_from_rank(k, N, stats.P)
        n_cycles = sum(C_k)
        sigma = iseven(N - n_cycles) ? 1.0 : -1.0
        val += q[k] * sigma
    end
    return val
end
function merge_stats(a::DensePermutationFamilyStats, b::DensePermutationFamilyStats)
    a.N == b.N && a.n_families == b.n_families || throw(ArgumentError("merge_stats"))
    DensePermutationFamilyStats(a.N, a.P, a.n_families,
        a.count .+ b.count, a.estimator .+ b.estimator)
end

"""Equilibrate once, run `steps` production steps, return partial stats."""
function permutation_family_mc_chain(sys, params; equil, steps, measure_every, seed::UInt64, prog=nothing)
    Random.seed!(seed)
    cfg = WormConfiguration(sys)
    for t in 1:equil
        worm_step!(cfg, sys, params)
    end
    acc = DensePermutationFamilyStats(sys.N)
    for t in 1:steps
        worm_step!(cfg, sys, params)
        prog !== nothing && t%50==0 && next!(prog)
        if t % measure_every == 0 && cfg.sector == Z_SECTOR
            C_vec = permutation_family_C(cfg)
            E_therm, E_virial = energy_estimators(cfg, sys)
            observe_permutation_family!(acc, C_vec, E_virial)
        end
    end
    acc
end
"""Split `steps` across `n_chains` (remainder to early chains); merge with `merge_stats`."""
function permutation_family_mc(sys, params; equil, steps, measure_every, show_progress::Bool=true, n_chains::Int=Threads.nthreads())
    q, r = divrem(steps, n_chains)
    base = rand(RandomDevice(), UInt64)
    parts = Vector{DensePermutationFamilyStats}(undef, n_chains)
    Threads.@threads for i in 1:n_chains
        nstep = q + (i ≤ r)
        prog = (show_progress && i == 1) ? Progress(nstep÷50; desc="MC:$(steps/1e6)M") : nothing
        parts[i] = permutation_family_mc_chain(sys, params; equil,
            steps=nstep, measure_every, seed=base + UInt64(i), prog=prog)
    end
    reduce(merge_stats, parts)
end


function MC_and_fit_model(; N::Int=33, θ::Float64=0.5, r_s::Float64=2.0, M::Int=100,
    equil::Int=100_000, steps::Int=1_000_000, measure_every::Int=5,
    lstm_epochs::Int=200, lstm_hidden::Int=64, lstm_embed::Int=16,
    lstm_lr::Float64=1e-3)
    λħ = 0.5
    (; L, β) = ueg_theta_parameters(; N, θ, r_s, λ=λħ)
    sys = make_periodic_fermion_system(; M, N, β, L, λ=λħ, pair=YakubRonchiPotential(L=L, g=1))
    params = default_worm_params(sys)

    # Add Jellium Background; Yakub-Ronchi
    E_bg = yakub_ronchi_background_constant(L, N)
    @printf("Calculated background N= %d L= %d E= %g\n",N,L,E_bg)

    println("Running threaded MC...")
    MC_data = permutation_family_mc(sys, params; equil, steps, measure_every)
    
    println("DensePermutationFamilyStats  N=$N  p(N)=$(MC_data.n_families)  ",
        "size=$(Base.summarysize(MC_data) ÷ 1024) KiB")    
    n_z=sum(MC_data.count)
    @printf("\n#MC Complete! N=%d  r_s=%g θ=%g  n_families=%d  Z-samples=%d\n", N, r_s, θ, MC_data.n_families, n_z)
    MC_E = calculate_mc_energy(MC_data)

    @printf("MC_E= %.8f MC_E/N= %.8f Ha = %.8f Ry (includes jellium bg)\n", MC_E + N*E_bg, (MC_E + N*E_bg) / N, 2*(MC_E + N*E_bg) / N)

    # ---------------------------------------------------------------
    # Exponential family fits
    # ---------------------------------------------------------------
    println("Fitting exponential family models...")
    model_mult = fit(MultiplicityModel, MC_data)
    model_dubois = fit(DuBoisModel, MC_data)
    model_map = fit(MAPHybridModel, MC_data; prior=model_dubois, τ₀=1.0)
    model_full = fit(MaxEntModel, MC_data; l2_reg=1e-4)

    # ---------------------------------------------------------------
    # LSTM fits: three variants with increasing prior knowledge
    # ---------------------------------------------------------------
    println("Unique data points (per epoch): $(length(MC_data.count))")

    lstm_kw = (; n_embed=lstm_embed, n_hidden=lstm_hidden, epochs=lstm_epochs, lr=lstm_lr)

    # 1. No prior — flat bias, LSTM must learn everything from scratch
    no_prior = MultiplicityModel(zeros(N))
    no_prior.θ .= 0.0  # θ=0 and the -log(v)-log(c+1) terms still active
    println("Training LSTM (no prior)...")
    model_lstm_flat = fit(LSTMPermutationModel, MC_data; prior=no_prior, lstm_kw...)

    # 2. Multiplicity prior — gets M(λ) factors for free (θ=0)
    println("Training LSTM (multiplicity prior)...")
    model_lstm_mult = fit(LSTMPermutationModel, MC_data; prior=model_mult, lstm_kw...)

    # 3. DuBois prior — gets M(λ) + κ-penalty for free
    println("Training LSTM (DuBois prior)...")
    model_lstm_DuBois = fit(LSTMPermutationModel, MC_data; prior=model_dubois, lstm_kw...)

    # ---------------------------------------------------------------
    # Compare KL divergences
    # ---------------------------------------------------------------
    models = [
        ("Infinite-T (degeneracy)", model_mult),
        ("DuBois 1-param", model_dubois),
        ("MAP hybrid", model_map),
        ("Full MaxEnt", model_full),
        ("LSTM (no prior)", model_lstm_flat),
        ("LSTM (mult prior)", model_lstm_mult),
        ("LSTM (DuBois prior)", model_lstm_DuBois),
    ]

    @printf("%-30s MC_E= %.8f MC_E/N= %.8f Ha = %.8f Ry (includes jellium bg)\n", 
            "MonteCarloAsCeperleyIntended", 
            MC_E + N*E_bg, 
            (MC_E + N*E_bg) / N, 2*(MC_E + N*E_bg) / N)

    println("Model fits:")
    for (label, m) in models
        KL = kl_divergence(m, MC_data)
        avgsign = calculate_reweighted_sign(m, MC_data)
        E = calculate_reweighted_energy(m, MC_data) + N*E_bg
        @printf("%-30s KL= %.5f nats AvgSign= %.8f E= %.8f E/N= %.8f Ha = %.8f Ry \n", label, KL, avgsign, E, E / N, 2*E/N)
    end
    println()
    println("  DuBois:           ", model_dubois)
    println("  MAP:              ", model_map)
    println("  MaxEnt:           ", model_full)
    println("  LSTM (no prior):  ", model_lstm_flat)
    println("  LSTM (mult):      ", model_lstm_mult)
    println("  LSTM (DuBois):    ", model_lstm_DuBois)

    # ---------------------------------------------------------------
    # Write sector probabilities comparison
    # ---------------------------------------------------------------
    q_mult = probabilities(model_mult, MC_data)
    q_dubois = probabilities(model_dubois, MC_data)
    q_full = probabilities(model_full, MC_data)
    q_lflat = probabilities(model_lstm_flat, MC_data)
    q_lmult = probabilities(model_lstm_mult, MC_data)
    q_ldub = probabilities(model_lstm_DuBois, MC_data)
    n_tot = sum(MC_data.count)

    open("PermutationFamily_LSTM_comparison.dat", "w") do io
        println(io, "# k  count  P_hat  P_mult  P_dubois  P_full  P_lstm_flat  P_lstm_mult  P_lstm_dub  C")
        for k in 1:MC_data.n_families
            c = MC_data.count[k]
            p_hat = c > 0 ? c / n_tot : 0.0
            C_k = C_from_rank(k, N, MC_data.P)
            @printf(io, "%8d  %8d  %.5e  %.5e  %.5e  %.5e  %.5e  %.5e  %.5e  %s\n",
                k, c, p_hat, q_mult[k], q_dubois[k], q_full[k],
                q_lflat[k], q_lmult[k], q_ldub[k], string(C_k))
        end
    end
    println("Wrote PermutationFamily_LSTM_comparison.dat")

    return MC_data, models
end

if abspath(PROGRAM_FILE) == @__FILE__
    #    MC_and_fit_model(; N=20, θ=1.0, r_s=10.0, steps=10_000_000)
    #    MC_and_fit_model(; N=33, θ=0.125, r_s=1.0) # DuBois Table 1, Weak Coupling / High Density
    #MC_and_fit_model(; N=33, θ=0.125, r_s=10.0)# DuBois Table 1, Strong Coupling / Low Density
    #MC_and_fit_model(; N=33, r_s=1.0, θ=0.125

    # N is magic-number from filling 3D Fermi sphere
    #   So N=1, 7, 19, 33
    Nmagic=33
    magicsteps=5_000_000
    # DuBois Table 1: rs=1.0, theta=1.0 (N=33)
    #     Expected E/N: 8.69 Ha
    MC_and_fit_model(; N=Nmagic, θ=1.0, r_s=1.0, steps=magicsteps)
    # DuBois Table 1: rs=10.0, theta=1.0 (N=33)
    #     Expected E/N: -0.0403 Ha
    MC_and_fit_model(; N=Nmagic, θ=1.0, r_s=10.0, steps=magicsteps)
    
    # Low temperature: theta=0.125
    # rs=1.0 -> 2.35 Ha
    MC_and_fit_model(; N=Nmagic, θ=0.125, r_s=1.0, steps=magicsteps)
    # rs=10.0 -> -0.1038 Ha
    MC_and_fit_model(; N=Nmagic, θ=0.125, r_s=10.0, steps=magicsteps)

    # Dornheim et al. 2025, JCP 163, 154101 - Reweighting estimator
    # MC_and_fit_model(; N=4, r_s=0.5, θ=1.0, steps=100_000_000,)
    # N=4,28 ('standard reference')
    # N=40,66 in other figures

end
