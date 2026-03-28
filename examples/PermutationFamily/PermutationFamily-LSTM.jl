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
                                 equil::Int=100_000, steps::Int=10_000_000, measure_every::Int=5,
                                 lstm_epochs::Int=200, lstm_hidden::Int=64, lstm_embed::Int=16,
                                 lstm_lr::Float64=1e-3)
    λħ = 0.5
    (; L, β) = ueg_theta_parameters(; N, θ, r_s, λ=λħ)
    sys = make_periodic_fermion_system(; M, N, β, L, λ=λħ, pair=YakubRonchiPotential(L=L,g=1))
    params = default_worm_params(sys)
    cfg = WormConfiguration(sys)

    for _ in 1:equil
        worm_step!(cfg, sys, params)
    end

    MC_data = DensePermutationFamilyStats(N)
    println("Allocated DensePermutationFamilyStats  N=$N  p(N)=$(MC_data.n_families)  ",
            "size=$(Base.summarysize(MC_data) ÷ 1024) KiB")
    n_z = 0

    @showprogress desc="MC:$(steps/1E6)M" for t in 1:steps
        worm_step!(cfg, sys, params)
        if t % measure_every == 0 && cfg.sector == Z_SECTOR
            λ_vec = permutation_family_lambda(cfg)
            observe_permutation_family!(MC_data, λ_vec, 0.0)
            n_z += 1
        end
    end
    n_z == 0 && error("No Z-sector samples")
    println(@sprintf("\n# Ideal fermions  N=%d  θ=%g  p(N)=%d  Z-samples=%d\n", N, θ, MC_data.n_families, n_z))

    # ---------------------------------------------------------------
    # Exponential family fits
    # ---------------------------------------------------------------
    println("Fitting exponential family models...")
    model_mult   = fit(MultiplicityModel, MC_data)
    model_dubois = fit(DuBoisModel, MC_data)
    model_map    = fit(MAPHybridModel, MC_data; prior=model_dubois, τ₀=1.0)
    model_full   = fit(MaxEntModel, MC_data; l2_reg=1e-4)

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
        ("DuBois 1-param",         model_dubois),
        ("MAP hybrid",             model_map),
        ("Full MaxEnt",            model_full),
        ("LSTM (no prior)",        model_lstm_flat),
        ("LSTM (mult prior)",      model_lstm_mult),
        ("LSTM (DuBois prior)",    model_lstm_DuBois),
    ]

    println("\nModel fits:")
    for (label, m) in models
        @printf("  KL %-30s = %.5f nats\n", label, kl_divergence(m, MC_data))
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
    q_mult   = probabilities(model_mult,       MC_data)
    q_dubois = probabilities(model_dubois,     MC_data)
    q_full   = probabilities(model_full,       MC_data)
    q_lflat  = probabilities(model_lstm_flat,  MC_data)
    q_lmult  = probabilities(model_lstm_mult,  MC_data)
    q_ldub   = probabilities(model_lstm_DuBois,   MC_data)
    n_tot = sum(MC_data.count)

    open("PermutationFamily_LSTM_comparison.dat", "w") do io
        println(io, "# k  count  P_hat  P_mult  P_dubois  P_full  P_lstm_flat  P_lstm_mult  P_lstm_dub  λ")
        for k in 1:MC_data.n_families
            c = MC_data.count[k]
            p_hat = c > 0 ? c / n_tot : 0.0
            λk = permutation_family_lambda_from_rank(k, N, MC_data.P)
            r = findfirst(iszero, λk)
            head = isnothing(r) ? λk : λk[1:(r - 1)]
            @printf(io, "%8d  %8d  %.5e  %.5e  %.5e  %.5e  %.5e  %.5e  %.5e  %s\n",
                    k, c, p_hat, q_mult[k], q_dubois[k], q_full[k],
                    q_lflat[k], q_lmult[k], q_ldub[k], string(collect(head)))
        end
    end
    println("Wrote PermutationFamily_LSTM_comparison.dat")

    return MC_data, models
end

if abspath(PROGRAM_FILE) == @__FILE__
#    MC_and_fit_model(; N=33, θ=0.5, r_s=2.0)
#    MC_and_fit_model(; N=33, θ=0.125, r_s=1.0) # DuBois Table 1, Weak Coupling / High Density
#MC_and_fit_model(; N=33, θ=0.125, r_s=10.0)# DuBois Table 1, Strong Coupling / Low Density
MC_and_fit_model(; N=33, r_s=1.0, θ=0.125)


end
