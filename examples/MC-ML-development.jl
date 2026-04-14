# PermutationFamily-LSTM.jl
# Trains the LSTM autoregressive model on  MC data and compares
# KL divergence against the exponential family models.
# Three LSTM variants are fitted:
#   1. No prior (flat logit bias)
#   2. Multiplicity prior (combinatorial M(λ) factors for free)
#   3. DuBois prior (M(λ) + exchange penalty κ)
# Then also added an energy-model family (both linear models now following DuBois, and LSTMs on top of this)
# And also Wang-Landau style guided MC to try and flatten the histogram
#  Next step is to intend to bring this functionality into the main package, but leaving this here for reference.

using Halcyon
using Printf
using ProgressMeter
using Random
using LinearAlgebra
using Flux

"""Wigner-Seitz density n = 3/(4π r_s³) in 3D (atomic units)."""
ueg_density_3d(r_s::Float64) = 3.0 / (4π * r_s^3)

"""Cubic box side L for N electrons at given r_s."""
ueg_box_length(N::Int, r_s::Float64) = (N / ueg_density_3d(r_s))^(1 / 3)

"""Fermi wavevector for fully spin-polarised 3D gas."""
fermi_wavenumber_polarised(n::Float64) = (6π^2 * n)^(1 / 3)

"""Fermi wavevector for unpolarised 3D gas. NOTE WE ONLY SUPPORT FULLY SPIN POLARISED CURRENTLY."""
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
        prog !== nothing && t % 50 == 0 && next!(prog)
        if t % measure_every == 0 && cfg.sector == Z_SECTOR
            C_vec = permutation_family_C(cfg)
            E_therm, E_virial = energy_estimators(cfg, sys)
            #observe_permutation_family!(acc, C_vec, E_virial)
            observe_permutation_family_reservoir!(acc, C_vec, E_virial)
        end
    end
    acc
end

"""Split `steps` across `n_chains` (remainder to early chains); merge with `merge_stats`."""
function permutation_family_mc(sys, params; equil, steps, measure_every, show_progress::Bool=true, n_chains::Int=Threads.nthreads())
    @printf("Starting threaded MC across %d chains.\n", n_chains)
    q, r = divrem(steps, n_chains)
    base = rand(RandomDevice(), UInt64)
    parts = Vector{DensePermutationFamilyStats}(undef, n_chains)
    Threads.@threads for i in 1:n_chains
        nstep = q + (i ≤ r)
        prog = (show_progress && i == 1) ? Progress(nstep ÷ 50; desc="MC:$(steps/1e6)M") : nothing
        parts[i] = permutation_family_mc_chain(sys, params; equil,
            steps=nstep, measure_every, seed=base + UInt64(i), prog=prog)
    end

    jk = jackknife_statistics(parts)

    aggregate_statistics = reduce(merge_stats, parts)

    return (aggregate_statistics, jk)
end


function MC_and_fit_model(; N::Int=33, θ::Float64=0.5, r_s::Float64=2.0, M::Int=100,
    equil::Int=100_000, steps::Int=1_000_000, measure_every::Int=5,
    lstm_epochs::Int=500, lstm_hidden::Int=64, lstm_embed::Int=16,
    lstm_lr::Float64=3e-3, use_kelbg::Bool=true,
    prefix="")
    λħ = 0.5
    (; L, β) = ueg_theta_parameters(; N, θ, r_s, λ=λħ)

    # Kelbg smoothing parameter: λ² = ħ²τ / (2μ). For identical particles μ = m/2,
    # so λ² = ħ²τ / m = 2 * (ħ²/2m) * τ = 2 * λ_sys * (β/M)
    λ_kelbg = sqrt(2 * λħ * (β / M))

    pair_pot = use_kelbg ? YakubRonchiKelbgPotential(L=L, g=1.0, λ=λ_kelbg) : YakubRonchiPotential(L=L, g=1.0)
    sys = make_periodic_fermion_system(; M, N, β, L, λ=λħ, pair=pair_pot)
    params = default_worm_params(sys)

    @printf("|>|>|>|>|>0> WORM: N= %d θ= %g r_s= %g (L= %g β= %g λ= %g) \n", N, θ, r_s, L, β, λħ)
    if use_kelbg
        @printf("  Using Kelbg-smoothed potential with λ= %g, λ² = %g\n", λ_kelbg, λ_kelbg^2)
    end

    # Add Jellium Background; Yakub-Ronchi
    E_bg = yakub_ronchi_background_constant(L, N)
    @printf("Calculated Yakub-Ronchi background N= %d L= %g E= %g Ha\n", N, L, E_bg)

    println("Running threaded MC...")
    MC_data, jk = permutation_family_mc(sys, params; equil, steps, measure_every)

    n_z = sum(MC_data.count)
    @printf("\n MC Complete! N= %d  r_s= %g θ= %g  n_families= %d  Visited_families= %d  Z-samples=%d\n", N, r_s, θ, MC_data.n_families, length(MC_data.count), n_z)

    # save reservoir of samples
    open("$(prefix)_reservoir.dat", "w") do io
        println(io, "# Estimator reservoir")

        # print permutation families as column heads
        for k in 1:MC_data.n_families
            @printf(io, "\"%s\" ", C_from_rank(k, MC_data.N, MC_data.P))
        end
        @printf(io, "\n")

        R = length(MC_data.reservoir[1])
        for r in 1:R
            for k in 1:MC_data.n_families
                E = MC_data.reservoir[k][r]
                if E == 0.0
                    @printf(io, "NaN ") # if as initialised, no more data; throughs off plots
                else
                    @printf(io, "%g ", E)
                end
            end
            @printf(io, "\n")
        end
    end
    println("Reservoir written; Well you should let me know when you're home and dry")

    # Calculate permutation histogram from MC data
    MC_permutation_histogram = permutation_histogram_from_stats(MC_data)
    @printf("MC P(ℓ): [%s]\n", join([@sprintf("%.4f", x) for x in MC_permutation_histogram], ", "))

    println("DensePermutationFamilyStats  N=$N  p(N)=$(MC_data.n_families)  ",
        "size=$(Base.summarysize(MC_data) ÷ 1024) KiB")
    E_MC, sigma = calculate_mc_energy(MC_data)
    @printf("E_MC= %.8f signσ=  %.8f E_MC/N= %.8f Ha = %.8f Ry (including Yakub-Ronchi Jellium background)\n",
        E_MC + N * E_bg, sigma, (E_MC + N * E_bg) / N, 2 * (E_MC + N * E_bg) / N)

    # OK, now we start doing the weird stuff! What Would Ceperley Do? (WWCD?)
    # Set smoothness to massive to enforce the Feynman/DuBois exchange penalty model
    # I think this is what they used for low T ?
    smoothness = (θ <= 0.5) ? 1e12 : 1e4
    linearEmodel = fit_LinearEnergyModel(LinearEnergyModel, MC_data; λ_smooth=smoothness)
    @show linearEmodel
    E_lin = calculate_reweighted_energy(fit(MultiplicityModel, MC_data), linearEmodel, MC_data) + (N * E_bg)
    @printf("E_Linear= %.8f signσ=  %.8f E_Linear/N= %.8f Ha = %.8f Ry (including Yakub-Ronchi Jellium background)\n",
        E_lin, sigma, E_lin / N, 2 * E_lin / N)


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
    println("Unique data points (per epoch): $(length(MC_data.count)) / p(N)=$(MC_data.n_families)")

    lstm_kw = (; n_embed=lstm_embed, n_hidden=lstm_hidden, epochs=lstm_epochs, lr=lstm_lr)

    # 1. Multiplicity prior — gets M(λ) factors for free (θ=0)
    #   - essentially this just tells us that LSTM alone is terrible, KL divergence stays massive
    println("\nTraining LSTM (multiplicity prior)...")
    model_lstm_mult = fit(LSTMPermutationModel, MC_data; prior=model_mult, lstm_kw...)
    # 2. DuBois prior — gets M(λ) + κ-penalty for free
    println("\nTraining LSTM (DuBois prior)...")
    model_lstm_DuBois = fit(LSTMPermutationModel, MC_data; prior=model_dubois, lstm_kw...)
    # 3. LSTM on MAP, so already optim Bayesian Theta-model
    println("\nTraining LSTM (MAP-on-DuBois prior)...")
    model_lstm_MAP = fit(LSTMPermutationModel, MC_data; prior=model_map, lstm_kw...)

    # ---------------------------------------------------------------
    # Extracted Energy LSTM Runs
    # ---------------------------------------------------------------
    println("\nTraining Shallow Energy Head on MAP-on-DuBois LSTM trunk...")
    lstm_energy_shallow = fit_LSTMEnergyResidualModel(model_lstm_MAP, linearEmodel, MC_data; hidden_layers=(32,), epochs=500)

    println("\nTraining Deep Energy Head on MAP-on-DuBois LSTM trunk...")
    lstm_energy_deep = fit_LSTMEnergyResidualModel(model_lstm_MAP, linearEmodel, MC_data; hidden_layers=(64, 64, 32,), epochs=500)

    # ---------------------------------------------------------------
    # Compare KL divergences
    # ---------------------------------------------------------------
    models = [
        ("Infinite-T", model_mult),
        ("DuBois κ", model_dubois),
        ("MAP-on-DuBois", model_map),
        ("MaxEnt l2=1e-4", model_full),
        ("LSTM (M(λ) prior)", model_lstm_mult),
        ("LSTM (DuBois κ prior)", model_lstm_DuBois),
        ("LSTM (MAP-on-DuBois prior)", model_lstm_MAP),
    ]

    @printf("\n# Estimates: (including Yakub-Ronchi Jellium background)")
    @printf("\n\n%-30s KL= %8.4f signσ= %8.4f E= %8.2f E/N= %8.4f Ha = %8.4f Ry\n",
        "MC/MC", 0.0,
        sigma, E_MC + N * E_bg, (E_MC + N * E_bg) / N, 2 * (E_MC + N * E_bg) / N)
    # really need to write a fn to eval with the linear engine moodel, but take the counts direcyl from MC

    for (label, m) in models
        KL = kl_divergence(m, MC_data)
        avgsign = calculate_reweighted_sign(m, MC_data)
        E = calculate_reweighted_energy(m, MC_data) + (N * E_bg)
        E_lin = calculate_reweighted_energy(m, linearEmodel, MC_data) + (N * E_bg)
        E_shallow = calculate_reweighted_energy(m, lstm_energy_shallow, MC_data) + (N * E_bg)
        E_deep = calculate_reweighted_energy(m, lstm_energy_deep, MC_data) + (N * E_bg)

        #        @printf("%-30s KL=%8.4f signσ=% 8.4f \n\tE_MC/N % 8.4f Ry | E_Lin/N % 8.4f Ry | E_Shal/N % 8.4f Ry | E_Deep/N % 8.4f Ry\n",
        #            label, KL, avgsign, 2 * E / N, 2 * E_lin / N, 2 * E_shallow / N, 2 * E_deep / N)

        @printf("  %-30s KL=%8.4f signσ=% 8.4f | E_MC/N % 8.4f Ry | E_Lin/N % 8.4f Ry | E_Shal/N % 8.4f Ry | E_Deep/N % 8.4f Ry\n",
            label, KL, avgsign,
            2 * E / N, 2 * E_lin / N, 2 * E_shallow / N, 2 * E_deep / N)

    end

    @printf("\n# Models:")
    for (label, m) in models
        @printf("%-30s ", label)
        println(m)
    end

    # ---------------------------------------------------------------
    # Write sector probabilities comparison
    # ---------------------------------------------------------------
    q_mult = probabilities(model_mult, MC_data)
    q_dubois = probabilities(model_dubois, MC_data)
    q_full = probabilities(model_full, MC_data)
    q_lmult = probabilities(model_lstm_mult, MC_data)
    q_ldub = probabilities(model_lstm_DuBois, MC_data)
    q_lMAP = probabilities(model_lstm_MAP, MC_data)
    n_tot = sum(MC_data.count)

    filename = "$(prefix)_PermutationFamily_ProbabilityModel.dat"
    open(filename, "w") do io
        println(io, "# k  count  P_hat  P_mult  P_dubois  P_full  P_lstm_mult  P_lstm_DuBois  P_lstm_MAP  PC")
        for k in 1:MC_data.n_families
            c = MC_data.count[k]
            p_hat = c > 0 ? c / n_tot : 0.0
            C_k = C_from_rank(k, N, MC_data.P)
            @printf(io, "%8d  %8d  %.5e  %.5e  %.5e  %.5e  %.5e  %.5e  %.5e  %s\n",
                k, c, p_hat, q_mult[k], q_dubois[k], q_full[k],
                q_lmult[k], q_ldub[k], q_lMAP[k], string(C_k))
        end
    end
    println("Wrote $(filename).")

    # ===============================================================
    # Importance-sampled MC using fitted model as bias
    # ===============================================================
    bias_α = 0.7  # gentle softening; α=1.0 for full flat-histogram; sort of Wang-Landau
    # experiments showed α=1.0 was terrible - threw MC into the OPPOSITE undersampling, i.e.
    # forced condensation if not present
    biasmodel = model_lstm_MAP
    bias = make_permutation_bias(biasmodel, MC_data; α=bias_α)
    biasmodel = "jackknife"
    bias = jackknife_error_guide(bias, jk; α=bias_α) # jacknife bias: minimise p*E error

    println("\n#### Biased MC (α=$bias_α, model=$biasmodel) ####")
    params_biased = default_worm_params(sys)
    params_biased.bias = bias

    MC_biased, jk = permutation_family_mc(sys, params_biased; equil, steps, measure_every)

    MC_biased_permutation_histogram = permutation_histogram_from_stats(MC_biased)
    @printf("MC Biased P(ℓ): [%s]\n", join([@sprintf("%.4f", x) for x in MC_biased_permutation_histogram], ", "))

    MC_data_visited = count(>(0), MC_data.count)
    MC_biased_visited = count(>(0), MC_biased.count)
    @printf("  Unbiased: visited %d / %d sectors (%d Z-samples)\n",
        MC_data_visited, MC_data.n_families, sum(MC_data.count))
    @printf("  Biased:   visited %d / %d sectors (%d Z-samples)\n",
        MC_biased_visited, MC_biased.n_families, sum(MC_biased.count))

    # ── Debiased energy estimator ──
    E_debiased, sign_debiased = debiased_mc_energy(MC_biased, bias)
    println("> probability factor of one to one ... we have normality, I repeat we  have  normality.")
    @printf("  E_MC_debiased = %.8f  signσ=%.4f  E/N= %.8f Ha = %.8f Ry\n",
        E_debiased + N * E_bg, sign_debiased,
        (E_debiased + N * E_bg) / N, 2 * (E_debiased + N * E_bg) / N)

    # ── Refit energy models on biased data (for better rare-sector coverage?) ──
    println("\n  Refitting Linear Energy Model on biased data...")
    linearEmodel_biased = fit_LinearEnergyModel(LinearEnergyModel, MC_biased; λ_smooth=smoothness)

    println("  Refitting LSTM Energy Heads on biased data...")
    lstm_energy_shallow_biased = fit_LSTMEnergyResidualModel(
        model_lstm_MAP, linearEmodel_biased, MC_biased; hidden_layers=(), epochs=500)
    lstm_energy_deep_biased = fit_LSTMEnergyResidualModel(
        model_lstm_MAP, linearEmodel_biased, MC_biased; hidden_layers=(32,), epochs=500)

    @printf("\n# Biased MC Energy Estimates (α=%.2f, debiased):\n", bias_α)
    @printf("%-30s  σ=% 8.4f  E/N= % 8.4f Ry (debiased MC)\n",
        "Debiased MC", sign_debiased, 2 * (E_debiased + N * E_bg) / N)

    for (label, m) in models
        E_mc_b = calculate_reweighted_energy(m, MC_biased) + (N * E_bg)
        E_lin_b = calculate_reweighted_energy(m, linearEmodel_biased, MC_biased) + (N * E_bg)
        E_shal_b = calculate_reweighted_energy(m, lstm_energy_shallow_biased, MC_biased) + (N * E_bg)
        E_deep_b = calculate_reweighted_energy(m, lstm_energy_deep_biased, MC_biased) + (N * E_bg)

        @printf("  %-30s E_MC/N % 8.4f Ry | E_Lin/N % 8.4f Ry | E_Shal/N % 8.4f Ry | E_Deep/N % 8.4f Ry\n",
            label, 2 * E_mc_b / N, 2 * E_lin_b / N, 2 * E_shal_b / N, 2 * E_deep_b / N)
    end

    @printf("\n# Biased MC Estimates (α=%.2f, debiased):\n", bias_α)



    return MC_data, MC_biased, bias, models
end

if abspath(PROGRAM_FILE) == @__FILE__
    #    MC_and_fit_model(; N=20, θ=1.0, r_s=10.0, steps=10_000_000)
    #    MC_and_fit_model(; N=33, θ=0.125, r_s=1.0) # DuBois Table 1, Weak Coupling / High Density
    #MC_and_fit_model(; N=33, θ=0.125, r_s=10.0)# DuBois Table 1, Strong Coupling / Low Density
    #MC_and_fit_model(; N=33, r_s=1.0, θ=0.125

    # N is magic-number from filling 3D Fermi sphere
    #   So N=1, 7, 19, 33
    Nmagic = 33
    magicsteps = 12_000_000
    # DuBois Table 1: rs=1.0, theta=1.0 (N=33)
    #     Expected E/N: 8.69 Ha
    #    MC_and_fit_model(; N=Nmagic, θ=1.0, r_s=1.0, steps=magicsteps, prefix="N$(Nmagic)_θeq1_rseq1")
    # DuBois Table 1: rs=10.0, theta=1.0 (N=33)
    #     Expected E/N: -0.0403 Ha
    #    MC_and_fit_model(; N=Nmagic, θ=1.0, r_s=10.0, steps=magicsteps, prefix="N$(Nmagic)_θeq1_rseq10")

    #    magicsteps = 1_000_000
    # Low temperature: theta=0.125
    # rs=1.0 -> 2.35 Ha
    MC_and_fit_model(; N=Nmagic, θ=0.125, r_s=1.0, steps=magicsteps, prefix="N$(Nmagic)_θeq0p125_rseq1")
    # rs=10.0 -> -0.1038 Ha
    #MC_and_fit_model(; N=Nmagic, θ=0.125, r_s=10.0, steps=magicsteps, prefix="N$(Nmagic)_θeq0p125_rseq10")

    # Dornheim et al. 2025, JCP 163, 154101 - Reweighting estimator
    # MC_and_fit_model(; N=4, r_s=0.5, θ=1.0, steps=100_000_000,)
    # N=4,28 ('standard reference')
    # N=40,66 in other figures
end
