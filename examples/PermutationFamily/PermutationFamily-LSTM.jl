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
using LinearAlgebra

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

abstract type AbstractEnergyModel end

struct LinearEnergyModel <: AbstractEnergyModel
    # Am I Ulysses, am I Ulysses?
    E_l::Vector{Float64}
end

function eval_energy(model::LinearEnergyModel, C::Vector{Int})
    return dot(model.E_l, C)
end

function fit_LinearEnergyModel(::Type{LinearEnergyModel}, stats::DensePermutationFamilyStats; λ_ridge=1e-6)
    N = stats.N
    n_visited = count(c -> c > 0, stats.count)

    # Gauge-fixed Linear Model: E_tot = E_MF * N + sum_{l=2}^N C_l * Δ_l
    # This mathematically resolves the fundamental dimension collinearity (sum(C_l * l) = N)
    # by anchoring the 1-cycle per-particle mean-field energy strictly to E_MF.
    # X_gf: Design matrix (n_visited × N). Col 1 is exactly N. Col 2..N are C_l for l=2..N.
    X_gf = zeros(Float64, n_visited, N)
    y = zeros(Float64, n_visited)
    W = zeros(Float64, n_visited)

    # collect all the data and shove it in matrices / vectors
    idx = 1
    for k in 1:stats.n_families
        count = stats.count[k]
        if count > 0
            C_k = C_from_rank(k, N, stats.P)
            X_gf[idx, 1] = N # mean-field contribution; N*E_MF
            for l in 2:N
                X_gf[idx, l] = C_k[l] # + delta for the individual cycles
            end

            E_mean = stats.estimator[k] / count
            y[idx] = E_mean

            # Linear-squares weight = N_k / Var_k: nb: currently just taking MC sample variance
            if count > 1
                var_k = (stats.estimator2[k] - count * E_mean^2) / (count - 1)
                var_k = max(var_k, 1e-12) # Safeguard against identical samples
                W[idx] = count / var_k
            else
                W[idx] = count
            end
            idx += 1
        end
    end

    # WWDD? (What Would David MacKay Do?)
    # Smoothness Prior on Δ_l (which cleanly translates to smoothing E_l)
    # Penalize second derivative of Δ_l to ensure linearity for unvisited cycles: 
    # extrapolate off into rarely visited cycle-space
    λ_smooth = 1e4 # prefactor for the design matrix
    D_mat = zeros(Float64, N - 2, N) # linear algebra is pretty mind blowing
    if N >= 3
        # First row penalizes second deriv around l=2: Δ_3 - 2Δ_2 + Δ_1, but Δ_1 = 0
        D_mat[1, 2] = -2.0
        D_mat[1, 3] = 1.0
        for i in 2:(N-2)
            D_mat[i, i] = 1.0  # Δ_i
            D_mat[i, i+1] = -2.0  # Δ_{i+1}
            D_mat[i, i+2] = 1.0  # Δ_{i+2}
        end
    end

    # Solve the Normal Equations
    W_diag = Diagonal(W)
    H = X_gf' * W_diag * X_gf +
        λ_ridge * I +
        λ_smooth * (D_mat' * D_mat)
    rhs = X_gf' * W_diag * y
    coeffs = H \ rhs # big bada boom


    # After all that high falutin stuff, we can STILL construct an absolute E_l vector: 
    #  this we can then simply dot-product still with C_k vector to get total energy of the cycle
    E_MF = coeffs[1]
    E_l = zeros(Float64, N)
    E_l[1] = E_MF
    for l in 2:N
        E_l[l] = E_MF * l + coeffs[l]
    end

    # --- Telemetry & Fit Quality ---
    y_pred = X_gf * coeffs
    WSSR = sum(W .* (y .- y_pred) .^ 2)
    y_mean_w = sum(W .* y) / max(sum(W), 1e-12)
    WTSS = sum(W .* (y .- y_mean_w) .^ 2)
    R2_W = 1.0 - (WSSR / max(WTSS, 1e-12))

    H_inv = inv(H)
    # Extract variances of reconstructed E_l
    # E_l = ε_bg * l + Δ_l = l * coeffs[1] + coeffs[l]
    # Var(E_l) = l^2 Var(coeffs[1]) + Var(coeffs[l]) + 2 * l * Cov(coeffs[1], coeffs[l])
    se_E_l = zeros(Float64, N)
    se_E_l[1] = sqrt(max(H_inv[1, 1], 0.0))
    for l in 2:N
        var_l = l^2 * H_inv[1, 1] + H_inv[l, l] + 2.0 * l * H_inv[1, l]
        se_E_l[l] = sqrt(max(var_l, 0.0))
    end

    @printf("Linear Model Telemetry: E_MF= %8.4f\n", E_MF)
    @printf("  Linear model acheives, Weighted R² = % .6f\n", R2_W)
    @printf("  E_l \t RawValue \t\t E_correlation \t Std Error\n")
    for l in 1:min(8, N)
        @printf("  E_%-2d \t %8.4f \t %8.4f \t %8.4f\n", l, E_l[l], (E_l[l] - l * E_l[1]), se_E_l[l])
    end

    return LinearEnergyModel(E_l)
end

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

function calculate_reweighted_energy(prob_model::AbstractPermutationModel, en_model::AbstractEnergyModel, stats::DensePermutationFamilyStats)
    q = probabilities(prob_model, stats)
    N = stats.N

    Etot = 0.0
    Z_sign = 0.0
    for k in 1:stats.n_families
        C_k = C_from_rank(k, N, stats.P) # grab the permutation-family / cycle vector
        E_k = eval_energy(en_model, C_k) # this is a dot product for a linear model: additive energy
        # expectation is that we will need to fit an LSTM for the win

        n_cycles = sum(C_k)
        sigma = iseven(N - n_cycles) ? 1.0 : -1.0 # sign, should make this a fictitious option

        Etot += sigma * q[k] * E_k # do you see the magic? NO DIRECT MC QUANTITY HERE!
        Z_sign += sigma * q[k]
    end
    return Etot / Z_sign # and don't forget!
end


function calculate_mc_energy(stats::DensePermutationFamilyStats)
    N = stats.N
    n_z = sum(stats.count)

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
    return Etot / Z_sign, Z_sign
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

function permutation_histogram_from_stats(stats::DensePermutationFamilyStats)
    N = stats.N
    n_z = sum(stats.count)

    hist = zeros(Int, N) # for each cycle

    for k in 1:stats.n_families
        if stats.count[k] > 0
            C_k = C_from_rank(k, N, stats.P)
            for i in 1:N
                hist[i] += C_k[i] * stats.count[k]
            end
        end
    end

    return hist ./ n_z
end

function merge_stats(a::DensePermutationFamilyStats, b::DensePermutationFamilyStats)
    a.N == b.N && a.n_families == b.n_families || throw(ArgumentError("merge_stats"))
    DensePermutationFamilyStats(a.N, a.P, a.n_families,
        a.count .+ b.count, a.estimator .+ b.estimator, a.estimator2 .+ b.estimator2)
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
        prog !== nothing && t % 50 == 0 && next!(prog)
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
        prog = (show_progress && i == 1) ? Progress(nstep ÷ 50; desc="MC:$(steps/1e6)M") : nothing
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
    @printf("Calculated background N= %d L= %d E= %g\n", N, L, E_bg)

    println("Running threaded MC...")
    MC_data = permutation_family_mc(sys, params; equil, steps, measure_every)

    n_z = sum(MC_data.count)
    @printf("\n MC Complete! N= %d  r_s= %g θ= %g  n_families= %d  Z-samples=%d\n", N, r_s, θ, MC_data.n_families, n_z)

    # Calculate permutation histogram from MC data
    MC_permutation_histogram = permutation_histogram_from_stats(MC_data)
    @printf("MC_permutation_histogram: %s\n", MC_permutation_histogram)

    println("DensePermutationFamilyStats  N=$N  p(N)=$(MC_data.n_families)  ",
        "size=$(Base.summarysize(MC_data) ÷ 1024) KiB")
    E_MC, sigma = calculate_mc_energy(MC_data)
    @printf("E_MC= %.8f σ=  %.8f E_MC/N= %.8f Ha = %.8f Ry (including Yakub-Ronchi Jellium background)\n",
        E_MC + N * E_bg, sigma, (E_MC + N * E_bg) / N, 2 * (E_MC + N * E_bg) / N)

    # OK, now we start doing the weird stuff! What Would Ceperley Do? (WWCD?)
    linearEmodel = fit_LinearEnergyModel(LinearEnergyModel, MC_data)
    @show linearEmodel
    E_lin = calculate_reweighted_energy(fit(MultiplicityModel, MC_data), linearEmodel, MC_data) + (N * E_bg)
    @printf("E_Linear= %.8f σ=  %.8f E_Linear/N= %.8f Ha = %.8f Ry (including Yakub-Ronchi Jellium background)\n",
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
    println("Unique data points (per epoch): $(length(MC_data.count))")

    lstm_kw = (; n_embed=lstm_embed, n_hidden=lstm_hidden, epochs=lstm_epochs, lr=lstm_lr)

    # 1. Multiplicity prior — gets M(λ) factors for free (θ=0)
    println("Training LSTM (multiplicity prior)...")
    model_lstm_mult = fit(LSTMPermutationModel, MC_data; prior=model_mult, lstm_kw...)
    # 2. DuBois prior — gets M(λ) + κ-penalty for free
    println("Training LSTM (DuBois prior)...")
    model_lstm_DuBois = fit(LSTMPermutationModel, MC_data; prior=model_dubois, lstm_kw...)
    # 3. LSTM on MAP, so already optim Bayesian Theta-model
    println("Training LSTM (MAP-on-DuBois prior)...")
    model_lstm_MAP = fit(LSTMPermutationModel, MC_data; prior=model_map, lstm_kw...)

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
    @printf("\n\n%-30s KL= %8.4f σ= %8.4f E= %8.2f E/N= %8.4f Ha = %8.4f Ry\n",
        "MC/MC", 0.0,
        sigma, E_MC + N * E_bg, (E_MC + N * E_bg) / N, 2 * (E_MC + N * E_bg) / N)
    # really need to write a fn to eval with the linear engine moodel, but take the counts direcyl from MC

    for (label, m) in models
        KL = kl_divergence(m, MC_data)
        avgsign = calculate_reweighted_sign(m, MC_data)
        E = calculate_reweighted_energy(m, MC_data) + (N * E_bg)
        E_lin = calculate_reweighted_energy(m, linearEmodel, MC_data) + (N * E_bg)

        @printf("%-30s KL= %8.4f σ= %8.4f E_MC= %8.5f E_lin= %8.5f E_MC/N(Ry)=%8.4f Ry E_lin/N(Ry)=%8.4f Ry\n",
            label, KL, avgsign, E, E_lin, 2 * E / N, 2 * E_lin / N)
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

    open("PermutationFamily_LSTM_comparison.dat", "w") do io
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
    Nmagic = 33
    magicsteps = 10_000_000
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
