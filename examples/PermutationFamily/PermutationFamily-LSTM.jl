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
using Flux

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

# Fallback wrapper for linear model and any future non-cached models; directly eval
eval_energy_at_rank(en_model::AbstractEnergyModel, k::Int, C_k::Vector{Int}) = eval_energy(en_model, C_k)

# ===================================================================
# LSTMEnergyResidualModel (simple dense network on top of pre-trained permutation LSTM model)
# ===================================================================

struct LSTMEnergyResidualModel{P<:AbstractPermutationModel,L<:LinearEnergyModel} <: AbstractEnergyModel
    linear_baseline::L
    prob_trunk::P      # Permutation probability model; LSTM (I'm not sure if we could use an exp model, as then no state vector to learn off?)
    energy_head::Chain # The dense MLP to train mapping representation to Delta E
    cached_energies::Vector{Float64} # O(1) lookup over all families
end

Flux.@layer LSTMEnergyResidualModel trainable = (energy_head,)

# Fast path evaluation using precomputed cached_energies
eval_energy_at_rank(en_model::LSTMEnergyResidualModel, k::Int, C_k::Vector{Int}) = en_model.cached_energies[k]
eval_energy(model::LSTMEnergyResidualModel, C::Vector{Int}) = error("Use eval_energy_at_rank")

function fit_LSTMEnergyResidualModel(prob_model::AbstractPermutationModel, linear_model::LinearEnergyModel, stats::DensePermutationFamilyStats;
    hidden_layers=(32,), lr=1e-3, epochs=500)

    N = stats.N
    # 1. Steal the permutation LSTM data setup; this is what it was trained on
    Xs, Ys_oh, _, M_vals = Halcyon._build_training_data(stats)

    trunk = prob_model.chain[1:end-1]
    Flux.testmode!(trunk)

    # Xs is (N, n_obs). trunk outputs (n_hidden, N, n_obs) in Flux
    hidden_seq = trunk(Xs)
    h_final = hidden_seq[:, end, :] # (n_hidden, n_obs) - the terminal representation
    n_hidden_in = size(h_final, 1) # dimension of hidden representation to be fed into our NN

    # 2. Build explicit numeric WLS (Weighted Least Squares) residual targets
    n_obs = count(c -> c > 0, stats.count)
    delta_E = zeros(Float32, 1, n_obs)
    E_raw = zeros(Float32, n_obs)
    W_arr = zeros(Float32, n_obs)

    # 3. Global (g_) variance for a Bayes prior on the sample variance
    # Filter out unvisited sectors (with a boolean mask)
    g_visited = stats.count .> 0
    g_c    = stats.count[g_visited]
    g_est  = stats.estimator[g_visited]
    g_est2 = stats.estimator2[g_visited]
    # Total variance across all samples 
    g_N = sum(g_c)
    g_mean = sum(g_est) / g_N
    g_var = max((sum(g_est2) / g_N) - g_mean^2, 1e-12)  # usual 1e-12 to stop underflow

    # bit painful this really; not sure if I should fill with dummy values or whatever... once we get N>100, we'll need to be more clever generally
    idx = 1
    for k in 1:stats.n_families
        c_k = stats.count[k]
        if c_k > 0
            E_mean = stats.estimator[k] / c_k # sample (permutation family) mean
            E_var  = max.((stats.estimator2[k] ./ c_k) .- E_mean.^2, 0.0) # sample var

            E_raw[idx] = Float32(E_mean)

            C_k = C_from_rank(k, N, stats.P)
            E_lin = eval_energy(linear_model, C_k)

            delta_E[1, idx] = Float32(E_mean - E_lin)

            # variance-smooth to attempt do use sample-variance without the previous
            # statistical blow-up for rare cycles; always have x of global variance mixed in
            # Empirical Bayes Variance Shrinkage
            # Act as if every sector has seen γ additional "global average" observation
            #  γ = 1, Laplace smoothing: 'uninformative' prior
            #  γ = 2, Inverse-Gamma prior ?
            γ = 3.0

            var_smooth = (c_k .* E_var .+ γ * g_var) ./ (c_k .+ γ)
            W_arr[idx] = Float32(c_k) / var_smooth
            # Was directly using MC counts... incorporating variance blew out atteniont to trivial regions
            @printf("LSTM Var smooth: Cycle= %d c_k= %d g_var= %g E_var= %g var_smooth= %g W_arr[%d]= %g\n",
                    k,c_k,g_var,E_var,var_smooth,idx,W_arr[idx])
            idx += 1
        end
    end

    W_arr ./= sum(W_arr) # rescale for stable gradient magnitudes

    # 3. Construct purely unregularised scalar regression head
    layers = []
    in_dim = n_hidden_in
    for h in hidden_layers
        if h > 0
            push!(layers, Dense(in_dim => h, relu))
            in_dim = h
        end
    end
    push!(layers, Dense(in_dim => 1))

    head = Chain(layers...)
    opt_state = Flux.setup(Adam(lr), head)

    # Weighted Least Squares Loss over the empirical delta batch
    wls_loss(m, x, y, w) = sum(w' .* (m(x) .- y) .^ 2)

    println("  Training LSTM Energy Head: ", head)
    for epoch in 1:epochs
        grad = Flux.gradient(head) do m
            wls_loss(m, h_final, delta_E, W_arr)
        end
        Flux.update!(opt_state, head, grad[1])

        if epoch % 100 == 0 || epoch == 1 # trains super fast as so simple
            l = wls_loss(head, h_final, delta_E, W_arr)
            @printf("  LSTM Energy Head (Layers %s) epoch %d/%d  WLS-MSE = %.2e\n", string(hidden_layers), epoch, epochs, Float64(l))
        end
    end
    Flux.testmode!(head)

    # Calculate Total Energy Model R² for telemetry
    pred_delta = Float64.(head(h_final))
    # E_total_pred = E_lin + Delta_pred = (E_raw - delta_E_target) + Delta_pred
    y_pred_total = vec(E_raw) .- vec(delta_E) .+ vec(pred_delta)

    WSSR = sum(W_arr .* (vec(E_raw) .- y_pred_total) .^ 2)
    y_mean_w = sum(W_arr .* vec(E_raw)) / max(sum(W_arr), 1e-12)
    WTSS = sum(W_arr .* (vec(E_raw) .- y_mean_w) .^ 2)
    R2_W = 1.0 - (WSSR / max(WTSS, 1e-12))

    @printf("  LSTM %s achieves Total Weighted R² = %2.3f %%\n", string(hidden_layers), 100 * R2_W)

    # 4. Cache evaluations over ALL permutation families; probably should move this into separate function
    Cmat = Halcyon.cycle_count_matrix(stats)
    bos = Int32(N + 2)

    xs = zeros(Int32, N, stats.n_families)
    for idx in 1:stats.n_families
        xs[1, idx] = bos
        C_counts = view(Cmat, idx, :)
        for t in 2:N
            k_prev = N - (t - 1) + 1
            xs[t, idx] = Int32(C_counts[k_prev] + 1)
        end
    end

    h_f = trunk(xs)[:, end, :] # (hidden, n_families)
    delta_E_pred = Float64.(head(h_f)) # (1, n_families)

    cached_energies = [eval_energy(linear_model, C_from_rank(idx, N, stats.P)) + delta_E_pred[1, idx]
                       for idx in 1:stats.n_families]

    return LSTMEnergyResidualModel(linear_model, prob_model, head, cached_energies)
end

function fit_LinearEnergyModel(::Type{LinearEnergyModel}, stats::DensePermutationFamilyStats; λ_ridge_Δ=1e-6, λ_smooth=1e4)
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
            W[idx] = count # now directly using MC counts ONLY
            idx += 1
        end
    end

    # WWDD? (What Would David MacKay Do?)
    # Smoothness Prior on Δ_l (which cleanly translates to smoothing E_l)
    # Penalize second derivative of Δ_l to ensure linearity for unvisited cycles: 
    # extrapolate off into rarely visited cycle-space
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
    P_ridge = Diagonal([0.0; fill(λ_ridge_Δ, N - 1)])
    # Do not penalize E_MF; therefore if you take λ_ridge_Δ -> ∞, 
    #  you recover the Ideal-gas limit of DuBois (just E_l=l*E_MF)
    # Take λ_smooth -> ∞ to enforce strict linearity (no curvature), 
    #  so recovers the exchange-penalty model of Feynman as described by DuBois
    H = X_gf' * W_diag * X_gf +
        P_ridge +
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
    WSSR = sum(W .* (y .- y_pred) .^ 2) # weighted sum of squared residuals
    y_mean_w = sum(W .* y) / max(sum(W), 1e-12)
    WTSS = sum(W .* (y .- y_mean_w) .^ 2) # weighted total sum of squares
    # OK, so additional (beyond mean-field) variance we do not fit
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

    # Effective Number of Parameters (gamma / d_eff)
    # Tr((X'WX + P)⁻¹ X'WX)
    M_data = X_gf' * W_diag * X_gf
    γ = tr(H \ M_data)

    @printf("fit_LinearEnergyModel(λ_ridge_Δ=%g, λ_smooth=%g): \n EffectiveDoF=γ= %g E_MF=E_1= %8.4f\n", λ_ridge_Δ, λ_smooth, γ, E_MF)

    @printf("  E_l \t RawValue \t\t E_correlation \t Std Error\n")
    for l in 1:min(8, N)
        @printf("  E_%-2d \t %8.4f \t %8.4f \t %8.4f\n", l, E_l[l], (E_l[l] - l * E_l[1]), se_E_l[l])
    end

    @printf("  Linear model Weighted R² = % 2.3f %%\n", 100 * R2_W)

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
        E_k = eval_energy_at_rank(en_model, k, C_k) # extracted efficiently if cached
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

    # initially I dot-added the elements of the reservoir like an idiot
    # Helper function: returns a memory-efficient view of only the filled samples
    valid(x, k) = view(x.reservoir[k], 1:min(x.count[k], length(x.reservoir[k])))
    # Map over all families to create the new merged reservoirs
    merged_res = map(1:a.n_families) do k
        pool = shuffle!(vcat(valid(a, k), valid(b, k)))
        keep = min(length(pool), length(a.reservoir[k]))

        out = zeros(Float64, length(a.reservoir[k]))
        out[1:keep] .= view(pool, 1:keep)
        out # Implicit return for the map
    end

    return DensePermutationFamilyStats(a.N, a.P, a.n_families,
        a.count .+ b.count, a.estimator .+ b.estimator, a.estimator2 .+ b.estimator2, merged_res)
end

# ===================================================================
# Importance Sampling: post-MC debiasing
# ===================================================================

"""
    debiased_empirical_probabilities(stats, bias) -> Vector{Float64}

Recover true sector probabilities from biased MC counts.
Under bias 1/p^α, raw counts satisfy n_k ∝ π_k / p_k^α.
True probabilities: π̂_k = n_k · p_k^α / Σ_j n_j · p_j^α.
"""
function debiased_empirical_probabilities(stats::DensePermutationFamilyStats,
    bias::Union{Nothing,PermutationBias})
    if bias === nothing
        return empirical_probabilities(stats)
    end
    w = zeros(Float64, stats.n_families)
    for k in 1:stats.n_families
        if stats.count[k] > 0
            w[k] = stats.count[k] * exp(bias.α * bias.log_p[k])
        end
    end
    Σ = sum(w)
    Σ == 0 && error("No observations recorded")
    return w ./ Σ
end

"""
    debiased_mc_energy(stats, bias) -> (E, avg_sign)

Fermion energy estimator from biased MC data.
Debiased sector probabilities × raw conditional-mean energies.
"""
function debiased_mc_energy(stats::DensePermutationFamilyStats,
    bias::Union{Nothing,PermutationBias})
    p_hat = debiased_empirical_probabilities(stats, bias)
    N = stats.N

    Etot = 0.0
    Z_sign = 0.0
    for k in 1:stats.n_families
        if stats.count[k] > 0
            E_k = stats.estimator[k] / stats.count[k]

            C_k = C_from_rank(k, N, stats.P)
            n_cycles = sum(C_k)
            σ = iseven(N - n_cycles) ? 1.0 : -1.0

            Etot += σ * p_hat[k] * E_k
            Z_sign += σ * p_hat[k]
        end
    end
    return Etot / Z_sign, Z_sign
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
            observe_permutation_family_reservoir!(acc, C_vec, E_virial)
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
    equil::Int=100_000, steps::Int=1_000_000, measure_every::Int=1,
    lstm_epochs::Int=500, lstm_hidden::Int=64, lstm_embed::Int=16,
    lstm_lr::Float64=3e-3, use_kelbg::Bool=true)
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
    MC_data = permutation_family_mc(sys, params; equil, steps, measure_every)

    n_z = sum(MC_data.count)
    @printf("\n MC Complete! N= %d  r_s= %g θ= %g  n_families= %d  Z-samples=%d\n", N, r_s, θ, MC_data.n_families, n_z)

    # save reservoir of samples
    open("reservoir.dat", "w") do io
        println(io, "# Estimator reservoir")
        for k in 1:MC_data.n_families
            for E in MC_data.reservoir[k]
                if E == 0.0
                    break
                end # if as initialised, no more data; throughs off plots
                @printf(io, "%g ", E)
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
    @printf("E_MC= %.8f σ=  %.8f E_MC/N= %.8f Ha = %.8f Ry (including Yakub-Ronchi Jellium background)\n",
        E_MC + N * E_bg, sigma, (E_MC + N * E_bg) / N, 2 * (E_MC + N * E_bg) / N)

    # OK, now we start doing the weird stuff! What Would Ceperley Do? (WWCD?)
    # Set smoothness to massive to enforce the Feynman/DuBois exchange penalty model
    # I think this is what they used for low T ?
    smoothness = (θ <= 0.5) ? 1e12 : 1e4
    linearEmodel = fit_LinearEnergyModel(LinearEnergyModel, MC_data; λ_smooth=smoothness)
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
    #   - essentially this just tells us that LSTM alone is terrible, KL divergence stays massive
    println("Training LSTM (multiplicity prior)...")
    model_lstm_mult = fit(LSTMPermutationModel, MC_data; prior=model_mult, lstm_kw...)
    # 2. DuBois prior — gets M(λ) + κ-penalty for free
    println("Training LSTM (DuBois prior)...")
    model_lstm_DuBois = fit(LSTMPermutationModel, MC_data; prior=model_dubois, lstm_kw...)
    # 3. LSTM on MAP, so already optim Bayesian Theta-model
    println("Training LSTM (MAP-on-DuBois prior)...")
    model_lstm_MAP = fit(LSTMPermutationModel, MC_data; prior=model_map, lstm_kw...)

    # ---------------------------------------------------------------
    # Extracted Energy LSTM Runs
    # ---------------------------------------------------------------
    println("\nTraining Shallow Energy Head on MAP-on-DuBois LSTM trunk...")
    lstm_energy_shallow = fit_LSTMEnergyResidualModel(model_lstm_MAP, linearEmodel, MC_data; hidden_layers=(), epochs=500)

    println("\nTraining Deep Energy Head on MAP-on-DuBois LSTM trunk...")
    lstm_energy_deep = fit_LSTMEnergyResidualModel(model_lstm_MAP, linearEmodel, MC_data; hidden_layers=(32,), epochs=500)

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
        E_shallow = calculate_reweighted_energy(m, lstm_energy_shallow, MC_data) + (N * E_bg)
        E_deep = calculate_reweighted_energy(m, lstm_energy_deep, MC_data) + (N * E_bg)

        #        @printf("%-30s KL=%8.4f σ=% 8.4f \n\tE_MC/N % 8.4f Ry | E_Lin/N % 8.4f Ry | E_Shal/N % 8.4f Ry | E_Deep/N % 8.4f Ry\n",
        #            label, KL, avgsign, 2 * E / N, 2 * E_lin / N, 2 * E_shallow / N, 2 * E_deep / N)

        @printf("  %-30s KL=%8.4f σ=% 8.4f | E_MC/N % 8.4f Ry | E_Lin/N % 8.4f Ry | E_Shal/N % 8.4f Ry | E_Deep/N % 8.4f Ry\n",
            label, KL, avgsign,
            2 * E / N, 2 * E_lin / N, 2 * E_shallow / N, 2 * E_deep / N)

    end

    @printf("\n# Models:")
    for (label, m) in models
        @printf("%-30s ", label)
        println(m)
    end

    # ===============================================================
    # Importance-sampled MC using fitted model as bias
    # ===============================================================
    bias_α = 0.5  # gentle softening; α=1.0 for full flat-histogram; sort of Wang-Landau
    # experiments showed α=1.0 was terrible - threw MC into the OPPOSITE undersampling, i.e.
    # forced condensation if not present
    biasmodel = model_map
    bias = make_permutation_bias(biasmodel, MC_data; α=bias_α)

    println("\n#### Biased MC (α=$bias_α, model=$biasmodel) ####")
    params_biased = default_worm_params(sys)
    params_biased.bias = bias

    MC_biased = permutation_family_mc(sys, params_biased; equil, steps, measure_every)

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
    @printf("  E_MC_debiased = %.8f  σ=%.4f  E/N= %.8f Ha = %.8f Ry\n",
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

    return MC_data, MC_biased, bias, models
end

if abspath(PROGRAM_FILE) == @__FILE__
    #    MC_and_fit_model(; N=20, θ=1.0, r_s=10.0, steps=10_000_000)
    #    MC_and_fit_model(; N=33, θ=0.125, r_s=1.0) # DuBois Table 1, Weak Coupling / High Density
    #MC_and_fit_model(; N=33, θ=0.125, r_s=10.0)# DuBois Table 1, Strong Coupling / Low Density
    #MC_and_fit_model(; N=33, r_s=1.0, θ=0.125

    # N is magic-number from filling 3D Fermi sphere
    #   So N=1, 7, 19, 33
    Nmagic = 7 
    magicsteps = 1_000_000
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

