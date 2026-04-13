# Extracted from MC-ML-development.jl; 13th April 2026
# Everything relating to Energy models

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
    h_lstm = hidden_seq[:, end, :] # (n_hidden, n_obs) - the terminal representation
    #    n_hidden_in = size(h_final, 1) # dimension of hidden representation to be fed into our NN

    # Direct permutation features, to learn on
    Cmat = Halcyon.cycle_count_matrix(stats) # Always the cowboy, never the cow
    visited_mask = stats.count .> 0
    Cmat_visited = Float32.(Cmat[visited_mask, :]')

    h_final = vcat(h_lstm, Cmat_visited ./ N) # Shape: (n_hidden + N, n_obs)
    # scale CMAT to [0..1] so similar scale to LSTM. But does this matter?
    # Perhaps switch to Lambda repr?

    n_hidden_in = size(h_final, 1)

    @printf("Built features: size(h_lstm,1): %d size(Cmat_visited): %d n_hidden_in: %d\n",
        size(h_lstm, 1), size(Cmat_visited, 1), n_hidden_in)

    # 2. Build explicit numeric WLS (Weighted Least Squares) residual targets
    n_obs = count(c -> c > 0, stats.count)
    delta_E = zeros(Float32, 1, n_obs)
    E_raw = zeros(Float32, n_obs)
    W_arr = zeros(Float32, n_obs)

    # 3. Global (g_) variance for a Bayes prior on the sample variance
    # Filter out unvisited sectors (with a boolean mask)
    g_visited = stats.count .> 0
    g_c = stats.count[g_visited]
    g_est = stats.estimator[g_visited]
    g_est2 = stats.estimator2[g_visited]
    # Total variance across all samples 
    g_N = sum(g_c)
    g_mean = sum(g_est) / g_N
    g_var = max((sum(g_est2) / g_N) - g_mean^2, 1e-12)  # usual 1e-12 to stop underflow

    # bit painful this really; not sure if I should fill with dummy values or whatever... once we get N>100, we'll need to be more clever generally
    # do this with the vectorised magic g_visited like above?
    idx = 1
    for k in 1:stats.n_families
        c_k = stats.count[k]
        if c_k > 0
            E_mean = stats.estimator[k] / c_k # sample (permutation family) mean
            E_var = max.((stats.estimator2[k] ./ c_k) .- E_mean .^ 2, 0.0) # sample var

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
            #            @printf("LSTM Var smooth: Cycle= %d c_k= %d g_var= %g E_var= %g var_smooth= %g W_arr[%d]= %g\n",
            #                    k,c_k,g_var,E_var,var_smooth,idx,W_arr[idx])
            idx += 1
        end
    end

    W_arr ./= sum(W_arr) # rescale for stable gradient magnitudes

    # 3. Construct purely unregularised scalar regression head
    layers = []
    in_dim = n_hidden_in
    for h in hidden_layers
        if h > 0
            push!(layers, Dense(in_dim => h, gelu))
            # gelu as apparently nicer for regression?

            # mild dropout on wider hidden layers to regularise
            #            if h >= 64
            #                push!(layers, Dropout(0.1))
            #            end
            # Jarv ~9-4-26 seems to stop it converging nicely c.f. shallow network. So turn
            # off for now? I mean, its not exactly a bit data regime, trying to fit the
            # permutation data... perhaps when N is large and we are not visiting all
            # sectors.

            in_dim = h
        end
    end
    # Zero-init the Final Projection, so we start by following the Linear model / prior
    #    push!(layers, Dense(in_dim => 1, init=Flux.zeros))
    push!(layers, Dense(in_dim => 1))


    head = Chain(layers...)
    #opt_state = Flux.setup(Adam(lr), head)
    # 'Weight decay' is just L2 norm on weights (paramters of fit) penalising large vlaues
    # called weight-decy 'cause you add it as a constant to the gradient, so is noiseless
    # c.f. adding directly to the loss and back prop
    opt_state = Flux.setup(OptimiserChain(WeightDecay(1e-3), Adam(lr)), head)
    #  Think this might be messing up the fits? Disabled for now. 
    #     Actually that might have been galaxy-brain-Jarv weighting the Cmat matrix FOR
    #     TRAINING, but not then doing the same at EVAL. 

    # Weighted Least Squares Loss over the empirical delta batch
    wls_loss(m, x, y, w) = sum(w' .* (m(x) .- y) .^ 2)

    # Trying to control affect of outliers. 
    # Huber Loss (L2 for small values, L1 for large)
    # delta (δ) controls where the loss transitions from quadratic to linear.
    # δ = 1.0f0 means mismatch < 1 Hartree are fit with MSE, mismatch > 1 Hartree are fit linearly.
    function huber_loss(m, x, y, w; δ=1.0)
        res = m(x) .- y
        abs_res = abs.(res)
        # 0.5 * x^2 if |x| <= δ, else δ * |x| - 0.5 * δ^2
        losses = ifelse.(abs_res .<= δ, 0.5 .* abs_res .^ 2, δ .* (abs_res .- 0.5 .* δ^2))
        return sum(w' .* losses)
    end

    println("  Training LSTM Energy Head: ", head)
    for epoch in 1:epochs
        grad = Flux.gradient(head) do m
            huber_loss(m, h_final, delta_E, W_arr)
        end
        Flux.update!(opt_state, head, grad[1])

        if epoch % 100 == 0 || epoch == 1 # trains super fast as so simple
            mywls = wls_loss(head, h_final, delta_E, W_arr)
            myhuber = huber_loss(head, h_final, delta_E, W_arr)
            @printf("  LSTM Energy Head (Layers %s) epoch %d/%d  WLS-MSE = %.2e Huber-loss= %.2e\n", string(hidden_layers),
                epoch, epochs, Float64(mywls), Float64(myhuber))
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

    h_f = vcat(trunk(xs)[:, end, :], Cmat' ./ N) # I hate the way my life turned out 
    # (hidden, n_families)
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


