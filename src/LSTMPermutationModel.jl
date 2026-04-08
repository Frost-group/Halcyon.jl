# LSTMPermutationModel.jl
# Autoregressive LSTM for permutation sector probability estimation. 
# Now with more C-vectors!
# 
# The LSTM factorises p(C) = ∏_k p(C_k | M) over the sequence of cycle counts
# C = [C_N, C_{N-1}, ..., C_1].
#
# Rebuilt this way (use to be Lambda-sequence based), as we can then get an 
# exact autoregressive orior from the exponential-family fit. 
# We precompute the exact conditional marginals P(C_k | M) as an NxN table, 
# using dynamic programming (finally! I knew those information olympiads would eventually
# come in use...). 
#
# At step k, the bias for predicting count c is log(P_prior(C_k=c)).
# This guarantees:
#   1. The initial zero-weight LSTM exactly recovers the exponential fit.
#   2. Invalid branch choices naturally get -Inf bias, eliminating any mask faff. 
#
# Token encoding:
#   count value c ∈ {0, 1, ..., N}  →  token index c+1 ∈ {1, ..., N+1}
#   BOS (start) token             →  index N+2

using Flux
using Flux: logsoftmax
using Printf

### Keep it old skool : LSTM ###

"""Autoregressive LSTM model for permutation sector probabilities.
After training, `cached_probs[k]` holds the normalised probability for sector k.

By predicting the counts C_k descendingly from k=N down to 1, we eliminate
the sorting constraints and allow exact O(N^2) DP marginalisation."""
struct LSTMPermutationModel{P<:AbstractPermutationModel} <: AbstractPermutationModel
    N::Int
    chain::Chain
    vocab_size::Int
    prior::P
    log_dp_table::Matrix{Float64}
    log_f::Vector{Float64}
    cached_probs::Vector{Float64}
end

Flux.@layer LSTMPermutationModel trainable=(chain,)

Base.show(io::IO, m::LSTMPermutationModel) = print(io,
    "LSTMPermutationModel(N=$(m.N), vocab=$(m.vocab_size), " *
    "prior=$(m.prior), params=$(sum(length, Flux.trainables(m.chain))))")

function log_factorial(n::Int)
    n <= 1 && return 0.0
    return sum(log.(1:n))
end

# ===================================================================
# Exact Autoregressive Prior — O(N^2) Dynamically-programmed Table
# ===================================================================

"""Build the 2D Canonical DP table Z[k, M] in the log-domain.
Z[k, M] is the normalisation factor for distributing mass M strictly amongst
cycles of lengths ≤ k."""
function _build_log_dp_table(prior::AbstractPermutationModel, N::Int)
    log_f = zeros(Float64, N)
    for k in 1:N
        # Topological multiplicity penalty + prior penalty
        log_f[k] = prior.θ[k] - log(k)
    end
    
    logZ = fill(-Inf, N, N + 1) # k = 1..N, M = 0..N
    
    # Base case k=1: Z[1, M] = f_1^M / M!
    for M in 0:N
        logZ[1, M + 1] = M * log_f[1] - log_factorial(M)
    end
    
    # Dynamic programming recursion bottom-up
    for k in 2:N
        for M in 0:N
            max_c = floor(Int, M / k)
            terms = fill(-Inf, max_c + 1)
            for c in 0:max_c
                terms[c + 1] = c * log_f[k] - log_factorial(c) + logZ[k - 1, M - c * k + 1]
            end # a thing of beauty and a joy forever
            
            # logsumexp trick for numerical stability. Slide rule for the win.
            mx = maximum(terms)
            if mx != -Inf
                logZ[k, M + 1] = mx + log(sum(exp.(terms .- mx)))
            end
        end
    end
    
    return logZ, log_f
end

# ===================================================================
# Prior bias lookup
# ===================================================================

"""Exact step-wise prior logit bias across a batch using the DP table."""
function _build_dp_prior_bias(M_vals::Matrix{Int}, N::Int, log_dp_table::Matrix{Float64}, log_f::Vector{Float64})
    seqlen, batch = size(M_vals)
    vocab_size = N + 2
    bias = fill(Float32(-Inf), vocab_size, seqlen, batch)

    @inbounds for b in 1:batch
        for t in 1:seqlen
            k = N - t + 1
            M = M_vals[t, b]
            
            log_Z_parent = log_dp_table[k, M + 1]
            max_c = floor(Int, M / k)
            
            for c in 0:max_c
                if k == 1
                    if c != M
                        continue
                    end
                    term = c * log_f[1] - log_factorial(c)
                    bias[c + 1, t, b] = Float32(term - log_Z_parent)
                else
                    term = c * log_f[k] - log_factorial(c) + log_dp_table[k - 1, M - c * k + 1]
                    bias[c + 1, t, b] = Float32(term - log_Z_parent)
                end
            end
        end
    end
    bias
end

# ===================================================================
# Training data from DensePermutationFamilyStats
# ===================================================================

"""Build AR training data for the sequence of cycle counts. In the C-vector repr, each
training vector is N long. So this is a nice dense block of memory.""" 
function _build_training_data(stats::DensePermutationFamilyStats)
    N = stats.N
    vocab_size = N + 2
    bos = Int32(N + 2) # prob makes no difference on CPU?

    obs_idx = Int[]
    obs_wt  = Float32[]
    for i in 1:stats.n_families
        c = stats.count[i]
        if c > 0
            push!(obs_idx, i)
            push!(obs_wt, Float32(c))
        end
    end
    length(obs_idx) == 0 && error("No observed partitions to train on")
    obs_wt ./= sum(obs_wt)
    n_obs = length(obs_idx)

    Xs     = zeros(Int32, N, n_obs)
    M_vals = zeros(Int, N, n_obs)
    Ys     = zeros(Int32, N, n_obs)

    Cmat = cycle_count_matrix(stats)
    
    for (i, idx) in enumerate(obs_idx)
        C_counts = view(Cmat, idx, :)
        
        M_rem = N
        for t in 1:N
            k = N - t + 1
            M_vals[t, i] = M_rem
            c = C_counts[k]
            
            if t == 1
                Xs[t, i] = bos
            else
                k_prev = N - (t - 1) + 1
                Xs[t, i] = Int32(C_counts[k_prev] + 1)
            end
            
            Ys[t, i] = Int32(c + 1)
            M_rem -= c * k
        end
    end

    Ys_onehot = Flux.onehotbatch(Ys, 1:vocab_size)
    return Xs, Ys_onehot, obs_wt, M_vals
end

# ===================================================================
# Loss function
# ===================================================================

function _masked_nll(chain, xs, ys_oh, prior_bias, weights)
    logits = chain(xs) .+ prior_bias
    lsm = logsoftmax(logits, dims=1)
    per_token = -sum(ys_oh .* lsm, dims=1)
    per_seq = dropdims(sum(per_token, dims=2), dims=(1, 2))
    return sum(weights .* per_seq)
end

# ===================================================================
# Inference: evaluate log p(C) for all p(N) sectors
# ===================================================================

function _evaluate_all_sectors(chain, log_dp_table::Matrix{Float64}, log_f::Vector{Float64},
                               stats::DensePermutationFamilyStats; batch_size::Int=512)
    N = stats.N
    n_fam = stats.n_families
    vocab_size = N + 2
    bos = Int32(N + 2)
    log_probs = zeros(Float64, n_fam)

    Cmat = cycle_count_matrix(stats)

    for bs in 1:batch_size:n_fam
        be = min(bs + batch_size - 1, n_fam)
        B = be - bs + 1

        xs     = zeros(Int32, N, B)
        c_vals = zeros(Int, N, B)
        M_vals = zeros(Int, N, B)

        for (i, idx) in enumerate(bs:be)
            C_counts = view(Cmat, idx, :)
            M_rem = N
            for t in 1:N
                k = N - t + 1
                M_vals[t, i] = M_rem
                c = C_counts[k]
                c_vals[t, i] = c
                
                if t == 1
                    xs[t, i] = bos
                else
                    k_prev = N - (t - 1) + 1
                    xs[t, i] = Int32(C_counts[k_prev] + 1)
                end
                M_rem -= c * k
            end
        end

        Flux.reset!(chain)
        logits = Float64.(chain(xs))

        @inbounds for i in 1:B
            lp = 0.0
            for t in 1:N
                k = N - t + 1
                M = M_vals[t, i]
                log_Z_parent = log_dp_table[k, M + 1]
                max_c = floor(Int, M / k)
                
                logit_t = fill(-Inf, vocab_size)
                for c in 0:max_c
                    if k == 1
                        if c != M
                            continue
                        end
                        term = c * log_f[1] - log_factorial(c)
                        z_prior = term - log_Z_parent
                        logit_t[c + 1] = logits[c + 1, t, i] + z_prior
                    else
                        term = c * log_f[k] - log_factorial(c) + log_dp_table[k - 1, M - c * k + 1]
                        z_prior = term - log_Z_parent
                        logit_t[c + 1] = logits[c + 1, t, i] + z_prior
                    end
                end
                
                mx = maximum(logit_t)
                lse = mx + log(sum(exp.(logit_t .- mx)))

                actual_c = c_vals[t, i]
                lp += logit_t[actual_c + 1] - lse
            end
            log_probs[bs + i - 1] = lp
        end
    end

    mx = maximum(log_probs)
    probs = exp.(log_probs .- mx)
    probs ./= sum(probs)
    probs
end

# ===================================================================
# fit / probabilities API
# ===================================================================

function fit(::Type{LSTMPermutationModel}, stats::DensePermutationFamilyStats;
             prior::AbstractPermutationModel=MultiplicityModel(zeros(stats.N)),
             n_embed::Int=16, n_hidden::Int=64, n_layers::Int=2,
             epochs::Int=100, lr::Float64=1e-3)
    N = stats.N
    vocab_size = N + 2

    # Precompute Exact Autoregressive Prior DP Table
    log_dp_table, log_f = _build_log_dp_table(prior, N)
    println("  DP Table built. log Z[N, N] = $(round(log_dp_table[N, N+1], digits=6))")

    lstm_stack = Any[LSTM(n_embed => n_hidden)]
    for _ in 2:n_layers
        push!(lstm_stack, LSTM(n_hidden => n_hidden))
    end
    chain = Chain(
        Embedding(vocab_size => n_embed),
        lstm_stack...,
        Dense(n_hidden => vocab_size))

    Xs, Ys_oh, weights, M_vals = _build_training_data(stats)
    prior_bias = _build_dp_prior_bias(M_vals, N, log_dp_table, log_f)

    opt_state = Flux.setup(Adam(lr), chain)

    for epoch in 1:epochs
        Flux.reset!(chain)
        grad = Flux.gradient(chain) do m
            _masked_nll(m, Xs, Ys_oh, prior_bias, weights)
        end
        Flux.update!(opt_state, chain, grad[1])

        if epoch % 20 == 0 || epoch == 1
            Flux.reset!(chain)
            l = _masked_nll(chain, Xs, Ys_oh, prior_bias, weights)
            @printf(" %d/%d NLL= %8.4f ",epoch,epochs,Float64(l)) # more compact w/o line breaks
        end
    end

    Flux.testmode!(chain)
    cached = _evaluate_all_sectors(chain, log_dp_table, log_f, stats)

    LSTMPermutationModel(N, chain, vocab_size, prior, log_dp_table, log_f, cached)
end

function probabilities(model::LSTMPermutationModel, stats::DensePermutationFamilyStats)
    model.cached_probs
end

