# LSTMPermutationModel.jl
# Autoregressive LSTM for permutation sector probability estimation.
# Adapted from SequenceModels.jl (originally from Karpathy nanoGPT -> Flux model-zoo,
#  Dan Stahlke's Flux port, then adapted for peptide prediction in Innisvouls)
#
# The LSTM factorises p(λ) = ∏_t p(λ_t | λ_{1:t-1}) over the integer partition
# sequence λ = [λ_1, λ_2, ..., λ_N].
#
# Exact Autoregressive Prior (Prefix Trie):
# An exponential-family prior (e.g. DuBoisModel) is injected by precomputing the
# exact conditional marginals P(v | prefix) over a Prefix Trie of all p(N) sectors.
# At step t the bias for token v is log(W_child_v / W_parent), where W is the sum
# of prior probabilities over all valid completions.  This guarantees:
#   1. A zero-weight LSTM exactly recovers the global prior distribution.
#   2. Invalid branches (violating descending-order or sum constraint) naturally
#      get -Inf bias (W=0), replacing any manual structural mask.
#
# Token encoding for integer partitions of N:
#   partition value v ∈ {0,1,...,N}  →  token index v+1 ∈ {1,...,N+1}
#   BOS (start) token               →  index N+2
#   vocab_size = N+2

using Flux
using Flux: logsoftmax

"""Autoregressive LSTM model for permutation sector probabilities.
After training, `cached_probs[k]` holds the normalised probability for sector k.

### Keep it old skool : LSTM ###
We create the RNN with two Flux LSTM layers and an output layer of the size of the alphabet.
(The alphabet here is integer partition values 0..N plus a BOS token.)

The frozen `prior` model injects exact autoregressive marginals via a Prefix Trie,
so the LSTM only learns the residual inter-sector correlation."""
struct LSTMPermutationModel{P<:AbstractPermutationModel} <: AbstractPermutationModel
    N::Int
    chain::Chain
    vocab_size::Int
    prior::P
    prior_trie::Dict{Int, Any}
    cached_probs::Vector{Float64}
end

Flux.@layer LSTMPermutationModel trainable=(chain,)

Base.show(io::IO, m::LSTMPermutationModel) = print(io,
    "LSTMPermutationModel(N=$(m.N), vocab=$(m.vocab_size), " *
    "prior=$(m.prior), params=$(sum(length, Flux.trainables(m.chain))))")

# ===================================================================
# Exact Autoregressive Prior — Prefix Trie
# ===================================================================
# Each node is Dict{Int,Any}.  Integer keys 0..N map to child nodes;
# key -1 stores the marginal weight W (sum of prior Q(λ) over all
# valid partitions reachable from this prefix).

"""Build a Prefix Trie from the prior model's global probabilities over all p(N) sectors.
Returns the root node (Dict{Int,Any}) with marginal weights summed bottom-up."""
function _build_prior_trie(prior::AbstractPermutationModel, stats::DensePermutationFamilyStats)
    N = stats.N
    q_all = probabilities(prior, stats)
    root = Dict{Int, Any}()

    for k in 1:stats.n_families
        q_k = q_all[k]
        q_k == 0.0 && continue
        λ = permutation_family_lambda_from_rank(k, N, stats.P)
        node = root
        for t in 1:N
            v = λ[t]
            if !haskey(node, v)
                node[v] = Dict{Int, Any}()
            end
            node = node[v]
        end
        node[-1] = q_k
    end

    _trie_sum_weights!(root)
    return root
end

"""Recursively compute marginal weights bottom-up.  Leaves already have -1 set;
internal nodes get the sum of their children."""
function _trie_sum_weights!(node::Dict{Int, Any})
    haskey(node, -1) && return node[-1]::Float64
    w = 0.0
    for (k, child) in node
        k == -1 && continue
        w += _trie_sum_weights!(child)
    end
    node[-1] = w
    return w
end

# ===================================================================
# Prior bias from Trie traversal
# ===================================================================

"""Exact step-wise prior bias for a batch of known partitions, via Trie lookup.
bias[v+1, t, b] = log(W_child_v / W_parent) at step t for batch element b.
Invalid tokens (absent children) naturally receive -Inf."""
function _build_prior_bias(λ_vals::Matrix{<:Integer}, N::Int, trie::Dict{Int, Any})
    seqlen, batch = size(λ_vals)
    vocab_size = N + 2
    bias = fill(Float32(-Inf), vocab_size, seqlen, batch)

    @inbounds for b in 1:batch
        node = trie
        for t in 1:seqlen
            log_w_parent = log(node[-1]::Float64)
            for (v, child) in node
                v == -1 && continue
                bias[v + 1, t, b] = Float32(log(child[-1]::Float64) - log_w_parent)
            end
            actual = λ_vals[t, b]
            node = node[actual]
        end
    end
    bias
end

# ===================================================================
# Training data from DensePermutationFamilyStats
# ===================================================================

"""Build autoregressive training data from observed partition counts.
Returns (Xs, Ys_onehot, weights, λ_vals) ready for the LSTM.

Xs (input):  [BOS, λ₁+1, ..., λ_{N-1}+1]  — shifted partition as token indices
Ys (target): [λ₁+1, ..., λ_N+1]            — partition as token indices (one-hot encoded)
weights:     count / total_count             — normalised sample weights
λ_vals:      (N, n_obs) raw partition values — needed for prior bias computation"""
function _build_training_data(stats::DensePermutationFamilyStats)
    N = stats.N
    vocab_size = N + 2
    bos = Int32(N + 2)

    obs_idx = Int[]
    obs_wt  = Float32[]
    for k in 1:stats.n_families
        c = stats.count[k]
        if c > 0
            push!(obs_idx, k)
            push!(obs_wt, Float32(c))
        end
    end
    length(obs_idx) == 0 && error("No observed partitions to train on")
    obs_wt ./= sum(obs_wt)
    n_obs = length(obs_idx)

    Xs     = zeros(Int32, N, n_obs)
    λ_vals = zeros(Int32, N, n_obs)
    Ys     = zeros(Int32, N, n_obs)

    for (i, k) in enumerate(obs_idx)
        λ = permutation_family_lambda_from_rank(k, N, stats.P)
        Xs[1, i] = bos
        for t in 2:N
            Xs[t, i] = Int32(λ[t-1] + 1)
        end
        for t in 1:N
            λ_vals[t, i] = Int32(λ[t])
            Ys[t, i]     = Int32(λ[t] + 1)
        end
    end

    # Ys (output) should be one-hot because this is what logitcrossentropy expects.
    Ys_onehot = Flux.onehotbatch(Ys, 1:vocab_size)

    return Xs, Ys_onehot, obs_wt, λ_vals
end

# ===================================================================
# Loss function
# ===================================================================

"""Weighted cross-entropy (NLL) loss in nats.
prior_bias injects the exact step-wise marginal log-probabilities from the Trie.
Invalid structural branches naturally have -Inf bias, acting as a perfect mask."""
function _masked_nll(chain, xs, ys_oh, prior_bias, weights)
    logits = chain(xs) .+ prior_bias
    lsm = logsoftmax(logits, dims=1)
    per_token = -sum(ys_oh .* lsm, dims=1)
    per_seq = dropdims(sum(per_token, dims=2), dims=(1, 2))
    return sum(weights .* per_seq)
end

# ===================================================================
# Inference: evaluate log p(λ) for all p(N) sectors
# ===================================================================

"""Compute normalised probability for every permutation family, batched for efficiency.
The exact autoregressive prior bias is traversed on the fly from the Trie."""
function _evaluate_all_sectors(chain, trie::Dict{Int, Any},
                               stats::DensePermutationFamilyStats; batch_size::Int=512)
    N = stats.N
    n_fam = stats.n_families
    vocab_size = N + 2
    bos = Int32(N + 2)
    log_probs = zeros(Float64, n_fam)

    for bs in 1:batch_size:n_fam
        be = min(bs + batch_size - 1, n_fam)
        B = be - bs + 1

        xs     = zeros(Int32, N, B)
        λ_vals = zeros(Int, N, B)
        for (i, k) in enumerate(bs:be)
            λ = permutation_family_lambda_from_rank(k, N, stats.P)
            λ_vals[:, i] .= λ
            xs[1, i] = bos
            for t in 2:N
                xs[t, i] = Int32(λ[t-1] + 1)
            end
        end

        Flux.reset!(chain)
        logits = Float64.(chain(xs))

        @inbounds for i in 1:B
            lp = 0.0
            node = trie
            for t in 1:N
                log_w_parent = log(node[-1]::Float64)
                logit_t = fill(-Inf, vocab_size)
                for (v, child) in node
                    v == -1 && continue
                    z_prior = log(child[-1]::Float64) - log_w_parent
                    logit_t[v + 1] = logits[v + 1, t, i] + z_prior
                end
                mx = maximum(logit_t)
                lse = mx + log(sum(exp.(logit_t .- mx)))

                actual = λ_vals[t, i]
                lp += logit_t[actual + 1] - lse

                node = node[actual]
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

"""
    fit(LSTMPermutationModel, stats; prior=MultiplicityModel(zeros(N)), n_embed=16, ...)

Train an autoregressive LSTM on the observed partition histogram.
The frozen `prior` model provides exact autoregressive marginals via a Prefix Trie:
at each step, log(P_prior(v | prefix)) is added to the LSTM logit as a fixed bias.
A zero-weight LSTM exactly recovers the prior distribution.  Invalid structural
branches naturally get -Inf bias, replacing any manual mask.
After training, evaluates p(λ) for all p(N) sectors and caches the result."""
function fit(::Type{LSTMPermutationModel}, stats::DensePermutationFamilyStats;
             prior::AbstractPermutationModel=MultiplicityModel(zeros(stats.N)),
             n_embed::Int=16, n_hidden::Int=64, epochs::Int=100, lr::Float64=1e-3)
    N = stats.N
    vocab_size = N + 2

    trie = _build_prior_trie(prior, stats)
    println("  Trie built: $(length(keys(trie))-1) root children, W_root=$(round(trie[-1], digits=6))")

    # We create the RNN with two Flux LSTM layers and an output layer of the size of the alphabet:
    chain = Chain(
        Embedding(vocab_size => n_embed),
        LSTM(n_embed => n_hidden),
        LSTM(n_hidden => n_hidden),
        Dense(n_hidden => vocab_size))

    Xs, Ys_oh, weights, λ_vals = _build_training_data(stats)
    prior_bias = _build_prior_bias(λ_vals, N, trie)

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
            println("  LSTM epoch $epoch/$epochs  NLL = $(round(Float64(l), digits=4))")
        end
    end

    Flux.testmode!(chain)
    cached = _evaluate_all_sectors(chain, trie, stats)

    LSTMPermutationModel(N, chain, vocab_size, prior, trie, cached)
end

"""Model distribution Q(k) for all p(N) sectors (returns cached probabilities from training)."""
function probabilities(model::LSTMPermutationModel, stats::DensePermutationFamilyStats)
    model.cached_probs
end
