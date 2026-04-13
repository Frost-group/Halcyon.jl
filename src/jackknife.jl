"""
Holds the Jackknife reduced-bias estimators and standard errors for permutation statistics.

Fields:
- `N::Int`: Particle count
- `n_families::Int`: Number of permutation families
- `P::Matrix{Int}`: Partition table
- `total_Z::Int`: Total samples
- `p_k_jack::Vector{Float64}`: Jackknifed probabilities of each family
- `p_k_se::Vector{Float64}`: Standard error of `p_k`
- `E_k_jack::Vector{Float64}`: Jackknifed conditional energies `E_k`
- `E_k_se::Vector{Float64}`: Standard error of `E_k`
- `pE_k_jack::Vector{Float64}`: Jackknifed sector expected value `pE_k = p_k * E_k`
- `pE_k_se::Vector{Float64}`: Standard error of `p_k * E_k`
"""
Base.@kwdef struct JackknifePermutationStats
    N::Int
    n_families::Int
    P::Matrix{Int}

    total_Z::Int

    p_k_jack::Vector{Float64}
    p_k_se::Vector{Float64}

    E_k_jack::Vector{Float64}
    E_k_se::Vector{Float64}

    pE_k_jack::Vector{Float64}
    pE_k_se::Vector{Float64}
end

# Dump actually quite a nice default view
Base.show(io::IO, obj::JackknifePermutationStats) = dump(io, obj; maxdepth=1)

"""
    jackknife_statistics(parts::Vector{DensePermutationFamilyStats}) -> JackknifePermutationStats

Calculates Jackknife reduced-bias estimators and standard errors by treating each element of `parts`
as an independent block (e.g. from threaded MC chains).
"""
function jackknife_statistics(parts::Vector{DensePermutationFamilyStats})
    n_chains = length(parts)
    n_chains > 1 || throw(ArgumentError("Jackknife requires n_chains > 1"))

    (; N, n_families, P) = parts[1]

    # Pre-calculate full totals across all chains
    full_counts = zeros(Int, n_families)
    full_sums = zeros(Float64, n_families)
    Z_parts = zeros(Int, n_chains) # Cache Z per individual chain

    for (j, p) in enumerate(parts)
        full_counts .+= p.count
        full_sums .+= p.estimator
        Z_parts[j] = sum(p.count)
    end

    Z_full = sum(full_counts)
    Z_full > 0 || error("No observations recorded in parts")

    # Output arrays
    p_k_jack, p_k_se = zeros(n_families), zeros(n_families)
    E_k_jack, E_k_se = zeros(n_families), zeros(n_families)
    pE_k_jack, pE_k_se = zeros(n_families), zeros(n_families)

    # Process each permutation family individually
    for k in 1:n_families
        full_p = full_counts[k] / Z_full
        full_E = full_counts[k] > 0 ? full_sums[k] / full_counts[k] : 0.0
        full_pE = full_sums[k] / Z_full

        p_pseudo = zeros(Float64, n_chains)
        E_pseudo = zeros(Float64, n_chains)
        pE_pseudo = zeros(Float64, n_chains)

        valid_E_chains = 0

        # Leave-One-Out block updates
        for (j, p) in enumerate(parts)
            loo_count = full_counts[k] - p.count[k]
            loo_sum = full_sums[k] - p.estimator[k]
            loo_Z = Z_full - Z_parts[j]

            loo_p = loo_Z > 0 ? loo_count / loo_Z : 0.0
            loo_pE = loo_Z > 0 ? loo_sum / loo_Z : 0.0

            # Compute jackknife pseudovalues
            p_pseudo[j] = n_chains * full_p - (n_chains - 1) * loo_p
            pE_pseudo[j] = n_chains * full_pE - (n_chains - 1) * loo_pE

            if loo_count > 0
                loo_E = loo_sum / loo_count
                E_pseudo[j] = n_chains * full_E - (n_chains - 1) * loo_E
                valid_E_chains += 1
            else
                # Fallback if LOO effectively removes all counts for a family
                E_pseudo[j] = full_E
            end
        end

        # calculate Jackknife statistics
        p_k_jack[k] = sum(p_pseudo) / n_chains
        var_p = sum(x -> (x - p_k_jack[k])^2, p_pseudo) / (n_chains - 1)
        p_k_se[k] = sqrt(max(0.0, var_p / n_chains))

        pE_k_jack[k] = sum(pE_pseudo) / n_chains
        var_pE = sum(x -> (x - pE_k_jack[k])^2, pE_pseudo) / (n_chains - 1)
        pE_k_se[k] = sqrt(max(0.0, var_pE / n_chains))

        if valid_E_chains > 1
            E_k_jack[k] = sum(E_pseudo) / n_chains
            var_E = sum(x -> (x - E_k_jack[k])^2, E_pseudo) / (n_chains - 1)
            E_k_se[k] = sqrt(max(0.0, var_E / n_chains))
        else
            E_k_jack[k] = full_E
            E_k_se[k] = 0.0
        end
    end

    # ToDo: make this ootional?
    @printf("Jackknife statistics calculated for N=%d, n_families=%d, Z=%d\n", N, n_families, Z_full)
    if n_families < 50
        @printf("p_k_jack: %s\n", p_k_jack)
        @printf("p_k_se: %s\n", p_k_se)
        @printf("E_k_jack: %s\n", E_k_jack)
        @printf("E_k_se: %s\n", E_k_se)
        @printf("pE_k_jack: %s\n", pE_k_jack)
        @printf("pE_k_se: %s\n", pE_k_se)
    end

    @printf("Jacknife Bosonic E: %f +/- %f\n", sum(pE_k_jack), sum(pE_k_se))
    @printf("Naive Bosonic E: %f\n", sum(full_sums) / Z_full)

    return JackknifePermutationStats(;
        N, n_families, P, total_Z=Z_full,
        p_k_jack, p_k_se, E_k_jack, E_k_se, pE_k_jack, pE_k_se
    )
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


# ===================================================================
# Importance Sampling: PermutationBias constructor (object in types.jl)
# ===================================================================

"""
    make_permutation_bias(model, stats; α=1.0) -> PermutationBias

Construct a bias table from a fitted AbstractPermutationModel.
"""
function make_permutation_bias(model::AbstractPermutationModel,
    stats::DensePermutationFamilyStats; α::Float64=1.0)
    q = probabilities(model, stats)
    log_p = [q[k] > 0 ? log(q[k]) : -Inf for k in 1:stats.n_families]
    PermutationBias(stats.P, log_p, α)
end

"""
    make_variance_optimised_bias(jk::JackknifePermutationStats; α=1.0) -> PermutationBias

Construct a bias vector targeting sectors that cause the most variance in the global energy estimator.
The variance of the global estimator is sum Var(pE_k), .'. target probabilities π_k ∝ SE(pE_k).
Uses softmax-like exponentiation with parameter α to shape the strength.
"""
function jackknife_error_guide(bias::PermutationBias, jk::JackknifePermutationStats; α::Float64=1.0)
    # conditional broadcasting (!)
    q = @. ifelse(jk.pE_k_se > 0.0, jk.pE_k_se, 0.0)
    # OK, experiments thing this is too harsh: only observe first sector after this (!)
    # I need to think a bit harder about turning the jacknife standard error into a probability to revisit.
    q = q ./ sum(q) # normalise probability
    min = 1e-4 * 1/jk.n_families # minimum intent: 1e-4 * natural occurance. Perhaps use number of microstates / degerenacy?
    log_p = [q[k] > min ? -log(q[k]) : -log(min) for k in 1:jk.n_families]
    @printf("jacknife_error_guide: q= %s log_p= %s\n", q, log_p)
    log_p .+= bias.log_p # mix in previous guidance
    @printf("jacknife_error_guide, combined: log_p= %s\n", log_p)
    PermutationBias(jk.P, log_p, α)
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
