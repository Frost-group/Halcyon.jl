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
    @printf("p_k_jack: %s\n", p_k_jack)
    @printf("p_k_se: %s\n", p_k_se)
    @printf("E_k_jack: %s\n", E_k_jack)
    @printf("E_k_se: %s\n", E_k_se)
    @printf("pE_k_jack: %s\n", pE_k_jack)
    @printf("pE_k_se: %s\n", pE_k_se)
    @printf("Jacknife Bosonic E: %f +/- %f\n", sum(pE_k_jack), sum(pE_k_se))
    @printf("Naive Bosonic E: %f\n", sum(full_sums) / Z_full)

    return JackknifePermutationStats(;
        N, n_families, P, total_Z=Z_full,
        p_k_jack, p_k_se, E_k_jack, E_k_se, pE_k_jack, pE_k_se
    )
end

"""
    make_variance_optimised_bias(jk::JackknifePermutationStats; α=1.0) -> PermutationBias

Construct a bias vector targeting sectors that cause the most variance in the global energy estimator.
The variance of the global estimator is sum Var(pE_k), .'. target probabilities π_k ∝ SE(pE_k). 
Uses softmax-like exponentiation with parameter α to shape the strength.
"""
function make_variance_optimised_bias(jk::JackknifePermutationStats; α::Float64=1.0)
    # conditional broadcasting (!)
    target = @. ifelse(jk.pE_k_se > 0.0, jk.pE_k_se^α, 0.0)

    tot_energy_error = sum(target)
    if tot_energy_error > 0.0
        target ./= tot_energy_error
    else
        # Fallback to flat if no variance measured 
        # (e.g. 0-step simulation or flat estimators)
        #  or should we error here? 
        target .= 1.0 / jk.n_families
    end

    # PermutationBias takes log probs; bit of an overloading, perhaps just create object here?
    log_p = @. ifelse(target > 0.0, log(target), -1e6)

    # CODE NOT TESTED YET

    return PermutationBias(jk.P, log_p, α)
end
