# Ideal spin-polarised Fermi gas: dense permutation-family histogram (conjugacy-class index).

using Halcyon
using Printf
using ProgressMeter
using Optim
using LinearAlgebra: dot

# Agreement with Dornheim 2019 Permutation setup code
"""Wigner–Seitz density n = 3/(4π r_s³) in 3D (atomic units)."""
ueg_density_3d(r_s::Float64) = 3.0 / (4π * r_s^3)

"""Cubic box side L for N electrons at given r_s (continuum UEG density)."""
function ueg_box_length(N::Int, r_s::Float64)
    n = ueg_density_3d(r_s)
    return (N / n)^(1 / 3)
end
"""Fermi wavevector for fully spin-polarised 3D gas: N = V k_F³/(6π²)."""
fermi_wavenumber_polarised(n::Float64) = (6π^2 * n)^(1 / 3)

"""
    ueg_theta_parameters(; N, θ, r_s, λ=0.5)

Degeneracy temperature θ = k_B T/E_F (k_B=1), inverse temperature β = 1/(θ E_F).
Returns `(L, β, E_F, n, k_F)` in atomic units (ℏ=m_e=1 as in the paper).
"""
function ueg_theta_parameters(; N::Int, θ::Float64, r_s::Float64, λ::Float64=0.5)
    L = ueg_box_length(N, r_s)
    n = N / L^3
    kF = fermi_wavenumber_polarised(n)
    EF = λ * kF^2
    β = 1.0 / (θ * EF)
    return (; L, β, E_F=EF, n, k_F=kF)
end

"""
    make_periodic_fermion_system(; M, N, β, L, λ=0.5, pair=NullPairPotential())

Free boundary: `V = HarmonicPotential(k=0)` (no trap). Fermi statistics for sign.
"""
function make_periodic_fermion_system(; M::Int, N::Int, β::Float64, L::Float64,
                                    λ::Float64=0.5, pair::PairPotential=NullPairPotential())
    System(M, N; D=3, β=β, λ=λ, L=L, V=HarmonicPotential(k=0.0), U=pair, statistics=Fermions)
end

default_worm_params(sys::System; C::Float64=1.0) =
    WormParams(C=C, j_max=sys.M ÷ 2, r_max=sys.L / 2)

function run_family_histogram(; N::Int=33, θ::Float64=0.5, r_s::Float64=2.0, M::Int=100,
                             l_fixed::Int=1, equil::Int=100_000, steps::Int=2_000_000,
                             measure_every::Int=5)
    λħ = 0.5
    (; L, β) = ueg_theta_parameters(; N, θ, r_s, λ=λħ)
    sys = make_periodic_fermion_system(; M, N, β, L, λ=λħ)
    params = default_worm_params(sys)
    cfg = WormConfiguration(sys)

    for _ in 1:equil
        worm_step!(cfg, sys, params)
    end

    PermFamHisto = DensePermutationFamilyStats(N)
    println("Allocated Dense PermutationFamily N=$(N) Permutation families: $(PermFamHisto.n_families)  Size(bytes): $(Base.summarysize(PermFamHisto)) Size(GB): $(Base.summarysize(PermFamHisto) / 1024^3)")
    
    acc_pl = zeros(Float64, N)
    acc_pcf_col = zeros(Float64, N)
    n_z = 0

    @showprogress desc="MC:$(steps/1E6)M" for t in 1:steps
        worm_step!(cfg, sys, params)
        if t % measure_every == 0 && cfg.sector == Z_SECTOR
            λ_vec = permutation_family_lambda(cfg)
            W = permutation_pcf_weights(cfg)
            observe_permutation_family!(PermFamHisto, λ_vec, 0.0)

            w_pl = permutation_pl_weights(cfg)
            acc_pl .+= w_pl
            acc_pcf_col .+= @view W[l_fixed, :]
            n_z += 1
        end
    end
    n_z == 0 && error("No Z-sector samples")

    P_mc = acc_pl ./ n_z
    col = acc_pcf_col ./ n_z
    Pu_col = [P_mc[l_fixed] * P_mc[k] for k in 1:N]
    n_tot = sum(PermFamHisto.count)

    println(@sprintf("# Ideal fermions  N=%d  θ=%g  p(N)=%d  Z-samples=%d  (l_fixed=%d)\n",
                     N, θ, PermFamHisto.n_families, n_z, l_fixed))

    hits = Tuple{Int,Int}[]
    for k in 1:PermFamHisto.n_families
        c = PermFamHisto.count[k]
        c > 0 && push!(hits, (k, c))
    end
    sort!(hits; by=x -> (-x[2], x[1]))

# I must not fear the overfit.
# Fear is the gradient-killer.

    # --- Shared setup: log-multiplicity and cycle-count matrix ---
    logM = zeros(Float64, PermFamHisto.n_families)
    C_matrix = zeros(Int, PermFamHisto.n_families, N)
    for k in 1:PermFamHisto.n_families
        λk = permutation_family_lambda_from_rank(k, N, PermFamHisto.P)
        C_k = C_permutation_sector(λk)
        C_matrix[k, :] .= C_k
        logM[k] = sum(log.(1:N))
        for l in 1:N
            C_k[l] > 0 && (logM[k] -= sum(log.(1:C_k[l])) + C_k[l] * log(l))
        end
    end
    empirical_p = Float64.(PermFamHisto.count) ./ n_tot

    # helpers
    function normalised_q(theta)
        lu = logM .+ C_matrix * theta
        mx = maximum(lu)
        exp.(lu .- mx .- log(sum(exp.(lu .- mx))))
    end
    function kl(p, q)
        s = 0.0
        for i in eachindex(p)
            p[i] > 0 && (s += p[i] * log(p[i] / q[i]))
        end
        s
    end

    # --- (0) Infinite-temperature limit (degeneracy only) ---
    # Q(k) ∝ M_k. 
    # Without quantum exchange penalties (β→0), sectors appear purely by their
    #  combinatorial degeneracy in S_N.
    q_mult = normalised_q(zeros(N))
    @printf("KL  Infinite-T (degeneracy-only)    = %.5f nats\n", kl(empirical_p, q_mult))
    println("  Q_mult: ", zeros(N))

    # --- (1) Phenomenological one-parameter extension (DuBois-like) ---
    # Penalise non-trivial exchanges: θ_l = -κ(l-1) with θ₁=0.
    # The sector log-probability is log M_k - κ(N-K), where K = ∑_l C_l is the number of cycles.
    # (N-K) is the minimum number of transpositions required to form the permutation from the identity.
    # We fit the scalar pair-exchange penalty κ by minimising the negative log-likelihood.
    n_minus_K = Float64[N - sum(C_matrix[k, :]) for k in 1:PermFamHisto.n_families]
    
    # f_nll: Loss function (negative log-likelihood) in nats.
    function nll_p2(κ_vec)
        κ = κ_vec[1]
        lu = logM .- κ .* n_minus_K
        mx = maximum(lu)
        logZ = log(sum(exp.(lu .- mx))) + mx
        return logZ - dot(empirical_p, lu)
    end
    
    res_p2 = optimize(nll_p2, [0.0], NelderMead())
    κ_hat = Optim.minimizer(res_p2)[1]
    theta_p2 = [-(l - 1) * κ_hat for l in 1:N]
    q_p2 = normalised_q(theta_p2)
    @printf("KL  1-param (κ=%.4f, p₂=%.4f)        = %.5f nats\n", κ_hat, exp(-κ_hat), kl(empirical_p, q_p2))
    println("  θ_p2: ", round.(theta_p2, digits=2))

    # --- (2) MAP hybrid (DuBois-regularised MaxEnt) ---
    # We fit N-1 parameters θ_₂ ... θ_N, but regularise them towards the DuBois 1-parameter 
    # baseline rather than zero. We use a flat ridge penalty (Gaussian prior) on the deviations:
    #   δ_l = θ_l - θ_l^DuBois  ~ N(0, τ₀²)
    # 
    # For small l (where MC data is abundant), the NLL gradient overpowers the prior, allowing 
    # the model to perfectly fit the empirical histogram structure.
    # For large l (where MC data is zero), the NLL gradient pushes the probability to zero (this is what we used to do),
    # but the prior then pushes it back smoothly to the physical(?) DuBois exponential decay.
    τ₀ = 1.0
    
    # f_map: Loss function (negative log-likelihood + log-prior penalty) in nats.
    function f_map(delta_reduced)
        # Pad δ₁=0 (gauge choice), then add corrections to the 1-parameter baseline
        delta = vcat(0.0, delta_reduced)
        theta = theta_p2 .+ delta
        
        # Unnormalised log-probabilities for all p(N) sectors
        lu = logM .+ C_matrix * theta
        
        # Log-sum-exp trick for the normalisation constant log(Z)
        mx = maximum(lu)
        logZ = log(sum(exp.(lu .- mx))) + mx
        
        # Flat L2 ridge penalty on the deviation from DuBois baseline
        reg = sum(delta[l]^2 for l in 2:N) / (2 * τ₀^2)
        
        # Loss = -<log Q>_P̂ + log Z + prior penalty
        return logZ - dot(empirical_p, lu) + reg
    end
    
    # g_map!: Gradient of the MAP loss for L-BFGS. 
    function g_map!(G, delta_reduced)
        delta = vcat(0.0, delta_reduced)
        theta = theta_p2 .+ delta
        
        # Recompute unnormalised log-probs to get the current model distribution Q
        lu = logM .+ C_matrix * theta
        mx = maximum(lu)
        unnorm = exp.(lu .- mx)
        q = unnorm ./ sum(unnorm)
        
        # Expected cycle counts under model (Q) vs empirical data (P̂)
        model_C = q' * C_matrix
        emp_C = empirical_p' * C_matrix
        
        # Gradient = NLL residual + derivative of the flat ridge penalty
        G .= (model_C .- emp_C)[2:end] .+ [delta[l] / τ₀^2 for l in 2:N]
    end
    
    res_map = optimize(f_map, g_map!, zeros(N - 1), LBFGS())
    delta_fit = vcat(0.0, Optim.minimizer(res_map))
    theta_map = theta_p2 .+ delta_fit
    q_map = normalised_q(theta_map)
    @printf("KL  MAP hybrid (τ₀=%.2f)        = %.5f nats\n", τ₀, kl(empirical_p, q_map))
    println("  θ_map: ", round.(theta_map, digits=2))

    # --- (3) Full MaxEnt (unconstrained cycle-length penalties) ---
    # The exponential family Q(k) ∝ M_k exp(∑ C_l θ_l) is the 
    # unique distribution that maximises entropy subject to matching the expected cycle counts
    # <C_l>_Q to the empirical averages <C_l>_P̂.
    # 
    # Inductive bias: Unlike the MAP model, there is NO physical inductive bias here. 
    # We fit N-1 parameters θ_₂ ... θ_N entirely independently.
    #  
    # weak isotropic L2 regularisation (1e-4) as numerical safeguard ONLY against collapse 
    # for long cycle lengths that are completely absent in the MC data (preventing θ_l → -∞).

    # Basic testing on UEG with 10^6 moves clearly showed unphysically large tails (-67 nats vs. -100 for the 33 cycle) c.f. the DeBois model 
    
    # f_nll: Loss function (NLL + weak L2) in nats.
    function f_nll(theta_reduced)
        # Pad θ₁=0 (gauge choice)
        theta = vcat(0.0, theta_reduced)
        
        # Unnormalised log-probabilities
        lu = logM .+ C_matrix * theta
        
        # Log-sum-exp trick for normalisation
        mx = maximum(lu)
        logZ = log(sum(exp.(lu .- mx))) + mx
        
        # Loss = -<log Q>_P̂ + log Z + weak isotropic L2 penalty
        return logZ - dot(empirical_p, lu) + 1e-4 * sum(theta_reduced .^ 2)
    end
    
    # g_nll!: Gradient of NLL + weak L2 for L-BFGS.
    function g_nll!(G, theta_reduced)
        theta = vcat(0.0, theta_reduced)
        
        # Current model distribution Q
        lu = logM .+ C_matrix * theta
        mx = maximum(lu)
        unnorm = exp.(lu .- mx)
        q = unnorm ./ sum(unnorm)
        
        # Gradient is exactly the MaxEnt matching condition: <C_l>_Q - <C_l>_P̂
        G .= ((q' * C_matrix) .- (empirical_p' * C_matrix))[2:end] .+ 2e-4 .* theta_reduced
    end
    
    res_full = optimize(f_nll, g_nll!, zeros(N - 1), LBFGS())
    theta_fit = vcat(0.0, Optim.minimizer(res_full))
    q_model = normalised_q(theta_fit)
    @printf("KL  full MaxEnt                 = %.5f nats\n", kl(empirical_p, q_model))
    println("  θ_full: ", round.(theta_fit, digits=2))

    # write fit residuals for ALL sectors (including unobserved ones)
    open("PermutationFamily_histogram_fit.dat", "w") do io
        println(io, "# count  family_index  P_hat  P_mult  P_p2  P_map  P_full  λ (nonzero parts, descending)")
        for k in 1:PermFamHisto.n_families
            c = PermFamHisto.count[k]
            p_hat = c / n_tot
            λk = permutation_family_lambda_from_rank(k, N, PermFamHisto.P)
            r = findfirst(iszero, λk)
            head = isnothing(r) ? λk : λk[1:(r - 1)]
            @printf(io, "%8d  %8d  %.5e  %.5e  %.5e  %.5e  %.5e  %s\n",
                    c, k, p_hat, q_mult[k], q_p2[k], q_map[k], q_model[k], string(collect(head)))
        end
    end

    open("PermutationFamily_histogram.dat", "w") do io
        println(io, "# count  family_index  P_hat  λ (nonzero parts, descending)")
        for (k, c) in hits
            p_hat = c / n_tot
            λk = permutation_family_lambda_from_rank(k, N, PermFamHisto.P)
            r = findfirst(iszero, λk)
            head = isnothing(r) ? λk : λk[1:(r - 1)]
            @printf(io, "%8d  %8d  %.5e  %s\n", c, k, p_hat, string(collect(head)))
        end
    end

    open("PermutationFamily_moments.dat", "w") do io
        @printf(io, "#  ∑_l l·P̂(l) = %.6f\n", permutation_pl_sum_rule(P_mc))
        cache = Dict{Tuple{Int,Float64},Float64}()
        println(io, "\n# k    P(l,k) MC    P_u(l,k)     P(k) exact")
        for k in 1:N
            p_ex = ideal_fermion_permutation_P_l(k, N, β, L, λħ; cache=cache)
            @printf(io, "%3d  %.5e  %.5e  %.5e\n", k, col[k], Pu_col[k], p_ex)
        end
    end

    # OK, calc some estimators ... ?

end

if abspath(PROGRAM_FILE) == @__FILE__
    run_family_histogram(; l_fixed=1)
end
