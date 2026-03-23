# Ideal spin-polarised Fermi gas: dense permutation-family histogram (conjugacy-class index).

using Halcyon
using Printf
using ProgressMeter
using Optim

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
                             l_fixed::Int=1, equil::Int=100_000, steps::Int=20_000_000,
                             measure_every::Int=25)
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

    # --- MaxEnt (Mean-Field) DuBois fit ---
    println("Fitting MaxEnt MeanField model...")
    logM = zeros(Float64, PermFamHisto.n_families)
    C_matrix = zeros(Int, PermFamHisto.n_families, N)
    for k in 1:PermFamHisto.n_families
        λk = permutation_family_lambda_from_rank(k, N, PermFamHisto.P)
        C_k = C_permutation_sector(λk)
        C_matrix[k, :] .= C_k
        
        logM[k] = sum(log.(1:N)) # log(N!)
        for l in 1:N
            if C_k[l] > 0
                logM[k] -= sum(log.(1:C_k[l])) + C_k[l] * log(l)
            end
        end
    end
    
    empirical_p = Float64.(PermFamHisto.count) ./ n_tot
    
    # nega log-likelihood baby
    function f_nll(theta_reduced)
        theta = vcat(0.0, theta_reduced)
        log_unnorm = logM .+ C_matrix * theta
        max_val = maximum(log_unnorm)
        logZ = log(sum(exp.(log_unnorm .- max_val))) + max_val
        return logZ - sum(empirical_p .* log_unnorm) + 1e-4 * sum(theta_reduced.^2)
    end

    # explicit gradient to avoid autodiff
    #  just <C_l-model> - <C_l-monte-carlo> 
    #   regularisation (little bid added on) required in case of no MC examples
    function g_nll!(G, theta_reduced)
        theta = vcat(0.0, theta_reduced)
        log_unnorm = logM .+ C_matrix * theta
        max_val = maximum(log_unnorm)
        unnorm = exp.(log_unnorm .- max_val)
        q_model = unnorm ./ sum(unnorm)
        
        model_C = q_model' * C_matrix
        emp_C = empirical_p' * C_matrix
        G .= (model_C .- emp_C)[2:end] .+ 2e-4 .* theta_reduced
    end
    
    # optimimise over log-likelihood 
    res = optimize(f_nll, g_nll!, zeros(N-1), LBFGS())
    theta_fit = vcat(0.0, Optim.minimizer(res))
    log_unnorm = logM .+ C_matrix * theta_fit # for all permutation familes
    logZ = log(sum(exp.(log_unnorm .- maximum(log_unnorm)))) + maximum(log_unnorm) # normalise with explicit sum over permutation families to get Z
    q_model = exp.(log_unnorm .- logZ) # evaluate all model probs
    
    # write out fit / residuals
    open("PermutationFamily_histogram_fit.dat", "w") do io
        println(io, "# count  family_index  P_hat  P_model  log_residual  λ (nonzero parts, descending)")
        for (k, c) in hits
            p_hat = c / n_tot
            p_m = q_model[k]
            res_val = p_hat > 0 ? log(p_hat / p_m) : 0.0 # units nats 
            
            λk = permutation_family_lambda_from_rank(k, N, PermFamHisto.P)
            r = findfirst(iszero, λk)
            head = isnothing(r) ? λk : λk[1:(r - 1)]
            @printf(io, "%8d  %8d  %.5e  %.5e  %8.5f  %s\n", c, k, p_hat, p_m, res_val, string(collect(head)))
        end
    end
    println("Fitted theta(=log P_l): ", round.(theta_fit, digits=3))
    println("Fitted P_l: ", round.(exp.(theta_fit), digits=3))

    kl_div = sum(p_hat .* log.(p_hat ./ p_m) for (p_hat, p_m) in zip(empirical_p, q_model) if p_hat > 0)
    @printf("KL(empirical || model) = %.5f nats\n", kl_div)

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
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_family_histogram(; l_fixed=1)
end
