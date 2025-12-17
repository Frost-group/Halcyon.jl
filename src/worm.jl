# worm.jl - Worm algorithm for PIMC with periodic boundary conditions
# Ref: Spada et al. 2022, "Path-integral Monte Carlo worm algorithm for Bose systems
#      with periodic boundary conditions", Condens. Matter 2022, 7, 30
# arXiv:2203.00010

"""
Sector indicator for worm algorithm.
- Z_SECTOR: Diagonal configurations (closed polymers)
- G_SECTOR: Off-diagonal configurations (worm present with head/tail)
Ref: Spada et al. 2022, Section II.C
"""
@enum Sector Z_SECTOR G_SECTOR

"""
    WormConfiguration

Worm algorithm configuration for PIMC with periodic boundary conditions.

Ref: Spada et al. 2022, Section II.C

Design (Option A): Implicit endpoint representation.
- Stores beads 0 to M-1 in r[N, M, D].
- Bead M (the endpoint) is derived from permutation, winding, and bead 0 positions.
- In Z-sector: endpoint_i = r[P[i], 1, :] + w[i, :] * L
- In G-sector: 
    - if i == i_head: endpoint_i = r_c_head + w[i, :] * L
    - else: endpoint_i = r[P[i], 1, :] + w[i, :] * L

Fields:
- sector::Sector: Z (diagonal/closed) or G (off-diagonal/worm present)
- P::Vector{Int}: Permutation vector, P[i] = particle that polymer i connects to
- i_head::Int: Particle index of worm head (only meaningful in G-sector)
- i_tail::Int: Particle index of worm tail = P[i_head] (meaningful in G-sector)
- r_c_head::Vector{Float64}: Compact head coordinate in [0, L)^D
- r::Array{Float64,3}: Bead positions [particle, bead, dimension], 1:M for beads 0:M-1
- w::Matrix{Int}: Winding numbers [particle, dimension]
"""
@kwdef mutable struct WormConfiguration
    sector::Sector = Z_SECTOR
    P::Vector{Int}
    i_head::Int = 1
    i_tail::Int = 1
    r_c_head::Vector{Float64} = Float64[]
    r::Array{Float64,3}
    w::Matrix{Int}
end

function WormConfiguration(sys::System)
    N, M, D, L = sys.N, sys.M, sys.D, sys.L
    λ, δτ = sys.λ, sys.τ
    
    # Initialize positions: beads 0 to M-1
    r = zeros(N, M, D)
    
    for i in 1:N
        # Bead 0 uniformly in [0, L)^D
        r[i, 1, :] .= L .* rand(D)
        
        # Sample beads 1 to M-1 using a temporary closed loop Levy bridge
        # from bead 0 (index 1) to its image (also index 1)
        for j in 2:M
            # steps from j-1 to M
            Δj_fwd = M - j + 1
            Δj_back = 1
            denom = Δj_fwd + Δj_back
            
            # Mean: bridge from r[j-1] to r[1]
            r_mean = (Δj_fwd .* r[i, j-1, :] .+ Δj_back .* r[i, 1, :]) ./ denom
            σ = sqrt(2λ * δτ * Δj_back * Δj_fwd / denom)
            r[i, j, :] .= r_mean .+ σ .* randn(D)
        end
    end
    
    # Identity permutation: each polymer closes on itself
    P = collect(1:N)
    w = zeros(Int, N, D)
    
    return WormConfiguration(
        sector = Z_SECTOR,
        P = P,
        i_head = 1,
        i_tail = 1,
        r_c_head = vec(copy(r[1, 1, :])),
        r = r,
        w = w
    )
end

function Base.show(io::IO, cfg::WormConfiguration)
    N, M, D = size(cfg.r)
    print(io, "WormConfiguration($(cfg.sector), N=$N, M=$M, D=$D")
    if cfg.sector == G_SECTOR
        print(io, ", head=$(cfg.i_head), tail=$(cfg.i_tail)")
    end
    print(io, ")")
end

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers for implicit endpoints and consistency
# ═══════════════════════════════════════════════════════════════════════════════

"""
    get_endpoint(cfg, i, L) -> Vector

Return the coordinate of the M-th bead (endpoint) of polymer i.
This is a derived quantity in Option A.
"""
function get_endpoint(cfg::WormConfiguration, i::Int, L::Float64)
    if cfg.sector == G_SECTOR && i == cfg.i_head
        return cfg.r_c_head .+ cfg.w[i, :] .* L
    else
        target = cfg.P[i]
        return cfg.r[target, 1, :] .+ cfg.w[i, :] .* L
    end
end

"""
    get_bead(cfg, i, j, L) -> Vector

Return the coordinate of bead j (0 to M) of polymer i.
For j < M, returns from cfg.r. For j = M, returns get_endpoint.
"""
function get_bead(cfg::WormConfiguration, i::Int, j::Int, L::Float64)
    M = size(cfg.r, 2)
    if j < M
        return cfg.r[i, j+1, :]
    elseif j == M
        return get_endpoint(cfg, i, L)
    else
        throw(BoundsError("Bead index $j out of range [0, $M]"))
    end
end

@inline function _wrap_compact_and_winding(x::Float64, L::Float64)
    n = floor(Int, x / L)
    return x - n * L, n
end

function update_head_compact!(cfg::WormConfiguration, L::Float64)
    D = size(cfg.r, 3)
    i_H = cfg.i_head
    # head endpoint position
    r_M = get_endpoint(cfg, i_H, L)
    for d in 1:D
        x_c, n = _wrap_compact_and_winding(r_M[d], L)
        cfg.r_c_head[d] = x_c
        cfg.w[i_H, d] = n
    end
    return nothing
end

"""
    get_cycle(cfg::WormConfiguration, i::Int) -> Vector{Int}

Find all particle indices in the permutation cycle containing particle i.
"""
function get_cycle(cfg::WormConfiguration, i::Int)
    N = length(cfg.P)
    cycle = [i]
    j = cfg.P[i]
    count = 0
    while j != i && count < N
        push!(cycle, j)
        j = cfg.P[j]
        count += 1
    end
    return cycle
end

"""
    recenter!(cfg::WormConfiguration, i::Int, L::Float64)

Shift polymer i so that bead 0 is within the fundamental cell [0, L)^D.
Updates winding numbers of polymer i and all polymers that connect to it.
"""
function recenter!(cfg::WormConfiguration, i::Int, L::Float64)
    D = size(cfg.r, 3)
    M = size(cfg.r, 2)
    
    # Compute shift needed to put bead 0 in [0, L)^D
    Δw = zeros(Int, D)
    for d in 1:D
        Δw[d] = floor(Int, cfg.r[i, 1, d] / L)
    end
    
    if all(Δw .== 0)
        return  # Already centered
    end
    
    # Shift beads 0..M-1 of polymer i
    shift = Δw .* L
    for j in 1:M
        cfg.r[i, j, :] .-= shift
    end
    
    # 2. Update i's own winding to keep its endpoint moving WITH its path.
    # The endpoint depends on r[P[i], 1] (or r_c_head if head).
    # Since only i was shifted, the winding must change unless the target also shifted.
    if cfg.sector == G_SECTOR && i == cfg.i_head
        cfg.w[i, :] .-= Δw
    elseif cfg.P[i] != i
        cfg.w[i, :] .-= Δw
    end
    
    # 3. Update windings of any polymer k that connects TO i.
    # Their endpoint r_end[k] = r[i, 1] + w[k]*L shifted by -shift.
    # To keep r_end[k] physically fixed, increase w[k].
    N = size(cfg.r, 1)
    for k in 1:N
        if cfg.P[k] == i && k != i
            if !(cfg.sector == G_SECTOR && k == cfg.i_head)
                cfg.w[k, :] .+= Δw
            end
        end
    end
end

"""
    fix_all_windings!(cfg::WormConfiguration, L::Float64)

Reset winding numbers for all particles to minimize spring lengths.
"""
function fix_all_windings!(cfg::WormConfiguration, L::Float64)
    N, M, D = size(cfg.r)
    for i in 1:N
        # Current bead M-1
        r_M_minus_1 = cfg.r[i, M, :]
        
        # Target for closure
        if cfg.sector == G_SECTOR && i == cfg.i_head
            target_0 = cfg.r_c_head
        else
            target_0 = cfg.r[cfg.P[i], 1, :]
        end
        
        # Choose w to minimize |r_M_minus_1 - (target_0 + w*L)|
        for d in 1:D
            cfg.w[i, d] = round(Int, (r_M_minus_1[d] - target_0[d]) / L)
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Lévy Construction for Staging
# ═══════════════════════════════════════════════════════════════════════════════

"""
    levy_sample!(cfg, i, j0, j1, λ, δτ, L)

Sample intermediate beads j0+1 to j1-1 of polymer i via Lévy construction.
j0, j1 in 0:M.
"""
function levy_sample!(cfg::WormConfiguration, i::Int, j0::Int, j1::Int, λ::Float64, δτ::Float64, L::Float64)
    D = size(cfg.r, 3)
    M = size(cfg.r, 2)
    
    # Endpoint coordinate
    r_end = get_bead(cfg, i, j1, L)
    
    # Starting coordinate
    r_prev = copy(get_bead(cfg, i, j0, L))
    
    # Sample beads sequentially from j0+1 to j1-1
    for j in (j0 + 1):(j1 - 1)
        # steps from j to j1
        Δj_fwd = j1 - j
        Δj_back = 1
        denom = Δj_fwd + Δj_back
        
        # Mean: bridge from r_prev to r_end
        r_star = (Δj_fwd .* r_prev .+ Δj_back .* r_end) ./ denom
        σ = sqrt(2λ * δτ * Δj_back * Δj_fwd / denom)
        
        # Update physical storage (beads 0..M-1)
        # Bead j is at index j+1 in Julia
        cfg.r[i, j + 1, :] .= r_star .+ σ .* randn(D)
        r_prev .= cfg.r[i, j + 1, :]
    end
end

"""
    ρ_free_sp(r1, r2, λ, Δτ) -> Float64

Single-particle free propagator (unnormalized).
"""
function ρ_free_sp(r1::AbstractVector, r2::AbstractVector, λ::Float64, Δτ::Float64)
    Δr² = sum(abs2, r1 .- r2)
    return exp(-Δr² / (4λ * Δτ))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Monte Carlo Moves
# ═══════════════════════════════════════════════════════════════════════════════

@kwdef mutable struct WormParams
    C::Float64 = 1.0
    j_max::Int = 10
    j_max_open::Int = 10
    j_max_swap::Int = 10
    r_max::Float64 = 0.5
end

"""
    translate!(cfg, sys, params) -> Bool
"""
function translate!(cfg::WormConfiguration, sys::System, params::WormParams)
    N, D, L = sys.N, sys.D, sys.L
    i = rand(1:N)
    cycle = get_cycle(cfg, i)
    Δr = params.r_max .* (2 .* rand(D) .- 1)
    
    # Non-interacting acceptance
    ΔU = 0.0
    
    if rand() < exp(-ΔU)
        M = size(cfg.r, 2)
        for p in cycle
            for j in 1:M
                cfg.r[p, j, :] .+= Δr
            end
        end

        if cfg.sector == G_SECTOR && any(==(cfg.i_head), cycle)
            i_H = cfg.i_head
            x = cfg.r_c_head .+ Δr
            for d in 1:D
                x_c, n = _wrap_compact_and_winding(x[d], L)
                cfg.r_c_head[d] = x_c
                cfg.w[i_H, d] += n
            end
        end
        
        for p in cycle
            recenter!(cfg, p, L)
        end
        return true
    end
    return false
end

"""
    redraw!(cfg, sys, params) -> Bool
"""
function redraw!(cfg::WormConfiguration, sys::System, params::WormParams)
    N, M, D, L = sys.N, sys.M, sys.D, sys.L
    λ, δτ = sys.λ, sys.τ
    
    if M < 2
        return false
    end
    
    i = rand(1:N)
    j0 = rand(0:M-1)
    Δj = rand(2:min(params.j_max, M))
    j1 = j0 + Δj
    
    if j1 > M
        j1 = M
    end
    
    r_old = copy(cfg.r[i, (j0+1):(min(j1,M)), :])
    
    levy_sample!(cfg, i, j0, j1, λ, δτ, L)
    
    # Non-interacting acceptance
    ΔU = 0.0
    
    if rand() < exp(-ΔU)
        if j0 == 0
            recenter!(cfg, i, L)
        end
        return true
    else
        cfg.r[i, (j0+1):(min(j1,M)), :] .= r_old
        return false
    end
end

"""
    open!(cfg, sys, params) -> Bool
"""
function open!(cfg::WormConfiguration, sys::System, params::WormParams)
    if cfg.sector == G_SECTOR
        return false
    end
    
    N, M, D, L = sys.N, sys.M, sys.D, sys.L
    λ, δτ = sys.λ, sys.τ
    V = L^D
    
    i_H = rand(1:N)
    j0 = rand(0:M-1)
    Δj = M - j0
    Δ = min(sqrt(2λ * Δj * δτ), L / 2)
    
    r_old_M = get_endpoint(cfg, i_H, L)
    r_j0 = get_bead(cfg, i_H, j0, L)
    
    # Propose new head position
    Δr = Δ .* (2 .* rand(D) .- 1)
    r_new_M = r_old_M .+ Δr
    
    # Store old segment
    r_old_segment = copy(cfg.r[i_H, (j0+1):M, :])
    w_old = copy(cfg.w[i_H, :])
    
    # Switch to G-sector temporarily to use get_endpoint with r_c_head
    cfg.sector = G_SECTOR
    cfg.i_head = i_H
    cfg.i_tail = cfg.P[i_H]
    
    # Initialize r_c_head and head winding
    for d in 1:D
        x_c, n = _wrap_compact_and_winding(r_new_M[d], L)
        cfg.r_c_head[d] = x_c
        cfg.w[i_H, d] = n
    end
    
    levy_sample!(cfg, i_H, j0, M, λ, δτ, L)
    
    # Free propagator ratio
    ρ_new = ρ_free_sp(r_j0, r_new_M, λ, Δj * δτ)
    ρ_old = ρ_free_sp(r_j0, r_old_M, λ, Δj * δτ)
    
    prefactor = params.C * N * (2Δ)^D / V
    A_O = prefactor * ρ_new / ρ_old
    
    if rand() < min(1.0, A_O)
        return true
    else
        # Revert
        cfg.sector = Z_SECTOR
        cfg.r[i_H, (j0+1):M, :] .= r_old_segment
        cfg.w[i_H, :] .= w_old
        return false
    end
end

"""
    close!(cfg, sys, params) -> Bool
"""
function close!(cfg::WormConfiguration, sys::System, params::WormParams)
    if cfg.sector == Z_SECTOR
        return false
    end
    
    N, M, D, L = sys.N, sys.M, sys.D, sys.L
    λ, δτ = sys.λ, sys.τ
    V = L^D
    
    i_H = cfg.i_head
    i_T = cfg.i_tail
    
    j0 = rand(0:M-1)
    Δj = M - j0
    Δ = min(sqrt(2λ * Δj * δτ), L / 2)
    
    r_head_M = get_endpoint(cfg, i_H, L)
    r_tail_0 = cfg.r[i_T, 1, :]
    
    # Find nearest image of tail for closure
    r_T = copy(r_tail_0)
    for d in 1:D
        n = round(Int, (r_head_M[d] - r_tail_0[d]) / L)
        r_T[d] = r_tail_0[d] + n * L
    end
    
    if any(abs.(r_head_M .- r_T) .> Δ)
        return false
    end
    
    # Store old state
    r_old_segment = copy(cfg.r[i_H, (j0+1):M, :])
    w_old = copy(cfg.w[i_H, :])
    r_old_M = copy(r_head_M)
    r_j0 = get_bead(cfg, i_H, j0, L)
    
    # Propose closure: update winding to match r_T
    for d in 1:D
        cfg.w[i_H, d] = round(Int, (r_T[d] - r_tail_0[d]) / L)
    end
    
    # Temporarily switch to Z-sector
    cfg.sector = Z_SECTOR
    
    levy_sample!(cfg, i_H, j0, M, λ, δτ, L)
    
    ρ_new = ρ_free_sp(r_j0, r_T, λ, Δj * δτ)
    ρ_old = ρ_free_sp(r_j0, r_old_M, λ, Δj * δτ)
    
    prefactor = V / (params.C * N * (2Δ)^D)
    A_C = prefactor * ρ_new / ρ_old
    
    if rand() < min(1.0, A_C)
        cfg.i_head = 1
        cfg.i_tail = 1
        return true
    else
        cfg.sector = G_SECTOR
        cfg.r[i_H, (j0+1):M, :] .= r_old_segment
        cfg.w[i_H, :] .= w_old
        return false
    end
end

"""
    move_head!(cfg, sys, params) -> Bool
"""
function move_head!(cfg::WormConfiguration, sys::System, params::WormParams)
    if cfg.sector == Z_SECTOR
        return false
    end
    M, D, L = sys.M, sys.D, sys.L
    λ, δτ = sys.λ, sys.τ
    i_H = cfg.i_head
    
    j0 = rand(0:M-1)
    Δj = M - j0
    
    r_old_segment = copy(cfg.r[i_H, (j0+1):M, :])
    w_old = copy(cfg.w[i_H, :])
    r_c_head_old = copy(cfg.r_c_head)
    
    r_j0 = get_bead(cfg, i_H, j0, L)
    
    # Sample new head endpoint
    σ = sqrt(2λ * Δj * δτ)
    r_new_M = r_j0 .+ σ .* randn(D)
    
    for d in 1:D
        x_c, n = _wrap_compact_and_winding(r_new_M[d], L)
        cfg.r_c_head[d] = x_c
        cfg.w[i_H, d] = n
    end
    
    levy_sample!(cfg, i_H, j0, M, λ, δτ, L)
    
    # Non-interacting acceptance = 1
    return true
end

"""
    move_tail!(cfg, sys, params) -> Bool
"""
function move_tail!(cfg::WormConfiguration, sys::System, params::WormParams)
    if cfg.sector == Z_SECTOR
        return false
    end
    M, D, L = sys.M, sys.D, sys.L
    λ, δτ = sys.λ, sys.τ
    i_T = cfg.i_tail
    
    j1 = rand(1:M)
    Δj = j1
    
    r_old_segment = copy(cfg.r[i_T, 1:j1, :])
    # r[i_T, j1+1] is fixed (j1 < M) or endpoint (j1=M)
    r_j1 = get_bead(cfg, i_T, j1, L)
    
    # Sample new bead 0
    σ = sqrt(2λ * Δj * δτ)
    cfg.r[i_T, 1, :] .= r_j1 .+ σ .* randn(D)
    
    levy_sample!(cfg, i_T, 0, j1, λ, δτ, L)
    
    recenter!(cfg, i_T, L)
    return true
end

"""
    swap!(cfg, sys, params) -> Bool
"""
function swap!(cfg::WormConfiguration, sys::System, params::WormParams)
    if cfg.sector == Z_SECTOR
        return false
    end
    
    N, M, D, L = sys.N, sys.M, sys.D, sys.L
    λ, δτ = sys.λ, sys.τ
    i_H = cfg.i_head
    i_T = cfg.i_tail
    
    # Select pivot time slice j_P from 1 to M
    j_max_possible = min(params.j_max_swap, M)
    if j_max_possible < 1 return false end
    j_P = rand(1:j_max_possible)
    r_c_H = cfg.r_c_head
    
    # Tower sampling for target i_0 (Spada Eq. 31)
    # Π_P(i) = ρ_free(r_c_H, r_{i,j_P}, j_P δτ)
    weights = zeros(N)
    for i in 1:N
        r_i_jP = get_bead(cfg, i, j_P, L)
        weights[i] = ρ_free_sp(r_c_H, r_i_jP, λ, j_P * δτ)
    end
    
    Σ_P = sum(weights)
    if Σ_P ≤ 0 return false end
    
    r_sample = rand() * Σ_P
    cumsum_w = 0.0
    i_0 = 0
    for i in 1:N
        cumsum_w += weights[i]
        if r_sample < cumsum_w
            i_0 = i
            break
        end
    end
    
    if i_0 == 0 || i_0 == i_T return false end
    
    # Σ_0 for inverse process (Spada Eq. 33)
    # Σ_0 = Σ_i ρ_free(r_c_{i_0,0}, r_{i,j_P}, j_P δτ)
    r_c_i0_0 = copy(cfg.r[i_0, 1, :])
    weights_inv = zeros(N)
    for i in 1:N
        r_i_jP = get_bead(cfg, i, j_P, L)
        weights_inv[i] = ρ_free_sp(r_c_i0_0, r_i_jP, λ, j_P * δτ)
    end
    Σ_0 = sum(weights_inv)
    if Σ_0 ≤ 0 return false end
    
    # Identify i* such that P(i*) = i_0 (Spada: new head index)
    i_star = findfirst(k -> cfg.P[k] == i_0, 1:N)
    if isnothing(i_star) return false end
    
    # Acceptance probability A_SW = Σ_P / Σ_0 (Spada Eq. 33)
    A_SW = Σ_P / Σ_0
    
    if rand() < min(1.0, A_SW)
        # Store old target bead 0 and current head coordinate
        r_c_i0_0_old = copy(cfg.r[i_0, 1, :])
        r_c_H_old = copy(cfg.r_c_head)
        
        # 1. Relocate i_0 bead 0 to the old head position
        cfg.r[i_0, 1, :] .= r_c_H_old
        
        # 2. Relocate head compact coordinate to the old i_0 bead 0 position
        cfg.r_c_head .= r_c_i0_0_old
        
        # 3. Update permutation (local rewire)
        # Old head (i_H) now points to i_0.
        # i_star (which pointed to i_0) now points to i_T (the worm tail).
        cfg.P[i_H] = i_0
        cfg.P[i_star] = i_T
        
        # 4. New head index is i_star
        cfg.i_head = i_star
        
        # 5. Redraw segment of i_0 from 0 to j_P via staging
        # This links the new i_0,0 (at r_c_H_old) to i_0,j_P.
        levy_sample!(cfg, i_0, 0, j_P, λ, δτ, L)
        
        recenter!(cfg, i_0, L)
        return true
    end
    return false
end

function energy_thermodynamic(cfg::WormConfiguration, sys::System)
    N, M, D = sys.N, sys.M, sys.D
    λ, δτ = sys.λ, sys.τ
    
    spring_sum = 0.0
    for i in 1:N
        for j in 0:M-1
            r_j = get_bead(cfg, i, j, sys.L)
            r_next = get_bead(cfg, i, j+1, sys.L)
            spring_sum += sum(abs2, r_j .- r_next)
        end
    end
    
    # Thermodynamic estimator (Spada Eq. A1):
    # E = DN/(2δτ) - spring_sum / (4λ δτ² M)
    E = D * N / (2 * δτ) - spring_sum / (4 * λ * δτ^2 * M)
    return E
end

"""
    energy_virial(cfg, sys) -> Float64

Virial energy estimator for non-interacting system.
Ref: Spada et al. 2022, Appendix Eq. A2 (with U=0)
"""
function energy_virial(cfg::WormConfiguration, sys::System)
    N, M, D = sys.N, sys.M, sys.D
    λ, δτ, β, L = sys.λ, sys.τ, sys.β, sys.L
    
    term2 = 0.0
    for i in 1:N
        r_M_minus_1 = get_bead(cfg, i, M - 1, L)
        r_M = get_bead(cfg, i, M, L)
        r_0 = get_bead(cfg, i, 0, L)
        
        # (r_{M-1} - r_M) · (r_M - r_0)
        term2 += sum((r_M_minus_1 .- r_M) .* (r_M .- r_0))
    end
    
    # E_vir = DN/(2β) + term2 / (4λ δτ² M)
    E = D * N / (2 * β) + term2 / (4 * λ * δτ^2 * M)
    return E
end

function worm_step!(cfg::WormConfiguration, sys::System, params::WormParams)
    r = rand()
    N = sys.N
    
    if N == 1
        if r < 0.15 return translate!(cfg, sys, params) ? :translate : :reject
        elseif r < 0.45 return redraw!(cfg, sys, params) ? :redraw : :reject
        elseif r < 0.55 return open!(cfg, sys, params) ? :open : :reject
        elseif r < 0.65 return close!(cfg, sys, params) ? :close : :reject
        elseif r < 0.825 return move_head!(cfg, sys, params) ? :move_head : :reject
        else return move_tail!(cfg, sys, params) ? :move_tail : :reject
        end
    else
        if r < 0.10 return translate!(cfg, sys, params) ? :translate : :reject
        elseif r < 0.35 return redraw!(cfg, sys, params) ? :redraw : :reject
        elseif r < 0.45 return open!(cfg, sys, params) ? :open : :reject
        elseif r < 0.55 return close!(cfg, sys, params) ? :close : :reject
        elseif r < 0.70 return swap!(cfg, sys, params) ? :swap : :reject
        elseif r < 0.85 return move_head!(cfg, sys, params) ? :move_head : :reject
        else return move_tail!(cfg, sys, params) ? :move_tail : :reject
        end
    end
end
