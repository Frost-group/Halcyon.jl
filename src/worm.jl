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
- In Z-sector: endpoint_i = r[next[i], 1, :] + w[i, :] * L
- In G-sector: 
    - if i == i_head: endpoint_i = r_c_head + w[i, :] * L
    - else: endpoint_i = r[next[i], 1, :] + w[i, :] * L

Fields:
- sector::Sector: Z (diagonal/closed) or G (off-diagonal/worm present)
- next::Vector{Int}: Forward permutation vector, next[i] = particle that polymer i connects to
- prev::Vector{Int}: Backward permutation vector, prev[j] = particle that connects to polymer j
- i_head::Int: Particle index of worm head (only meaningful in G-sector)
- i_tail::Int: Particle index of worm tail = next[i_head] (meaningful in G-sector)
- r_c_head::Vector{Float64}: Compact head coordinate in [0, L)^D
- r::Array{Float64,3}: Bead positions [particle, bead, dimension], 1:M for beads 0:M-1
- w::Matrix{Int}: Winding numbers [particle, dimension]
"""
@kwdef mutable struct WormConfiguration
    sector::Sector = Z_SECTOR
    next::Vector{Int}
    prev::Vector{Int}
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
    next = collect(1:N)
    prev = collect(1:N)
    w = zeros(Int, N, D)
    
    return WormConfiguration(
        sector = Z_SECTOR,
        next = next,
        prev = prev,
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
    get_endpoint(cfg, i, d, L) -> Float64

Return the d-th coordinate of the M-th bead (endpoint) of polymer i.
Zero-allocation accessor.
"""
@inline function get_endpoint(cfg::WormConfiguration, i::Int, d::Int, L::Float64)
    if cfg.sector == G_SECTOR && i == cfg.i_head
        return cfg.r_c_head[d] + cfg.w[i, d] * L
    else
        target = cfg.next[i]
        return cfg.r[target, 1, d] + cfg.w[i, d] * L
    end
end

"""
    get_endpoint(cfg, i, L) -> Vector

Return the coordinate of the M-th bead (endpoint) of polymer i.
This is a derived quantity in Option A.
NOTE: This allocates. Use the dimension-aware version for inner loops.
"""
function get_endpoint(cfg::WormConfiguration, i::Int, L::Float64)
    D = size(cfg.r, 3)
    out = zeros(D)
    for d in 1:D
        out[d] = get_endpoint(cfg, i, d, L)
    end
    return out
end

"""
    get_bead(cfg, i, j, L) -> Vector (copy or view)

Return the coordinate of bead j (0 to M) of polymer i.
NOTE: This allocates for j = M. Use the dimension-aware version for inner loops.
"""
function get_bead(cfg::WormConfiguration, i::Int, j::Int, L::Float64)
    M = size(cfg.r, 2)
    if j < M
        return @view cfg.r[i, j+1, :]
    elseif j == M
        return get_endpoint(cfg, i, L)
    else
        throw(BoundsError("Bead index $j out of range [0, $M]"))
    end
end

"""
    get_bead(cfg, i, j, d, L) -> Float64

Return the d-th coordinate of bead j (0 to M) of polymer i.
Zero-allocation accessor.
"""
@inline function get_bead(cfg::WormConfiguration, i::Int, j::Int, d::Int, L::Float64)
    M = size(cfg.r, 2)
    if j < M
        return cfg.r[i, j+1, d]
    elseif j == M
        return get_endpoint(cfg, i, d, L)
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
    N = length(cfg.next)
    cycle = [i]
    j = cfg.next[i]
    count = 0
    while j != i && count < N
        push!(cycle, j)
        j = cfg.next[j]
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
    # Δw[d] is how many boxes we are away from fundamental cell
    Δw_1 = floor(Int, cfg.r[i, 1, 1] / L)
    Δw_2 = floor(Int, cfg.r[i, 1, 2] / L)
    Δw_3 = floor(Int, cfg.r[i, 1, 3] / L)
    
    if Δw_1 == 0 && Δw_2 == 0 && Δw_3 == 0
        return nothing
    end
    
    # Shift beads 0..M-1 of polymer i
    for j in 1:M
        cfg.r[i, j, 1] -= Δw_1 * L
        cfg.r[i, j, 2] -= Δw_2 * L
        cfg.r[i, j, 3] -= Δw_3 * L
    end
    
    # 2. Update i's own winding to keep its endpoint moving WITH its path.
    # The endpoint depends on r[next[i], 1] (or r_c_head if head).
    # Since only i was shifted, the winding must change unless the target also shifted.
    if cfg.sector == G_SECTOR && i == cfg.i_head
        cfg.w[i, 1] -= Δw_1
        cfg.w[i, 2] -= Δw_2
        cfg.w[i, 3] -= Δw_3
    elseif cfg.next[i] != i
        cfg.w[i, 1] -= Δw_1
        cfg.w[i, 2] -= Δw_2
        cfg.w[i, 3] -= Δw_3
    end
    
    # 3. Update windings of the polymer k that connects TO i.
    # Their endpoint r_end[k] = r[i, 1] + w[k]*L shifted by -shift.
    # To keep r_end[k] physically fixed, increase w[k].
    # OPTIMIZED: Use prev[i] instead of O(N) search.
    k = cfg.prev[i]
    if k != i
        if !(cfg.sector == G_SECTOR && k == cfg.i_head)
            cfg.w[k, 1] += Δw_1
            cfg.w[k, 2] += Δw_2
            cfg.w[k, 3] += Δw_3
        end
    end
    return nothing
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
            target_0 = cfg.r[cfg.next[i], 1, :]
        end
        
        # Choose w to minimize |r_M_minus_1 - (target_0 + w*L)|
        for d in 1:D
            cfg.w[i, d] = round(Int, (r_M_minus_1[d] - target_0[d]) / L)
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Interaction Helpers
# ═══════════════════════════════════════════════════════════════════════════════

"""
    get_interaction_action(cfg, sys, i, j) -> Float64

Interaction action for particle i at time slice j (between bead j and j+1)
with all other particles.
Ref: Spada et al. 2022, Eq. 57.
"""
function get_interaction_action(cfg::WormConfiguration, sys::System, i::Int, j::Int)
    N, L, D, M = sys.N, sys.L, sys.D, sys.M
    if N == 1 || sys.U isa NullPairPotential
        return 0.0
    end
    
    λ, δτ = sys.λ, sys.τ
    a_hs = sys.U isa HardSpherePotential ? sys.U.a : 0.0
    
    # HOIST particle i coordinates
    ri_j_1 = get_bead(cfg, i, j, 1, L)
    ri_j_2 = get_bead(cfg, i, j, 2, L)
    ri_j_3 = get_bead(cfg, i, j, 3, L)
    ri_n_1 = get_bead(cfg, i, j+1, 1, L)
    ri_n_2 = get_bead(cfg, i, j+1, 2, L)
    ri_n_3 = get_bead(cfg, i, j+1, 3, L)
    
    action = 0.0

    # BRANCH REMOVAL: Specialise the loop based on whether j+1 is an endpoint
    if j < M - 1
        # FAST PATH: k,j and k,j+1 are both internal beads (stored in cfg.r)
        @inbounds for k in 1:N
            k == i && continue
            
            rk_j_1 = cfg.r[k, j+1, 1]
            rk_j_2 = cfg.r[k, j+1, 2]
            rk_j_3 = cfg.r[k, j+1, 3]
            rk_n_1 = cfg.r[k, j+2, 1]
            rk_n_2 = cfg.r[k, j+2, 2]
            rk_n_3 = cfg.r[k, j+2, 3]
            
            dx_j1 = ri_j_1 - rk_j_1; dx_j1 -= L * round(dx_j1 / L)
            dx_j2 = ri_j_2 - rk_j_2; dx_j2 -= L * round(dx_j2 / L)
            dx_j3 = ri_j_3 - rk_j_3; dx_j3 -= L * round(dx_j3 / L)
            d2_j = dx_j1*dx_j1 + dx_j2*dx_j2 + dx_j3*dx_j3

            dx_n1 = ri_n_1 - rk_n_1; dx_n1 -= L * round(dx_n1 / L)
            dx_n2 = ri_n_2 - rk_n_2; dx_n2 -= L * round(dx_n2 / L)
            dx_n3 = ri_n_3 - rk_n_3; dx_n3 -= L * round(dx_n3 / L)
            d2_next = dx_n1*dx_n1 + dx_n2*dx_n2 + dx_n3*dx_n3
            
            dot_prod = dx_j1*dx_n1 + dx_j2*dx_n2 + dx_j3*dx_n3
            
            if sys.U isa HardSpherePotential
                ratio = cao_berne_ratio(sqrt(d2_j), sqrt(d2_next), dot_prod, a_hs, λ, δτ)
                if ratio <= 0.0; return Inf; end
                action -= log(ratio)
            else
                action += (sys.U(sqrt(d2_j)) + sys.U(sqrt(d2_next))) * 0.5 * δτ
            end
        end
    else
        # SLOW PATH: At least one bead is an endpoint (derived)
        @inbounds for k in 1:N
            k == i && continue
            
            rk_j_1 = get_bead(cfg, k, j, 1, L)
            rk_j_2 = get_bead(cfg, k, j, 2, L)
            rk_j_3 = get_bead(cfg, k, j, 3, L)
            rk_n_1 = get_bead(cfg, k, j+1, 1, L)
            rk_n_2 = get_bead(cfg, k, j+1, 2, L)
            rk_n_3 = get_bead(cfg, k, j+1, 3, L)
            
            dx_j1 = ri_j_1 - rk_j_1; dx_j1 -= L * round(dx_j1 / L)
            dx_j2 = ri_j_2 - rk_j_2; dx_j2 -= L * round(dx_j2 / L)
            dx_j3 = ri_j_3 - rk_j_3; dx_j3 -= L * round(dx_j3 / L)
            d2_j = dx_j1*dx_j1 + dx_j2*dx_j2 + dx_j3*dx_j3

            dx_n1 = ri_n_1 - rk_n_1; dx_n1 -= L * round(dx_n1 / L)
            dx_n2 = ri_n_2 - rk_n_2; dx_n2 -= L * round(dx_n2 / L)
            dx_n3 = ri_n_3 - rk_n_3; dx_n3 -= L * round(dx_n3 / L)
            d2_next = dx_n1*dx_n1 + dx_n2*dx_n2 + dx_n3*dx_n3
            
            dot_prod = dx_j1*dx_n1 + dx_j2*dx_n2 + dx_j3*dx_n3
            
            if sys.U isa HardSpherePotential
                ratio = cao_berne_ratio(sqrt(d2_j), sqrt(d2_next), dot_prod, a_hs, λ, δτ)
                if ratio <= 0.0; return Inf; end
                action -= log(ratio)
            else
                action += (sys.U(sqrt(d2_j)) + sys.U(sqrt(d2_next))) * 0.5 * δτ
            end
        end
    end
    return action
end

"""
    get_interaction_action_external(cfg, sys, i, j, cycle_mask) -> Float64

Interaction action for particle i at slice j with all particles NOT in cycle_mask.
Used to optimize translate! move.
"""
function get_interaction_action_external(cfg::WormConfiguration, sys::System, i::Int, j::Int, cycle_mask::AbstractVector{Bool})
    N, L, D, M = sys.N, sys.L, sys.D, sys.M
    if N == 1 || sys.U isa NullPairPotential
        return 0.0
    end
    
    λ, δτ = sys.λ, sys.τ
    a_hs = sys.U isa HardSpherePotential ? sys.U.a : 0.0
    
    # HOIST particle i coordinates
    ri_j_1 = get_bead(cfg, i, j, 1, L)
    ri_j_2 = get_bead(cfg, i, j, 2, L)
    ri_j_3 = get_bead(cfg, i, j, 3, L)
    ri_n_1 = get_bead(cfg, i, j+1, 1, L)
    ri_n_2 = get_bead(cfg, i, j+1, 2, L)
    ri_n_3 = get_bead(cfg, i, j+1, 3, L)
    
    action = 0.0

    if j < M - 1
        @inbounds for k in 1:N
            cycle_mask[k] && continue
            
            rk_j_1 = cfg.r[k, j+1, 1]
            rk_j_2 = cfg.r[k, j+1, 2]
            rk_j_3 = cfg.r[k, j+1, 3]
            rk_n_1 = cfg.r[k, j+2, 1]
            rk_n_2 = cfg.r[k, j+2, 2]
            rk_n_3 = cfg.r[k, j+2, 3]
            
            dx_j1 = ri_j_1 - rk_j_1; dx_j1 -= L * round(dx_j1 / L)
            dx_j2 = ri_j_2 - rk_j_2; dx_j2 -= L * round(dx_j2 / L)
            dx_j3 = ri_j_3 - rk_j_3; dx_j3 -= L * round(dx_j3 / L)
            d2_j = dx_j1*dx_j1 + dx_j2*dx_j2 + dx_j3*dx_j3

            dx_n1 = ri_n_1 - rk_n_1; dx_n1 -= L * round(dx_n1 / L)
            dx_n2 = ri_n_2 - rk_n_2; dx_n2 -= L * round(dx_n2 / L)
            dx_n3 = ri_n_3 - rk_n_3; dx_n3 -= L * round(dx_n3 / L)
            d2_next = dx_n1*dx_n1 + dx_n2*dx_n2 + dx_n3*dx_n3
            
            dot_prod = dx_j1*dx_n1 + dx_j2*dx_n2 + dx_j3*dx_n3
            
            if sys.U isa HardSpherePotential
                ratio = cao_berne_ratio(sqrt(d2_j), sqrt(d2_next), dot_prod, a_hs, λ, δτ)
                if ratio <= 0.0; return Inf; end
                action -= log(ratio)
            else
                action += (sys.U(sqrt(d2_j)) + sys.U(sqrt(d2_next))) * 0.5 * δτ
            end
        end
    else
        @inbounds for k in 1:N
            cycle_mask[k] && continue
            
            rk_j_1 = get_bead(cfg, k, j, 1, L)
            rk_j_2 = get_bead(cfg, k, j, 2, L)
            rk_j_3 = get_bead(cfg, k, j, 3, L)
            rk_n_1 = get_bead(cfg, k, j+1, 1, L)
            rk_n_2 = get_bead(cfg, k, j+1, 2, L)
            rk_n_3 = get_bead(cfg, k, j+1, 3, L)
            
            dx_j1 = ri_j_1 - rk_j_1; dx_j1 -= L * round(dx_j1 / L)
            dx_j2 = ri_j_2 - rk_j_2; dx_j2 -= L * round(dx_j2 / L)
            dx_j3 = ri_j_3 - rk_j_3; dx_j3 -= L * round(dx_j3 / L)
            d2_j = dx_j1*dx_j1 + dx_j2*dx_j2 + dx_j3*dx_j3

            dx_n1 = ri_n_1 - rk_n_1; dx_n1 -= L * round(dx_n1 / L)
            dx_n2 = ri_n_2 - rk_n_2; dx_n2 -= L * round(dx_n2 / L)
            dx_n3 = ri_n_3 - rk_n_3; dx_n3 -= L * round(dx_n3 / L)
            d2_next = dx_n1*dx_n1 + dx_n2*dx_n2 + dx_n3*dx_n3
            
            dot_prod = dx_j1*dx_n1 + dx_j2*dx_n2 + dx_j3*dx_n3
            
            if sys.U isa HardSpherePotential
                ratio = cao_berne_ratio(sqrt(d2_j), sqrt(d2_next), dot_prod, a_hs, λ, δτ)
                if ratio <= 0.0; return Inf; end
                action -= log(ratio)
            else
                action += (sys.U(sqrt(d2_j)) + sys.U(sqrt(d2_next))) * 0.5 * δτ
            end
        end
    end
    return action
end

"""
    get_interaction_action_segment(cfg, sys, i, j0, j1) -> Float64

Sum of interaction actions for particle i for slices j in j0:j1-1.
"""
function get_interaction_action_segment(cfg::WormConfiguration, sys::System, i::Int, j0::Int, j1::Int)
    S = 0.0
    for j in j0:(j1-1)
        S_slice = get_interaction_action(cfg, sys, i, j)
        if isinf(S_slice)
            return Inf
        end
        S += S_slice
    end
    return S
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
    
    # Sample beads sequentially from j0+1 to j1-1
    for j in (j0 + 1):(j1 - 1)
        Δj_fwd = j1 - j
        Δj_back = 1
        denom = Δj_fwd + Δj_back
        σ = sqrt(2λ * δτ * Δj_back * Δj_fwd / denom)
        
        for d in 1:D
            r_prev_d = get_bead(cfg, i, j - 1, d, L)
            r_end_d = get_bead(cfg, i, j1, d, L)
            
            # Mean: bridge from r_prev to r_end
            r_star_d = (Δj_fwd * r_prev_d + Δj_back * r_end_d) / denom
            
            # Update physical storage (beads 0..M-1)
            # Bead j is at index j+1 in Julia
            cfg.r[i, j + 1, d] = r_star_d + σ * randn()
        end
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
    N, M, D, L = sys.N, sys.M, sys.D, sys.L
    i = rand(1:N)
    cycle = get_cycle(cfg, i)
    Δr = params.r_max .* (2 .* rand(D) .- 1)
    
    # Interaction energy change
    # Optimized: only interactions between cycle and rest change.
    in_cycle = zeros(Bool, N)
    for p in cycle; in_cycle[p] = true; end
    
    S_int_old = 0.0
    for p in cycle, j in 0:M-1
        S_int_old += get_interaction_action_external(cfg, sys, p, j, in_cycle)
    end
    
    # Store old positions and head state
    r_old = copy(cfg.r[cycle, :, :])
    r_c_H_old = copy(cfg.r_c_head)
    w_old = copy(cfg.w[cycle, :])
    
    # Shift
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
    
    S_int_new = 0.0
    for p in cycle, j in 0:M-1
        S_int_new += get_interaction_action_external(cfg, sys, p, j, in_cycle)
    end
    
    if rand() < exp(S_int_old - S_int_new)
        return true
    else
        # Revert
        cfg.r[cycle, :, :] .= r_old
        cfg.r_c_head .= r_c_H_old
        cfg.w[cycle, :] .= w_old
        return false
    end
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
    S_int_old = get_interaction_action_segment(cfg, sys, i, j0, j1)
    
    levy_sample!(cfg, i, j0, j1, λ, δτ, L)
    S_int_new = get_interaction_action_segment(cfg, sys, i, j0, j1)
    
    ΔS = S_int_new - S_int_old
    
    if rand() < exp(-ΔS)
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
    S_int_old = get_interaction_action_segment(cfg, sys, i_H, j0, M)
    
    # Switch to G-sector temporarily to use get_endpoint with r_c_head
    cfg.sector = G_SECTOR
    cfg.i_head = i_H
    cfg.i_tail = cfg.next[i_H]
    
    # Initialize r_c_head and head winding
    for d in 1:D
        x_c, n = _wrap_compact_and_winding(r_new_M[d], L)
        cfg.r_c_head[d] = x_c
        cfg.w[i_H, d] = n
    end
    
    levy_sample!(cfg, i_H, j0, M, λ, δτ, L)
    S_int_new = get_interaction_action_segment(cfg, sys, i_H, j0, M)
    
    # Free propagator ratio
    ρ_new = ρ_free_sp(r_j0, r_new_M, λ, Δj * δτ)
    ρ_old = ρ_free_sp(r_j0, r_old_M, λ, Δj * δτ)
    
    prefactor = params.C * N * (2Δ)^D / V
    A_O = prefactor * (ρ_new / ρ_old) * exp(S_int_old - S_int_new)
    
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
    S_int_old = get_interaction_action_segment(cfg, sys, i_H, j0, M)
    
    # Propose closure: update winding to match r_T
    for d in 1:D
        cfg.w[i_H, d] = round(Int, (r_T[d] - r_tail_0[d]) / L)
    end
    
    # Temporarily switch to Z-sector
    cfg.sector = Z_SECTOR
    
    levy_sample!(cfg, i_H, j0, M, λ, δτ, L)
    S_int_new = get_interaction_action_segment(cfg, sys, i_H, j0, M)
    
    ρ_new = ρ_free_sp(r_j0, r_T, λ, Δj * δτ)
    ρ_old = ρ_free_sp(r_j0, r_old_M, λ, Δj * δτ)
    
    prefactor = V / (params.C * N * (2Δ)^D)
    A_C = prefactor * (ρ_new / ρ_old) * exp(S_int_old - S_int_new)
    
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
    S_int_old = get_interaction_action_segment(cfg, sys, i_H, j0, M)
    
    # Sample new head endpoint
    σ = sqrt(2λ * Δj * δτ)
    r_new_M = r_j0 .+ σ .* randn(D)
    
    for d in 1:D
        x_c, n = _wrap_compact_and_winding(r_new_M[d], L)
        cfg.r_c_head[d] = x_c
        cfg.w[i_H, d] = n
    end
    
    levy_sample!(cfg, i_H, j0, M, λ, δτ, L)
    S_int_new = get_interaction_action_segment(cfg, sys, i_H, j0, M)
    
    if rand() < exp(S_int_old - S_int_new)
        return true
    else
        # Revert
        cfg.r[i_H, (j0+1):M, :] .= r_old_segment
        cfg.w[i_H, :] .= w_old
        cfg.r_c_head .= r_c_head_old
        return false
    end
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
    S_int_old = get_interaction_action_segment(cfg, sys, i_T, 0, j1)
    
    # Sample new bead 0
    σ = sqrt(2λ * Δj * δτ)
    r_new_0 = r_j1 .+ σ .* randn(D)
    cfg.r[i_T, 1, :] .= r_new_0
    
    levy_sample!(cfg, i_T, 0, j1, λ, δτ, L)
    S_int_new = get_interaction_action_segment(cfg, sys, i_T, 0, j1)
    
    if rand() < exp(S_int_old - S_int_new)
        recenter!(cfg, i_T, L)
        return true
    else
        cfg.r[i_T, 1:j1, :] .= r_old_segment
        return false
    end
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
    
    # Identify i* such that next(i*) = i_0 (Spada: new head index)
    # OPTIMIZED: Use prev[i_0] instead of O(N) search.
    i_star = cfg.prev[i_0]
    
    # Acceptance probability A_SW = Σ_P / Σ_0 (Spada Eq. 33)
    # Plus interaction energy difference
    S_int_old = get_interaction_action_segment(cfg, sys, i_0, 0, j_P)
    
    if rand() < min(1.0, Σ_P / Σ_0)  # Initial check to avoid unnecessary work? 
                                    # No, Spada says A_SW = (Σ_P/Σ_0) * exp(ΔU)
        # Actually, let's do it properly
    end
    
    # Store old target bead 0 and segment
    r_old_segment = copy(cfg.r[i_0, 1:j_P, :])
    r_c_H_old = copy(cfg.r_c_head)
    
    # 1. Relocate i_0 bead 0 to the old head position
    cfg.r[i_0, 1, :] .= r_c_H_old
    
    # 2. Relocate head compact coordinate to the old i_0 bead 0 position
    cfg.r_c_head .= r_c_i0_0
    
    # 3. Redraw segment of i_0 from 0 to j_P via staging
    levy_sample!(cfg, i_0, 0, j_P, λ, δτ, L)
    S_int_new = get_interaction_action_segment(cfg, sys, i_0, 0, j_P)
    
    A_SW = (Σ_P / Σ_0) * exp(S_int_old - S_int_new)
    
    if rand() < min(1.0, A_SW)
        # 4. Update permutation (local rewire)
        cfg.next[i_H] = i_0
        cfg.prev[i_0] = i_H
        
        cfg.next[i_star] = i_T
        cfg.prev[i_T] = i_star
        
        # 5. New head index is i_star
        cfg.i_head = i_star
        
        recenter!(cfg, i_0, L)
        return true
    else
        # Revert
        cfg.r[i_0, 1:j_P, :] .= r_old_segment
        cfg.r_c_head .= r_c_H_old
        return false
    end
    return false
end

function energy_thermodynamic(cfg::WormConfiguration, sys::System)
    N, M, D = sys.N, sys.M, sys.D
    λ, δτ, L = sys.λ, sys.τ, sys.L
    
    spring_sum = 0.0
    for i in 1:N
        # FAST PATH: Internal beads
        @inbounds for j in 0:M-2
            d2 = 0.0
            for d in 1:D
                dx = cfg.r[i, j+1, d] - cfg.r[i, j+2, d]
                d2 += dx * dx
            end
            spring_sum += d2
        end
        # SLOW PATH: Endpoint
        d2_end = 0.0
        for d in 1:D
            # Use get_bead to avoid vector allocations from get_endpoint
            dx = cfg.r[i, M, d] - get_bead(cfg, i, M, d, L)
            d2_end += dx * dx
        end
        spring_sum += d2_end
    end
    
    # Interaction term (Spada Eq. A1)
    int_contribution = 0.0
    if N > 1 && !(sys.U isa NullPairPotential)
        a_hs = sys.U isa HardSpherePotential ? sys.U.a : 0.0
        # FAST PATH: Internal beads j in 0:M-2
        for j in 0:M-2
            @inbounds for i in 1:N-1
                # Hoist i
                ri_j_1 = cfg.r[i, j+1, 1]; ri_j_2 = cfg.r[i, j+1, 2]; ri_j_3 = cfg.r[i, j+1, 3]
                ri_n_1 = cfg.r[i, j+2, 1]; ri_n_2 = cfg.r[i, j+2, 2]; ri_n_3 = cfg.r[i, j+2, 3]
                
                for k in i+1:N
                    rk_j_1 = cfg.r[k, j+1, 1]; rk_j_2 = cfg.r[k, j+1, 2]; rk_j_3 = cfg.r[k, j+1, 3]
                    rk_n_1 = cfg.r[k, j+2, 1]; rk_n_2 = cfg.r[k, j+2, 2]; rk_n_3 = cfg.r[k, j+2, 3]
                    
                    dx_j1 = ri_j_1 - rk_j_1; dx_j1 -= L * round(dx_j1 / L)
                    dx_j2 = ri_j_2 - rk_j_2; dx_j2 -= L * round(dx_j2 / L)
                    dx_j3 = ri_j_3 - rk_j_3; dx_j3 -= L * round(dx_j3 / L)
                    d2_j = dx_j1*dx_j1 + dx_j2*dx_j2 + dx_j3*dx_j3

                    dx_n1 = ri_n_1 - rk_n_1; dx_n1 -= L * round(dx_n1 / L)
                    dx_n2 = ri_n_2 - rk_n_2; dx_n2 -= L * round(dx_n2 / L)
                    dx_n3 = ri_n_3 - rk_n_3; dx_n3 -= L * round(dx_n3 / L)
                    d2_next = dx_n1*dx_n1 + dx_n2*dx_n2 + dx_n3*dx_n3
                    
                    dot_prod = dx_j1*dx_n1 + dx_j2*dx_n2 + dx_j3*dx_n3
                    
                    if sys.U isa HardSpherePotential
                        # Skip du_dδτ for Cao-Berne (wrong M-dependence)
                        # Thermodynamic estimator not reliable for pair-product approx
                    else
                        int_contribution += 0.5 * (sys.U(sqrt(d2_j)) + sys.U(sqrt(d2_next)))
                    end
                end
            end
        end
        # SLOW PATH: Endpoint j = M-1
        j = M-1
        for i in 1:N-1
            ri_j_1 = get_bead(cfg, i, j, 1, L); ri_j_2 = get_bead(cfg, i, j, 2, L); ri_j_3 = get_bead(cfg, i, j, 3, L)
            ri_n_1 = get_bead(cfg, i, j+1, 1, L); ri_n_2 = get_bead(cfg, i, j+1, 2, L); ri_n_3 = get_bead(cfg, i, j+1, 3, L)
            for k in i+1:N
                rk_j_1 = get_bead(cfg, k, j, 1, L); rk_j_2 = get_bead(cfg, k, j, 2, L); rk_j_3 = get_bead(cfg, k, j, 3, L)
                rk_n_1 = get_bead(cfg, k, j+1, 1, L); rk_n_2 = get_bead(cfg, k, j+1, 2, L); rk_n_3 = get_bead(cfg, k, j+1, 3, L)
                
                dx_j1 = ri_j_1 - rk_j_1; dx_j1 -= L * round(dx_j1 / L)
                dx_j2 = ri_j_2 - rk_j_2; dx_j2 -= L * round(dx_j2 / L)
                dx_j3 = ri_j_3 - rk_j_3; dx_j3 -= L * round(dx_j3 / L)
                d2_j = dx_j1*dx_j1 + dx_j2*dx_j2 + dx_j3*dx_j3

                dx_n1 = ri_n_1 - rk_n_1; dx_n1 -= L * round(dx_n1 / L)
                dx_n2 = ri_n_2 - rk_n_2; dx_n2 -= L * round(dx_n2 / L)
                dx_n3 = ri_n_3 - rk_n_3; dx_n3 -= L * round(dx_n3 / L)
                d2_next = dx_n1*dx_n1 + dx_n2*dx_n2 + dx_n3*dx_n3
                
                dot_prod = dx_j1*dx_n1 + dx_j2*dx_n2 + dx_j3*dx_n3
                
                if sys.U isa HardSpherePotential
                    # Skip du_dδτ for Cao-Berne (wrong M-dependence)
                else
                    int_contribution += 0.5 * (sys.U(sqrt(d2_j)) + sys.U(sqrt(d2_next)))
                end
            end
        end
    end
    
    # E = D*N/(2*δτ) - spring_sum/(4*λ*δτ^2*M) + 1/M * sum(dU/dδτ)
    E = (D * N / (2 * δτ)) - (spring_sum / (4 * λ * δτ^2 * M)) + (int_contribution / M)
    return E
end

"""
    energy_virial(cfg, sys) -> Float64

Virial energy estimator.
Ref: Spada et al. 2022, Appendix Eq. A2
"""
function energy_virial(cfg::WormConfiguration, sys::System)
    N, M, D = sys.N, sys.M, sys.D
    λ, δτ, β, L = sys.λ, sys.τ, sys.β, sys.L
    
    # Term 2: (R_{M-1} - R_M) · (R_M - R_0)
    term2 = 0.0
    for i in 1:N
        @inbounds for d in 1:D
            # R_{M-1} is internal (index M)
            r_M_minus_1_d = cfg.r[i, M, d]
            # R_M is endpoint
            r_M_d = get_bead(cfg, i, M, d, L)
            # R_0 is internal (index 1)
            r_0_d = cfg.r[i, 1, d]
            term2 += (r_M_minus_1_d - r_M_d) * (r_M_d - r_0_d)
        end
    end
    
    # Term 3: Coordinate derivatives
    term3 = 0.0
    # Term 4: δτ derivatives
    term4 = 0.0
    
    if N > 1 && !(sys.U isa NullPairPotential)
        a_hs = sys.U isa HardSpherePotential ? sys.U.a : 0.0
        
        for j in 0:M-1
            @inbounds for i in 1:N-1
                # Hoist coordinates for particle i
                ri_j_1 = get_bead(cfg, i, j, 1, L); ri_j_2 = get_bead(cfg, i, j, 2, L); ri_j_3 = get_bead(cfg, i, j, 3, L)
                ri_n_1 = get_bead(cfg, i, j+1, 1, L); ri_n_2 = get_bead(cfg, i, j+1, 2, L); ri_n_3 = get_bead(cfg, i, j+1, 3, L)
                ri_0_1 = get_bead(cfg, i, 0, 1, L);   ri_0_2 = get_bead(cfg, i, 0, 2, L);   ri_0_3 = get_bead(cfg, i, 0, 3, L)
                
                for k in i+1:N
                    rk_j_1 = get_bead(cfg, k, j, 1, L); rk_j_2 = get_bead(cfg, k, j, 2, L); rk_j_3 = get_bead(cfg, k, j, 3, L)
                    rk_n_1 = get_bead(cfg, k, j+1, 1, L); rk_n_2 = get_bead(cfg, k, j+1, 2, L); rk_n_3 = get_bead(cfg, k, j+1, 3, L)
                    rk_0_1 = get_bead(cfg, k, 0, 1, L);   rk_0_2 = get_bead(cfg, k, 0, 2, L);   rk_0_3 = get_bead(cfg, k, 0, 3, L)
                    
                    dx_j1 = ri_j_1 - rk_j_1; dx_j1 -= L * round(dx_j1 / L)
                    dx_j2 = ri_j_2 - rk_j_2; dx_j2 -= L * round(dx_j2 / L)
                    dx_j3 = ri_j_3 - rk_j_3; dx_j3 -= L * round(dx_j3 / L)
                    d2_j = dx_j1*dx_j1 + dx_j2*dx_j2 + dx_j3*dx_j3

                    dx_n1 = ri_n_1 - rk_n_1; dx_n1 -= L * round(dx_n1 / L)
                    dx_n2 = ri_n_2 - rk_n_2; dx_n2 -= L * round(dx_n2 / L)
                    dx_n3 = ri_n_3 - rk_n_3; dx_n3 -= L * round(dx_n3 / L)
                    d2_next = dx_n1*dx_n1 + dx_n2*dx_n2 + dx_n3*dx_n3
                    
                    dot_prod = dx_j1*dx_n1 + dx_j2*dx_n2 + dx_j3*dx_n3
                    
                    if sys.U isa HardSpherePotential
                        r_j = sqrt(d2_j)
                        r_n = sqrt(d2_next)
                        du_dr, du_dr_prime, du_dcos, du_dδτ = cao_berne_derivatives(r_j, r_n, dot_prod, a_hs, λ, δτ)
                        
                        # Handle overlaps or forbidden regions
                        if isinf(du_dr)
                            return Inf
                        end
                        
                        # NOTE: Skip term4 for Cao-Berne. The du_dδτ derivative has wrong 
                        # M-dependence for pair-product approximations (decreases as M→∞
                        # instead of converging). The virial term (term3) captures interaction.
                        # term4 += du_dδτ
                        
                        cos_θ = dot_prod / (r_j * r_n)
                        cos_θ = clamp(cos_θ, -1.0, 1.0)
                        
                        # Grad wrt r_ij (start of slice j)
                        # g_i_j = (du_dr - du_dcos * cos_θ / r_j) * (dx_j / r_j) + (du_dcos / r_j) * (dx_next / r_n)
                        inv_rj = 1.0/r_j; inv_rn = 1.0/r_n
                        f1 = (du_dr - du_dcos * cos_θ * inv_rj) * inv_rj
                        f2 = du_dcos * inv_rj * inv_rn
                        
                        g1_1 = f1*dx_j1 + f2*dx_n1; g1_2 = f1*dx_j2 + f2*dx_n2; g1_3 = f1*dx_j3 + f2*dx_n3
                        
                        # Term 3 contribution for start of slice
                        term3 += (ri_j_1 - ri_0_1 - rk_j_1 + rk_0_1) * g1_1
                        term3 += (ri_j_2 - ri_0_2 - rk_j_2 + rk_0_2) * g1_2
                        term3 += (ri_j_3 - ri_0_3 - rk_j_3 + rk_0_3) * g1_3
                        
                        # Grad wrt r_ij_next (end of slice j)
                        # g_i_n = (du_dr_prime - du_dcos * cos_θ / r_n) * (dx_next / r_n) + (du_dcos / r_n) * (dx_j / r_j)
                        f3 = (du_dr_prime - du_dcos * cos_θ * inv_rn) * inv_rn
                        f4 = du_dcos * inv_rn * inv_rj
                        
                        g2_1 = f3*dx_n1 + f4*dx_j1; g2_2 = f3*dx_n2 + f4*dx_j2; g2_3 = f3*dx_n3 + f4*dx_j3
                        
                        # Term 3 contribution for end of slice
                        term3 += (ri_n_1 - ri_0_1 - rk_n_1 + rk_0_1) * g2_1
                        term3 += (ri_n_2 - ri_0_2 - rk_n_2 + rk_0_2) * g2_2
                        term3 += (ri_n_3 - ri_0_3 - rk_n_3 + rk_0_3) * g2_3
                    else
                        term4 += 0.5 * (sys.U(sqrt(d2_j)) + sys.U(sqrt(d2_next)))
                    end
                end
            end
        end
    end
    
    # Spada Eq. A2: E_vir/N = D/(2β) + term2/(4λδτ²NM) + term3/(2βN) + term4/(NM)
    # term2 summed over N particles, term3 and term4 summed over pairs and slices
    E = (D * N / (2 * β)) + (term2 / (4 * λ * δτ^2 * M)) + (term3 / (2 * β)) + (term4 / M)
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
