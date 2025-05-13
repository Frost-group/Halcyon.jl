# types.jl - base types of computation 

using StaticArrays

struct System
    M ::Int64
    N ::Int64
    m ::Float64
    
    D ::Int64
    β ::Float64
    potential ::Function
    λ ::Float64 
    τ ::Float64   # imaginary time step
   
    L ::Float64 # box length; yes, this is definitely going to come back and bite me. Assumes square / cubic box for now.

    function System(M::Integer, N::Integer; 
                   m=1.0, 
                   D=3, 
                   β=100.0, 
                   potential=Harmonic,
                   λ=1.0,  # default thermal wavelength
                   τ=β/M,
                   L=1.0)  # default imaginary time step
        new(N, M, m, D, β, potential, λ, τ, L)
    end
end

Base.show(io::IO, s::System) = print(io, 
    "PIMC system with $(s.M) beads and $(s.N) particles. \n" *
    "β=$(s.β), τ=$(s.τ). Potential=$(s.potential)\n" *
    "Particle m of $(s.m), λ=$(s.λ). $(s.D) spatial dimensions." * 
    "Box length $(s.L).")

# Central path structure
# Contains a ~linked list with next and prev; to allow for sampled particle exchange
# Contains a ~linked list with next and prev
# Intent: Rewrite (?) this with StaticArrays MVectors; for
#  so the struct can be constant in memory, but contents mutable
mutable struct Path
    r::Array{Float64,3}
    next::Array{Int64}
    prev::Array{Int64}

    function Path(S::System)
        r=randn(S.N,S.M,S.D)
        list=reshape(repeat(1:S.N, S.M), S.N, S.M) 
        new(r,list,list)
    end
end

function Base.show(io::IO, p::Path)
    print(io, "Path $(p.r) \n Next $(p.next) \n Prev $(p.prev)")
end

