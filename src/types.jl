# types.jl - base types of computation 

using StaticArrays

struct System
    nbeads ::Int64
    nparticles ::Int64
    mass ::Float64
    
    spatialdimensions ::Int64
    β ::Float64
    potential ::Function
    λ ::Float64 
    τ ::Float64   # imaginary time step
   
    L ::Float64 # box length; yes, this is definitely going to come back and bite me. Assumes square / cubic box for now.

    function System(nbeads::Integer, nparticles::Integer; 
                   mass=1.0, 
                   spatialdimensions=3, 
                   β=100.0, 
                   potential=Harmonic,
                   λ=1.0,  # default thermal wavelength
                   τ=β/nbeads,
                   L=1.0)  # default imaginary time step
        new(nparticles, nbeads, mass, spatialdimensions, β, potential, λ, τ, L)
    end
end

Base.show(io::IO, s::System) = print(io, 
    "PIMC system with $(s.nbeads) beads and $(s.nparticles) particles. \n" *
    "β=$(s.β), τ=$(s.τ). Potential=$(s.potential)\n" *
    "Particle mass of $(s.mass), λ=$(s.λ). $(s.spatialdimensions) spatial dimensions." * 
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
        r=randn(S.nparticles,S.nbeads,S.spatialdimensions)
        list=reshape(repeat(1:S.nparticles, S.nbeads), S.nparticles, S.nbeads) 
        new(r,list,list)
    end
end

function Base.show(io::IO, p::Path)
    print(io, "Path $(p.r) \n Next $(p.next) \n Prev $(p.prev)")
end

