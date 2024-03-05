# types.jl - base types of computation 

using StaticArrays

struct System
    nbeads ::Int64
    nparticles ::Int64
    mass ::Float64
    spatialdimensions ::Int64

    function System(nbeads::Integer, nparticles::Integer; mass=1.0, spatialdimensions=3)
        new(nparticles, nbeads, mass, spatialdimensions)
    end
end
Base.show(io::IO, s::System) = print(io, "PIMC system with $(s.nbeads) beads and $(s.nparticles) particles. \nParticle mass of $(s.mass). $(s.spatialdimensions) spatial dimensions.")

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
        r=rand(S.nparticles,S.nbeads,S.spatialdimensions)
        list=reshape(repeat(1:S.nparticles, S.nbeads), S.nparticles, S.nbeads) 
        new(r,list,list)
    end
end

function Base.show(io::IO, p::Path)
    print(io, "Path $(p.r) \n Next $(p.next) \n Prev $(p.prev)")
end

