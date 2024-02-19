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

# Central path structure
# Contains a ~linked list with next and prev
# Objects are mutable vectors, so the struct is fixed in memory, but contents mutable
mutable struct Path
    r::Array{Float64,3}
    next::Array{Int64}
    prev::Array{Int64}

    function Path(S::System)
        r=rand(S.nparticles,S.nbeads,S.spatialdimensions)
        next=reshape(repeat(2:S.nbeads+1, S.nparticles), S.nparticles, S.nbeads) 
        prev=reshape(repeat(0:S.nbeads-1, S.nparticles), S.nparticles, S.nbeads)
        new(r,next,prev)
    end
end

