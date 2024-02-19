# types.jl - base types of computation 

using StaticArrays

struct System
    nbeads ::Int64
    nparticles ::Int64
    mass ::Float64
    spatialdimensions ::Int64

    p::Array{Float64,3}

    function System(nbeads::Integer, nparticles::Integer; mass=1.0, spatialdimensions=3)
        p=rand(nparticles, nbeads, spatialdimensions)
        new(nparticles, nbeads, mass, spatialdimensions, p)
    end
end

# Central path structure
# Contains a ~linked list with next and prev
# Objects are mutable vectors, so the struct is fixed in memory, but contents mutable
mutable struct Path
    r::Array{Float64,3}
    next::Array{Int64}
    prev::Array{Int64}

    function Path(nparticles::Integer, nbeads::Integer=1; spatialdimensions=3)
        r=rand(nparticles,nbeads,spatialdimensions)
        next=reshape(repeat(2:nbeads+1, nparticles), nparticles, nbeads) 
        prev=reshape(repeat(0:nbeads-1, nparticles), nparticles, nbeads)
        new(r,next,prev)
    end
end

