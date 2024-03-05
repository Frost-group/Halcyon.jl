using Halcyon

using Test

include("potentials.jl")

@assert 1==1

const T=300
const tau=0.5
nbeads=round(Int,1/(tau*T)) # ??? 

nbeads=4
nparticles=2
spatialdimensions=1

Halcyon.Path(nbeads, nparticles,  spatialdimensions=spatialdimensions)

