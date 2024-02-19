# Initially targetting 1D system; potential + etc. on [0,1]

function sampledV(V, N=50)
    discreteV=[ V(i/(N-1) - 1/2) for i=0:(N-1) ]
    return discreteV
end


