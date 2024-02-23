# Definition of some simple testing potentials.

Harmonic(x; k=1)=k*sum(x.^2)

DoubleWell(x; A=10, B=1)=sum(A*x.^4 - B*x.^2)

