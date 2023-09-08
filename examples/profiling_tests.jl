# using Plots
#pyplot()

using LinearAlgebra
using SchrodingerFE
using SparseArrays

ne = 3; # Nb of particles
L = 10.0; #Interval length
N = 10; #Nb of discretization points of the interval [-L,L]
α = 1. #parameter in the laplace operator

# Spatial mesh
xx = collect(range(-L, L, length=N + 1))[2:end-1]

# External potential
a = 1.0
b = -1.0
vext(x) = -(1.0 ./ sqrt.((x - a) .^ 2 .+ 1)
            .+
            1.0 ./ sqrt.((x - b) .^ 2 .+ 1)
)

# Electron-electron interaction
vee(x) = 1.0 ./ sqrt.(x .^ 2 .+ 1)

# For P1 Finite Elements
ham = ham1d(L, N; alpha_lap=α, vext=vext, vee);

# For P2 Finite Elements 
# ham = ham1d(L, N; vext, vee, nx=4, ny=4, element="P2");


# Solve the eigenvalue problem with different methods
E_FCIsparse, wf_FCIsparse = WaveFunction(ne, ham, "FCI_sparse"; maxiter=50, kdim=50)

# myfunc() = WaveFunction(ne, ham, "FCI_sparse"; maxiter=50, kdim=50)

# # using Profile
# # Profile.Allocs.@profile myfunc()
# # Profile.Allocs.print()
# using BenchmarkTools
# @btime myfunc()
