# using Plots
#pyplot()
using Profile
using BenchmarkTools
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
@time E_FCIsparse2, wf_FCIsparse2 = WaveFunction(ne, ham, "FCI_sparse2"; maxiter=50, kdim=50)

@time E_FCIsparse, wf_FCIsparse = WaveFunction(ne, ham, "FCI_sparse"; maxiter=50, kdim=50)
# myfunc() = WaveFunction(ne, ham, "FCI_sparse"; maxiter=50, kdim=50)

# # using Profile
# # Profile.Allocs.@profile myfunc()
# # Profile.Allocs.print()
# using BenchmarkTools
# @btime myfunc()

# N = 10
# A = rand(N,N)
# A = A'*A

# function allocate(N)
#    phi = zeros(N)
#    return phi
# end

# x0 = ones(N)


# phi = allocate(N)




# function Ax2!(x, phi)
#    phi .= 0 
#    return A*x, x
# end

# using KrylovKit
# kdim = 5
# maxiter = 100
# E, Ψt, cvinfo = geneigsolve(Ax2!, x0, 1, :SR; krylovdim=kdim, maxiter=maxiter, issymmetric=true,
# isposdef=true)

# function 


# M_Ψ(Ψ::Array{Float64,1}) = ham_free_tensor!(
#    ne, N, Ψ, ham.AΔ, ham.AV, ham.C, ham.Bee,
#    Φh, Φm, combBasis; 
#    alpha_lap=ham.alpha_lap)

# E, Ψt, cvinfo = geneigsolve(M_Ψ, x0, 1, :SR; krylovdim=kdim, maxiter=maxiter, issymmetric=true,
#    isposdef=true)