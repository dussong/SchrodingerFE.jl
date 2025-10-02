using Plots
#pyplot()

using LinearAlgebra
using SchrodingerFE
using SparseArrays

using Combinatorics

ne = 2; # Nb of particles
L = 10.0; #Interval length
N = 80; #Nb of discretization points of the interval [-L,L]
α = 1. #parameter in the laplace operator

# Spatial mesh
xx = collect(range(-L, L, length=N + 1))[2:end-1]

# External potential
a = 2.0
b = -2.0
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
E_FCIfull, wf_FCIfull = WaveFunction(ne, ham, "FCI_full"; maxiter=50, kdim=50)

plot(xx,SchrodingerFE.density(wf_FCIfull, ham))
heatmap(xx,xx,SchrodingerFE.one_body_DM_coef(ne, wf_FCIfull.wf, ham.C))