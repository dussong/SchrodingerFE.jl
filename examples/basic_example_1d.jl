using Plots
#pyplot()

using LinearAlgebra
using SchrodingerFE
using SparseArrays

ne = 2; # Nb of particles
L = 10.0; #Interval length
N = 80; #Nb of discretization points of the interval [-L,L]
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
E_FCIfull, wf_FCIfull = WaveFunction(ne, ham, "FCI_full"; maxiter=50, kdim=50)
E_FCIsparse, wf_FCIsparse = WaveFunction(ne, ham, "FCI_sparse"; maxiter=50, kdim=50)
E_CDFCIsparse, wf_CDFCIsparse = WaveFunction(ne, ham, "CDFCI_sparse"; max_iter=1000, k=500)
E_SCIsparse, wf_SCIsparse = WaveFunction(ne, ham, "selected_CI_sparse";max_iter=1000, k=500, num=100, M=2)

# To convert the sparse WF as a full WF
wf_SCIsparse_converted = WaveFunction_full(ne, Array(wf_SCIsparse.wfP))

@show norm(E_FCIfull - E_FCIsparse)
@show norm(E_FCIfull - E_CDFCIsparse)
@show norm(E_FCIfull - E_SCIsparse)


# Compute density
ρ_FCIfull = SchrodingerFE.density(wf_FCIfull, ham)
ρ_FCIfull = (ρ_FCIfull * N * ne) / (norm(ρ_FCIfull, 1) * 2L)

ρ_FCIsparse = SchrodingerFE.density(wf_FCIsparse, ham)
ρ_FCIsparse = (ρ_FCIsparse * N * ne) / (norm(ρ_FCIsparse, 1) * 2L)

ρ_CDFCIsparse = SchrodingerFE.density(wf_CDFCIsparse, ham)
ρ_CDFCIsparse = (ρ_CDFCIsparse * N * ne) / (norm(ρ_CDFCIsparse, 1) * 2L)

ρ_SCIsparse = SchrodingerFE.density(wf_SCIsparse, ham)
ρ_SCIsparse = (ρ_SCIsparse * N * ne) / (norm(ρ_SCIsparse, 1) * 2L)

@show norm(ρ_FCIfull - ρ_CDFCIsparse)
@show norm(ρ_FCIfull - ρ_SCIsparse)


P = plot(xx, ρ_FCIfull, label="FCI", xlims=(-L, L), linewidth=5, tickfontsize=16)
P = plot!(P, xx, ρ_CDFCIsparse, label="CDFCIsparse", ls=:dash, linewidth=3)
P = plot!(P, xx, ρ_SCIsparse, label="SCIsparse", ls=:dashdot, linewidth=2)
display(P)


# Cmpute pair density
ρ2_FCIfull = pair_density(wf_FCIfull, ham)
ρ2_FCIfull = (ρ2_FCIfull * N^2 * binomial(ne, 2)) / (norm(ρ2_FCIfull, 1) * 4 * L^2)
ρ2_FCIfull = reshape(ρ2_FCIfull, N - 1, N - 1)

ρ2_FCIsparse = pair_density(wf_FCIsparse, ham)
ρ2_FCIsparse = (ρ2_FCIsparse * N^2 * binomial(ne, 2)) / (norm(ρ2_FCIsparse, 1) * 4 * L^2)
ρ2_FCIsparse = reshape(ρ2_FCIsparse, N - 1, N - 1)

ρ2_CDFCIsparse = pair_density(wf_CDFCIsparse, ham)
ρ2_CDFCIsparse = (ρ2_CDFCIsparse * N^2 * binomial(ne, 2)) / (norm(ρ2_CDFCIsparse, 1) * 4 * L^2)
ρ2_CDFCIsparse = reshape(ρ2_CDFCIsparse, N - 1, N - 1)

ρ2_SCIsparse = pair_density(wf_SCIsparse, ham)
ρ2_SCIsparse = (ρ2_SCIsparse * N^2 * binomial(ne, 2)) / (norm(ρ2_SCIsparse, 1) * 4 * L^2)
ρ2_SCIsparse = reshape(ρ2_SCIsparse, N - 1, N - 1)

@show norm(ρ2_FCIfull - ρ2_CDFCIsparse)
@show norm(ρ2_FCIfull - ρ2_SCIsparse)

P2 = plot(xx, xx, ρ2_FCIfull, c=:viridis,
   aspect_ratio=:equal, xlims=(-L, L),
   fill=:true)

P2 = plot(xx, xx, ρ2_FCIsparse, c=:viridis,          
          aspect_ratio=:equal, xlims=(-L, L), 
          fill=:true)

P2 = plot(xx, xx, ρ2_CDFCIsparse, c=:viridis,
         aspect_ratio=:equal, xlims=(-L, L),
         fill=:true)


P2 = plot(xx, xx, ρ2_FCIsparse, c=:viridis,
         aspect_ratio=:equal, xlims=(-L, L),
         fill=:true)

# Compute the partial pair density
rho2_spin_s = pair_density_spin(1,1, wf_FCIfull, ham)+pair_density_spin(0,0, wf_FCIfull, ham)

rho2_spin_as = pair_density_spin(1, 0, wf_FCIfull, ham) + pair_density_spin(0, 1, wf_FCIfull, ham)

rho2 = pair_density(wf_FCIfull, ham)

# Spatial antisymmetric part of pair density
P3 = contour(xx, xx, rho2_spin_s, c=:viridis,
            aspect_ratio=:equal, xlims=(-L, L),
            fill=:true)

# Spatial symmetric part of pair density            
P3 = contour(xx, xx, rho2_spin_as, c=:viridis,
   aspect_ratio=:equal, xlims=(-L, L),
   fill=:true)

# Total pair density
P3 = contour(xx, xx, rho2, c=:viridis,
   aspect_ratio=:equal, xlims=(-L, L),
   fill=:true)