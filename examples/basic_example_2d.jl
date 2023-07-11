using Plots, Plots.PlotMeasures
#pyplot()

using LinearAlgebra
using PairDensities
using SparseArrays

ne = 2; # Nb of particles
Lx = 5.0; #x interval length 
Ly = 4.0; #y interval length 
Nx = 10; #Nb of discretization points of the interval [-Lx,Lx]
Ny = 8; #Nb of discretization points of the interval [-Ly,Ly]
L = [Lx, Ly]
N = [Nx, Ny]
α = 0.1 #parameter in the laplace operator

xx = collect(range(-Lx, Lx; length=Nx + 1))[2:end-1]
yy = collect(range(-Ly, Ly; length=Ny + 1))[2:end-1]

# External potential
b = 0.1;
vext(x, y) = b * (x .^ 2 .+ y .^ 2)

# Electron-electron interaction
vee(x, y) = 1 / sqrt(1e-3 + x^2 + y^2)

ham = ham2d(L, N; alpha_lap=α, vext, vee);

# Solve the eigenvalue problem
E_FCIfull, wf_FCIfull = WaveFunction(ne, ham, "FCI_full"; maxiter=50, kdim=50)
E_FCIsparse, wf_FCIsparse = WaveFunction(ne, ham, "FCI_sparse"; maxiter=50, kdim=50)
E_CDFCIsparse, wf_CDFCIsparse = WaveFunction(ne, ham, "CDFCI_sparse"; max_iter=1000, k=500)
E_SCIsparse, wf_SCIsparse = WaveFunction(ne, ham, "selected_CI_sparse"; max_iter=1000, k=500, num=100)
#wf_SCIsparse_converted = WaveFunction_full(ne, Array(wf_SCIsparse.wfP))

@show norm(E_FCIfull - E_CDFCIsparse)
@show norm(E_FCIfull - E_SCIsparse)

# Compute density
ρ_FCIfull = PairDensities.density(wf_FCIfull, ham)
ρ_FCIfull = (ρ_FCIfull * Nx * Ny * ne) / (norm(ρ_FCIfull, 1) * 4 * Lx * Ly)
ρ_FCIfull = reshape(ρ_FCIfull, Ny - 1, Nx - 1)

ρ_FCIsparse = PairDensities.density(wf_FCIsparse, ham)
ρ_FCIsparse = (ρ_FCIsparse * Nx * Ny * ne) / (norm(ρ_FCIsparse, 1) * 4 * Lx * Ly)
ρ_FCIsparse = reshape(ρ_FCIsparse, Ny - 1, Nx - 1)

ρ_CDFCIsparse = PairDensities.density(wf_CDFCIsparse, ham)
ρ_CDFCIsparse = (ρ_CDFCIsparse * Nx * Ny * ne) / (norm(ρ_CDFCIsparse, 1) * 4 * Lx * Ly)

ρ_SCIsparse = PairDensities.density(wf_SCIsparse, ham)
ρ_SCIsparse = (ρ_SCIsparse * Nx * Ny * ne) / (norm(ρ_SCIsparse, 1) * 4 * Lx * Ly)

@show norm(ρ_FCIfull - ρ_CDFCIsparse)
@show norm(ρ_FCIfull - ρ_SCIsparse)


P1 = contour(xx, yy, ρ_FCIfull, c=:darktest, aspect_ratio=:equal, size=(650, 600), right_margin=10mm, colorbar=:true, fill=:true, tickfontsize=16, colorbar_tickfontsize=16)

P2 = contour(xx, yy, ρ_CDFCIsparse, c=:darktest, aspect_ratio=:equal, size=(650, 600), right_margin=10mm, colorbar=:true, fill=:true, tickfontsize=16, colorbar_tickfontsize=16)

P3 = contour(xx, yy, ρ_SCIsparse, c=:darktest, aspect_ratio=:equal, size=(650, 600), right_margin=10mm, colorbar=:true, fill=:true, tickfontsize=16, colorbar_tickfontsize=16)


