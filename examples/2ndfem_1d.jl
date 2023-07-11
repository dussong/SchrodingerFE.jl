using PairDensities
using KrylovKit
using SparseArrays
using Plots
using LinearAlgebra

# H2
ne = 2
N = 25
L = 8.0;
vext(x) = -1 / sqrt(1 + (x + 0.7)^2) - 1 / sqrt(1 + (x - 0.7)^2);
vee(x) = 1 / sqrt(1 + x^2);
ham = ham1d(L, N; vext, vee, nx=4, ny=4, element="P2");

#generate wavefunction
E, wf = WaveFunction(ne, ham, "FCI_sparse"; maxiter=30, kdim=10)

#plot density 
x = collect(range(-L, L, length=N + 1))[2:end-1];
n1 = PairDensities.density(wf, ham)
n1 = (n1 * N * ne) / (norm(n1, 1) * 2L)
cols = collect(palette(:tab10))
P = plot(x, n1, lw=3, ls=:solid)

#plot pair density
n2 = pair_density(wf, ham)
n2 = (n2 * N^2 * binomial(ne, 2)) / (norm(n2, 1) * 4 * L^2)
P2 = contour(x, x, reshape(n2, N - 1, N - 1), c=:darktest, aspect_ratio=:equal,
    colorbar=:true, fill=:true, tickfontsize=16, colorbar_tickfontsize=16)

# He
ne = 2
N = 20
L = 8.0;
vext(x) = -2 / sqrt(1 + x^2);
vee(x) = 1 / sqrt(1 + x^2);
ham = ham1d(L, N; vext, vee, nx=4, ny=4, element="P2");

#generate wavefunction
E, wf = WaveFunction(ne, ham, "FCI_sparse"; maxiter=30, kdim=10)

#plot density 
x = collect(range(-L, L, length=N + 1))[2:end-1];
n1 = PairDensities.density(wf, ham)
n1 = (n1 * N * ne) / (norm(n1, 1) * 2L)
P = plot(x, n1, lw=3, ls=:solid)

#plot pair density
n2 = pair_density(wf, ham)
n2 = (n2 * N^2 * binomial(ne, 2)) / (norm(n2, 1) * 4 * L^2)
P2 = contour(x, x, reshape(n2, N - 1, N - 1), c=:darktest, aspect_ratio=:equal,
    colorbar=:true, fill=:true, tickfontsize=16, colorbar_tickfontsize=16)


