using SchrodingerFE
using KrylovKit
using SparseArrays
using Plots

# H2
ne = 2
N = 25
L = 8.0;
vext(x) = -1 / sqrt(1 + (x + 0.7)^2) - 1 / sqrt(1 + (x - 0.7)^2);
vee(x) = 1 / sqrt(1 + x^2);
ham = ham1d(L, N; vext, vee, nx=4, ny=4, element="P2");

#generate wavefunction
E, wf = WaveFunction(ne, ham, FCI_sparse(); maxiter=30, kdim=10)

#plot density 
xx = collect(range(-L, L, length=N + 1))[2:end-1];
n1 = SchrodingerFE.density(wf, ham)
n1 = (n1 * N * ne) / (norm(n1, 1) * 2L)
P = plot(xx, n1, xlims=(-L, L), linewidth=5, tickfontsize=16)


#plot pair density
n2 = pair_density(wf, ham)
n2 = (n2 * N^2 * binomial(ne, 2)) / (norm(n2, 1) * 4 * L^2)
P2 = contour(xx, xx, n2, c=:viridis, aspect_ratio=:equal, xlims=(xx[1], xx[end]),fill=:true)
