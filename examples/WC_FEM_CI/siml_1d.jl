using PairDensities
using SparseArrays
using Plots, LaTeXStrings, DelimitedFiles
using Plots.PlotMeasures
using LinearAlgebra

#-----------------------------------------------------------
# 4ne for HF
#-----------------------------------------------------------
ne = 4; # Nb of particles
L = 5.0; #Interval length
N = 49; #Nb of discretization points of the interval [-L,L]
h = 2L / N;
a = 1.0; #Parameter in the kinetic potential
b = 0.5; #Parameter in the external potential
vee(x) = @. 1 / sqrt(1e-3 + x^2)
vext(x) = @. b * x^2
ham = ham1d(L, N; alpha_lap=a, vext, vee);
println("-------------------------------------------------------------------------------")
println("\n $(ne)-particle", "  N = $(N)", "  alpha = $(a)")
@time Esp, wf1 = WaveFunction(ne, ham, "CDFCI_sparse"; max_iter=1000, k=500, tol=1e-8)
# plot density
wf2 = WaveFunction_full(ne, Array(wf1.wfP))
x = collect(range(-L, L, length=N + 1))[2:end-1]
@time n1 = PairDensities.density(wf2, ham)
n1 = (n1 * N * ne) / (norm(n1, 1) * 2L);
cols = collect(palette(:tab10))
markers = filter((m -> begin
        m in Plots.supported_markers()
    end), Plots._shape_keys)
P1H = plot(xaxis=(L"x"), yaxis=(L"\rho(x)"), legend=:topright, grid=:off, box=:on, title=L"N=%$ne",
    size=(900, 770), ls=:solid, tickfontsize=20, legendfontsize=18, guidefontsize=26, titlefontsize=30, left_margin=4mm)
plot!(P1H, x, n1, lw=3, ls=:solid, label=L"\alpha = %$a")
# plot pair density
@time n2 = pair_density(wf2, ham)
n2 = reshape(n2, N - 1, N - 1)
n2 = (n2 * N^2 * binomial(ne, 2)) / (norm(n2, 1) * 4 * L^2);
P2H = contour(x, x, n2, title=L"\alpha=%$a", c=:darktest, aspect_ratio=:equal, size=(650, 600), right_margin=10mm, colorbar=:true, fill=:true, tickfontsize=16, colorbar_tickfontsize=16)



#-----------------------------------------------------------
# 4ne for SCE
#-----------------------------------------------------------
ne = 4;
L = 5.0;
N = 49;
h = 2L / N;
a = 0.1;
b = 0.5;
vext(x) = b * x^2
vee(x) = @. 1 / sqrt(1e-3 + x^2)
ham = ham1d(L, N; alpha_lap=a, vext, vee);
println("-------------------------------------------------------------------------------")
println("\n $(ne)-particle", "  N = $(N)", "  alpha = $(a)")
Esp, wfsp = WaveFunction(ne, ham, "selected_CI_sparse"; max_iter=1000, k=500, num=20, tol=1e-8, M=1)
# plot density
x = collect(range(-L, L, length=N + 1))[2:end-1];
@time n1 = PairDensities.density(wfsp, ham)
n1 = (n1 * N * ne) / (norm(n1, 1) * 2L);
P1S = plot(xaxis=(L"x"), yaxis=(L"\rho(x)"), legend=:topright, grid=:off, box=:on, title=L"N=%$ne",
    size=(900, 770), ls=:solid, tickfontsize=20, legendfontsize=18, guidefontsize=26, titlefontsize=30, left_margin=4mm)
plot!(P1S, x, n1, lw=3, ls=:solid, label=L"\alpha = %$a")
# plot pair density
#@time n2 = pair_density(wfsp, ham)
wff = WaveFunction_full(ne, Array(wfsp.wfP))
@time n2 = pair_density(wff, ham)
n2 = (n2 * N^2 * binomial(ne, 2)) / (norm(n2, 1) * 4 * L^2);
P2S = contour(x, x, reshape(n2, N - 1, N - 1), title=L"\alpha=%$a", c=:darktest, aspect_ratio=:equal, size=(650, 600), right_margin=10mm, colorbar=:true, fill=:true, tickfontsize=16, colorbar_tickfontsize=16)
