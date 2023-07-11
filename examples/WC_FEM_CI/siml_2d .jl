using PairDensities
using SparseArrays
using Plots, LaTeXStrings, DelimitedFiles
using Plots.PlotMeasures
using LinearAlgebra

#-----------------------------------------------------------
# 3ne for HF
#-----------------------------------------------------------
ne = 3;
L = 5.0;
N = 21 #31;
h = 2L / N;
a = 0.8;
b = 0.2;
vext(x, y) = b * (x .^ 2 .+ y .^ 2)
vee(x, y) = 1 / sqrt(1e-3 + x^2 + y^2)
ham = ham2d(L, N; alpha_lap=a, vext, vee);
println("-------------------------------------------------------------------------------")
println("\n $(ne)-particle", "  N = $(N)", "  alpha = $(a)")
@time wfhf, U, Hv, Mv = HF(ne, ham);
maxt = 500 #5000
k = 100 #6000
@time y, c, citer = CDFCI_matfree_block_nbd(wfhf, ham, k, Hv, Mv; max_iter=maxt, tol=1e-8);
println("  density:")
x = collect(range(-L, L, length=N + 1));
combiter = map(x -> num2seq_ns(2ham.C.n, ne, x), citer.nzind)
@time n1 = density_hf_full([x,x], ne, U, combiter, citer.nzval, wfhf.combBasis, wfhf.val, ham);
n1 = (n1 * ne * N^2) / (norm(n1, 1) * 4 * L^2)
PH = contour(x, x, n1, c=:darktest, aspect_ratio=:equal, size=(650, 600), right_margin=10mm, colorbar=:true, fill=:true, tickfontsize=16, colorbar_tickfontsize=16)

#-----------------------------------------------------------
# 3ne for SCE
#-----------------------------------------------------------
ne = 3
L = 5.0;
N = 21 #31;
h = 2L / N;
a = 0.1;
b = 0.2;
vext(x, y) = b * (x .^ 2 .+ y .^ 2)
vee(x, y) = 1 / sqrt(1e-3 + x^2 + y^2)
ham = ham2d(L, N; alpha_lap=a, vext, vee);
println("\n $(ne)-particle", "  N = $(N)", "  alpha = $(a)", "  k = $(k)")
Esp, wfsp = WaveFunction(ne, ham, "selected_CI_sparse"; max_iter=500, k=100, num=500, tol=1e-8, M=[1,1])
println("dof:$(length(wfsp.val))")
x = collect(range(-L, L, length=N + 1))[2:end-1]
@time n1 = PairDensities.density(wfsp, ham)
PS = contour(x, x, n1, c=:darktest, aspect_ratio=:equal,size=(650,600),right_margin=10mm, colorbar=:true, fill=:true, tickfontsize=16, colorbar_tickfontsize=16)
