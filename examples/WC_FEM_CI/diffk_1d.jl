using PairDensities
using SparseArrays
using Plots, LaTeXStrings
using DelimitedFiles
using Plots.PlotMeasures

#compare energy errors and dof with different k
# 4 particles
ne = 4; # Nb of particles
L = 5.0; #Interval length
N = 49; #Nb of discretization points of the interval [-L,L]
h = 2L / N;
a = 0.1; #Parameter in the kinetic potential
b = 0.5; #Parameter in the external potential
vext(x) = b * x^2
vee(x) = @. 1 / sqrt(1e-3 + x^2)
ham = ham1d(L, N; alpha_lap=a, vext);
E = 8.229757675996822 # reference ground state energy of alpha = 0.1, beta = 0.5
println("-------------------------------------------------------------------------------")
println("\n $(ne)-particle", "  N = $(N)", "  alpha = $(a)")
# generate SCE initial state
d = InitPT(ne, ham; num=10)
r0 = [round.(Int, (d[i] .+ L) ./ h) for i = 1:length(d)]
unique!(r0)
@time wf, H1, M1 = sce_iv(ne, r0, ham; M=1); # wf : wave_function; H1/M1 : initial ground state energy
maxt = 200
k = collect(1:5) .* 100 # size of iteration block
l = length(k)
@time y_n = map(x -> SCI_matfree(wf, ham, x; max_iter=maxt, tol=1e-8), k); # iteration by SCI algorithm
e = [y_n[i][1][end] for i = 1:l] .- E; # energy errors of different k
dof = [y_n[i][2][end] for i = 1:l] # DOF of different k
# plot errors and DOF
markers = filter((m -> begin
        m in Plots.supported_markers()
    end), Plots._shape_keys)
cols = collect(palette(:tab10))
p1 = plot(k, e, lw=3, yforeground_color_text=cols[1], yforeground_color_guide=cols[1], seriestype=:path, markershape=:circle, markersize=7, markerstrokecolor=cols[1], tickfontsize=20, legendfontsize=20, legend=(0.16, 0.9), label="Energy Error", ylabel="Energy Error", foreground_color_legend=nothing, guidefontsize=24, color=cols[1], grid=:off, xlabel=L"\mathfrak{K}", title=L"N=4", size=(930, 680), titlefontsize=30, right_margin=27mm, left_margin=4mm, top_margin=3mm, bottom_margin=2mm, xaxis=(formatter = x -> string(round(Int, x / 10^2)))) # plot errors
p = twinx()
plot!(p, k, dof, lw=3, yforeground_color_text=cols[2], yforeground_color_guide=cols[2], seriestype=:path, markershape=:circle, markerstrokecolor=cols[2], markersize=7, box=:on, label="DOF", legend=(0.86, 0.74), color=cols[2], tickfontsize=20, legendfontsize=20, guidefontsize=24, grid=:off, ylabel="DOF", foreground_color_legend=nothing, xaxis=(formatter = x -> string(round(Int, x / 10^2))), yaxis=(formatter = y -> string(round(Int, y / 10^3)))) #plot DOF
annotate!([(1145, maximum(dof) * 1.0, Plots.text(L"\times10^{3}", 21, cols[2], :center))])
annotate!([(k[l] * 1.0, dof[1] * 0.52, Plots.text(L"\times10^{2}", 21, :black, :center))])



# 6 particles
ne = 6;
L = 5.0;
N = 43;
h = 2L / N;
a = 0.1;
b = 0.5;
vext(x) = b * x^2
vee(x) = @. 1 / sqrt(1e-3 + x^2)
ham = ham1d(L, N; alpha_lap=a, vext);
ϵ = 5.0e-7;
E = 19.34854878852888
println("-------------------------------------------------------------------------------")
println("\n $(ne)-particle", "  N = $(N)", "  alpha = $(a)")
d = InitPT(ne, ham; num=50) # find the minimizers
r0 = [round.(Int, (d[i] .+ L) ./ h) for i = 1:length(d)]
unique!(r0)
@time wf, H1, M1 = sce_iv(ne, r0, ham; M=1); # wf : wave_function; H1/M1 : initial ground state energy
maxt = 200
k = collect(1:3) .* 100
l = length(k)
@time y_n = map(x -> SCI_matfree(wf, ham, x; max_iter=maxt, tol=1e-8), k); # iteration by SCI algorithm
e = [y_n[i][1][end] for i = 1:l] .- E;
dof = [y_n[i][2][end] for i = 1:l]
markers = filter((m -> begin
        m in Plots.supported_markers()
    end), Plots._shape_keys)
cols = collect(palette(:tab10))
p1 = plot(k, e, lw=3, yforeground_color_text=cols[1], yforeground_color_guide=cols[1], seriestype=:path, markershape=:circle, markersize=7, markerstrokecolor=cols[1], tickfontsize=20, legendfontsize=20, legend=(0.15, 0.9), label="Energy Error", ylabel="Energy Error", foreground_color_legend=nothing, guidefontsize=24, color=cols[1], grid=:off, xlabel=L"\mathfrak{K}", title=L"N=6", size=(930, 680), titlefontsize=30, right_margin=27mm, left_margin=8.7mm, top_margin=3mm, bottom_margin=2mm, xaxis=(formatter = x -> string(round(Int, x / 10^2))))
p = twinx()
plot!(p, k, dof, lw=3, yforeground_color_text=cols[2], yforeground_color_guide=cols[2], seriestype=:path, markershape=:circle, box=:on, markerstrokecolor=cols[2], markersize=7, label="DOF", legend=(0.86, 0.76), color=cols[2], tickfontsize=20, legendfontsize=20, guidefontsize=24, grid=:off, ylabel="DOF", foreground_color_legend=nothing, xaxis=(formatter = x -> string(round(Int, x / 10^2))), yaxis=(formatter = y -> string(round(Int, y / 10^4))))
annotate!([(5780, dof[l] * 1.007, Plots.text(L"\times10^{4}", 21, cols[2], :center))])
annotate!([(k[l] * 1.0, 222500, Plots.text(L"\times10^{2}", 21, :black, :center))])
