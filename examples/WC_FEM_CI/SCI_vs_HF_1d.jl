using PairDensities
using SparseArrays
using Plots, LaTeXStrings
using DelimitedFiles
using Plots.PlotMeasures

#compared with HF initial value
ne = 4; # Nb of particles
L = 5.0; #Interval length
N = 49; #Nb of discretization points of the interval [-L,L]
h = 2L / N;
a = 0.1; #Parameter in the kinetic potential
b = 0.5; #Parameter in the external potential
vext(x) = b * x^2
vee(x) = @. 1 / sqrt(1e-3 + x^2)
ham = ham1d(L, N; alpha_lap=a, vext);
ϵ = 5.0e-7; #tolerance of compression
#E = 20.274857420261288 # alpha = 10, beta = 0.5
#E = 11.643132859628299 # alpha = 1, beta = 0.5
E = 8.229757675996822 # reference ground state energy of alpha = 0.1, beta = 0.5
println("-------------------------------------------------------------------------------")
println("\n $(ne)-particle", "  N = $(N)", "  alpha = $(a)")
# generate SCE initial state
d = InitPT(ne, ham; num=50) # find the minimizers
r0 = [round.(Int, (d[i] .+ L) ./ h) for i = 1:length(d)]
unique!(r0)
@time wf, H1, M1 = sce_iv(ne, r0, ham; M=1); # wf : wave_function; H1/M1 : initial ground state energy

# generate HF initial state
@time wfhf, U, Hv, Mv = HF(ne, ham);


max_iter = 1000
k = 200 # size of iteration block
println("  SCE:")
@time y1, num1, c1 = SCI_matfree(wf, ham, k; max_iter=max_iter, tol=1e-8); # iteration by SCI algorithm
e1 = abs.(y1 .- E) # energy error 
println("  Hartree Fock:")
@time y2, num2, c2 = CDFCI_matfree_block(wfhf, ham, k; max_iter=max_iter, tol=1e-8); # iteration by CDFCI algorithm
e2 = abs.(y2 .- E);
# plot energy errors
l = minimum([length(e1), length(e2)])
ℓ1 = 500
l2 = fld(l, ℓ1)
x2 = vcat(1, collect(ℓ1:ℓ1:ℓ1*l2))
x = collect(1:l)
markers = filter((m -> begin
        m in Plots.supported_markers()
    end), Plots._shape_keys)
cols = collect(palette(:tab10))
p = plot(x, tickfontsize=18, e1[1:l], lw=3, ls=:solid, label="semi-classical limit ", legendfontsize=20, ylabel="Energy Error", guidefontsize=22, color=cols[1], title=L"\alpha=%$a",
    legend=:topright, grid=:off, box=:on, xlabel="Iteration", yscale=:log10, size=(720, 620), titlefontsize=30, right_margin=3mm, top_margin=3mm)#, ylims = (1e-2,1.2*e1[1]))
plot!(p, x, e2[1:l], lw=3, ls=:solid, label="Hartree-Fock", color=cols[2])
