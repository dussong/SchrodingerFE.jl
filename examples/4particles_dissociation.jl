using Plots
pyplot()

#using JSON
using LinearAlgebra
#using Dierckx
using PairDensities
using SparseArrays

ne = 4; # Nb of particles
L = 10.0; #Interval length
N = 60; #Nb of discretization points of the interval [-L,L]
α = 1.0; #Parameter in the potential
A = [3.0, 4.0, 5.0] #spatial coef for the potential
α_lap = 0.1
B = [-3.0, -4.0, -5]
h = 2L / N

xx = range(-L, L; length=N - 1)


for (i, a) in enumerate(A)
   b = B[i]
   @show i

   vext(x) = -α * (1.0 ./ sqrt.((x - a - 1.5) .^ 2 .+ 1)
                   .+
                   1.0 ./ sqrt.((x - a + 1.5) .^ 2 .+ 1)
                   .+
                   1.0 ./ sqrt.((x - b - 1.5) .^ 2 .+ 1)
                   .+
                   1.0 ./ sqrt.((x - b + 1.5) .^ 2 .+ 1))

   #double well taken from Wagner, L.O., Stoudenmire, E.M., Burke, K., White, S.R.: Reference electronic structure calculations in one dimension. Phys. Chem. Chem. Phys. 14, 8581–8590 (2012). https://doi.org/10.1039/c2cp24118h
   vee(x) = 1.0 ./ sqrt.(x .^ 2 .+ 1)

   #a1 = 1.0
   b1 = 1.0
   Vext(x) = b1 * vext(x)
   ham = ham1d(L, N; alpha_lap=α_lap, vext=Vext, vee)

   # wfsp = WaveFunction(ne, ham, "FCI_full"; maxiter=50,
      # kdim=500)
   Esp, wfsp = WaveFunction(ne, ham, "selected_CI_sparse"; L=L, max_iter=3000, k=500, num=500)

   @time ρ = PairDensities.density(wfsp, ham)
   ρ = (ρ * N * ne) / (norm(ρ, 1) * 2L)

   ρ2MF = 0.5 * ρ * ρ'

   # plot pair density
   wff = WaveFunction_full(ne, Array(wfsp.wfP))
   @time ρ2 = pair_density(wff, ham)
   ρ2 = (ρ2 * N^2 * binomial(ne, 2)) / (norm(ρ2, 1) * 4 * L^2)
   ρ2 = reshape(ρ2, N - 1, N - 1)

   # Ratio to mean-field
   ratio_MF = 2 * ne / (ne - 1) * ρ2 ./ (ρ * ρ')

   # Transport map for ρ
   T = 2 * L / N / ne * cumsum(ρ)

   P = plot(xx, vext.(xx), label="", xlims=(-L, L), linewidth=4, tickfontsize=16)
   savefig("plots_examples/dissociation22/external_pot_$i.pdf")

   display(P)
   sleep(0.5)

   P = plot(xx, ρ, label="", xlims=(-L, L), linewidth=4, tickfontsize=16)
   savefig("plots_examples/dissociation22/density_$i.pdf")

   display(P)
   sleep(0.5)

   P = contour(xx, xx, ρ2, c=:viridis, aspect_ratio=:equal,
      colorbar=:true, fill=:true, tickfontsize=16, colorbar_tickfontsize=16)
   savefig("plots_examples/dissociation22/pair_density$i.pdf")

   display(P)
   sleep(0.5)

   P = contour(xx, xx, ρ2MF, c=:viridis, aspect_ratio=:equal, colorbar=:true, fill=:true, tickfontsize=16, colorbar_tickfontsize=16)
   savefig("plots_examples/dissociation22/mean_field$i.pdf")

   display(P)
   sleep(0.5)

   P = contour(T, T, ratio_MF, c=:viridis, aspect_ratio=:equal, colorbar=:true, fill=:true, clim=(0.0, 1.7), tickfontsize=16, colorbar_tickfontsize=16)
   savefig("plots_examples/dissociation22/copula_$i.pdf")

   display(P)
   sleep(0.5)

end
