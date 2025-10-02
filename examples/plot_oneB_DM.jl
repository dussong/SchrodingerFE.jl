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

# SchrodingerFE.one_body_DM_coef(ne, wf_FCIfull.wf, C)


# density_coef(ne, wf_FCIfull.wf, ham.C)

# C = ham.C
# Ψ = wf_FCIfull.wf
# n = ne

# N = C.n
# Ψtensor = zeros(Float64, ntuple(x -> 2 * N, n))
# basis1body = 1:2*N
# combBasis = collect(combinations(basis1body, n))
# # compute the permutations and parity
# v = 1:n
# p = collect(permutations(v))[:]
# ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]

# mat_gamma = zeros(Float64,N,N)

# # reshape the vector Ψ to the (antisymmetric) tensor
# for j = 1:length(combBasis)
#     ij = combBasis[j]
#     for k = 1:length(p)
#         ik = seq2num_ns(2N, n, ij[p[k]])
#         Ψtensor[ik] = Ψ[j] * ε[k]
#     end
# end


# mass = overlap(n - 1, C)
# for kx = 1:2*N # first variable x
#     for ky = 1:2*N # first variable y
#         sptr = zeros(Int, n - 1, 1)
#         for s = 1:2^(n-1) #loop over spin variables
#             # all other variables
#             sp = sptr * N
#             ukx = getindex(Ψtensor, kx, ntuple(x -> sp[x]+1:sp[x]+N, n - 1)...)
#             ukvecx = reshape(ukx, N^(n - 1), 1)[:]
#             uky = getindex(Ψtensor, ky, ntuple(x -> sp[x]+1:sp[x]+N, n - 1)...)
#             ukvecy = reshape(uky, N^(n - 1), 1)[:]
#             if kx <= N
#                 if ky <= N
#                     mat_gamma[kx,ky] += dot(ukvecx, mass, ukvecy)
#                 else 
#                     mat_gamma[kx,ky-N] += dot(ukvecx, mass, ukvecy)
#                 end
#             else
#                 if ky <= N
#                     mat_gamma[kx-N,ky] += dot(ukvecx, mass, ukvecy)
#                 else 
#                     mat_gamma[kx-N,ky-N] += dot(ukvecx, mass, ukvecy)
#                 end
#             end
#             # adjust sptr
#             sptr[1] += 1
#             if n >= 3
#                 for ℓ = 1:n-2
#                     if sptr[ℓ] == 2
#                         sptr[ℓ] = 0
#                         sptr[ℓ+1] += 1
#                     end
#                 end
#             end # end if
#         end
#     end
# end
# return 0.5*(mat_gamma+mat_gamma')

# 0.5*(mat_gamma+mat_gamma')

# heatmap(xx,xx,0.5*(mat_gamma+mat_gamma'))


# plot(xx,SchrodingerFE.density(wf_FCIfull, ham))


# heatmap(xx,xx,SchrodingerFE.one_body_DM_coef(ne, wf_FCIfull.wf, C))

