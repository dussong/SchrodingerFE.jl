using SchrodingerFE
using LinearAlgebra
using Plots

ne = 1 # Nb of particles

L = 10.0; #Interval length
N = 2; #Nb of discretization points of the interval [-L,L]
α = 1.0 #parameter in the laplace operator

norb = N - 1

# External potential
a = 1.0
b = -1.0
vext(x) = 0.

# -(1.0 ./ sqrt.((x - a) .^ 2 .+ 1)
#             .+
#             1.0 ./ sqrt.((x - b) .^ 2 .+ 1)
# )

# Electron-electron interaction
vee(x) = 0.
# 1.0 ./ sqrt.(x .^ 2 .+ 1)

# For P1 Finite Elements
ham = ham1d(L, N; alpha_lap=α, vext=vext, vee);

Hamglobal = hamiltonian(1, ham)
Matrix(Hamglobal[1])
Matrix(Hamglobal[2])

E_FCIfull, wf_FCIfull = WaveFunction(ne, ham, "FCI_full"; maxiter=50, kdim=50)
# plot(density(wf_FCIfull,ham))

# Orthonormalize the basis
S = Matrix(ham.C)
CC = S^(-1/2)

hamAV = CC * ham.AV * CC
hamAΔ = CC * ham.AΔ * CC

hamAV2 = zeros(size(ham.AV))
hamAΔ2 = zeros(size(ham.AV))
for i in 1:norb
   for j in 1:norb
      hamAV2[i,j] = sum(sum(ham.AV[k,l]*CC[i,k]*CC[j,l] for k in 1:norb) for l in 1:norb)
      hamAΔ2[i,j] = sum(sum(ham.AΔ[k,l]*CC[i,k]*CC[j,l] for k in 1:norb) for l in 1:norb)
   end
end
hamAV
hamAV2
@show norm(hamAV - hamAV2)

C_on = Matrix(1.0I, norb, norb)
ham2Bno = ham.Bee
ham2B = zeros(size(ham.Bee))
for i1 in 1:size(ham2B, 1)
   for j1 in 1:size(ham2B, 2)
      for k1 in 1:size(ham2B, 3)
         for l1 in 1:size(ham2B, 4)
            ham2B[i1, j1, k1, l1] = sum(sum(sum(sum(
               CC[i1, i2] * CC[j1, j2] * CC[k1, k2] * CC[l1, l2] * ham2Bno[j2, i2, l2, k2]
               for i2 in 1:size(ham2B, 1))
                                                for j2 in 1:size(ham2B, 2))
                                            for k2 in 1:size(ham2B, 3))
                                        for l2 in 1:size(ham2B, 4))
         end
      end
   end
end

ham_on = ham1d(ham.L,ham.N,hamAΔ2,hamAV2,C_on,ham2B,ham.alpha_lap,ham.vext,ham.vee,ham.element)

Hamglobal_on = hamiltonian(1, ham_on)
Matrix(Hamglobal_on[1])
@show Matrix(Hamglobal_on[2])

E_FCIfull2, wf_FCIfull2 = WaveFunction(ne, ham_on, "FCI_full"; maxiter=50, kdim=50)
plot!(density(wf_FCIfull2, ham_on))

@show (E_FCIfull2 - E_FCIfull)