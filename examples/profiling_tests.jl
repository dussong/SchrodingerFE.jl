# using Plots
#pyplot()
using Profile
using BenchmarkTools
using LinearAlgebra
using SchrodingerFE
using SparseArrays

ne = 3; # Nb of particles
L = 10.0; #Interval length
N = 20; #Nb of discretization points of the interval [-L,L]
α = 1. #parameter in the laplace operator

# Spatial mesh
xx = collect(range(-L, L, length=N + 1))[2:end-1]

# External potential
a = 1.0
b = -1.0
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

E_FCIsparse2, wf_FCIsparse2 = WaveFunction(ne, ham, "FCI_sparse2"; maxiter=50, kdim=50)
# Solve the eigenvalue problem with different methods
# @time E_FCIsparse2, wf_FCIsparse2 = WaveFunction(ne, ham, "FCI_sparse2"; maxiter=50, kdim=50)

# @time E_FCIsparse, wf_FCIsparse = WaveFunction(ne, ham, "FCI_sparse"; maxiter=50, kdim=50)



# 2
# myfunc() = WaveFunction(ne, ham, "FCI_sparse2"; maxiter=50, kdim=50)

# @profile myfunc()

# using ProfileView
# ProfileView.view()
# 2
# # using Profile
# # Profile.Allocs.@profile myfunc()
# # Profile.Allocs.print()
# using BenchmarkTools
# @btime myfunc()

# Ψ = [-0.039431755067422414, 0.11359693760885128, -0.2977193734472515, -0.08848803301956995, 0.36109850187753967, -0.13754431097171724, 0.36109850187753967, 0.11359693760885137, 0.36109850187753967, -0.08848803301956995, -0.08848803301956995, -0.2977193734472515, 0.11359693760885128, -0.29771937344725125, -0.13754431097171674, -0.29771937344725125, -0.08848803301956973, 0.3610985018775397, 0.11359693760885119, -0.03943175506742242]

# dim = (binomial(2ham.C.n, ne))
# x0 = ones(dim)
# N = ham.C.n
# println("Dimension of the problem: $(dim)")

# using Combinatorics

# phih, phim, combBasis = preallocate1(ne, N)
# Ψtensor, phihtensor, phimtensor, phiAtensor1, phiBtensor1, phiCtensor1 = preallocate2(ne, N)
# M_Ψ(Ψ::Array{Float64,1}) = ham_free_tensor!(
#    ne, N, Ψ, ham.AΔ, ham.AV, ham.C, ham.Bee,
#    phih, phim, combBasis,
#    Ψtensor, phihtensor, phimtensor, phiAtensor1, phiBtensor1, phiCtensor1;
#    alpha_lap=ham.alpha_lap)

# function ham_free_tensor!(ne::Int, N::Int, Ψ::Array{Float64,1},
#    AΔ::SparseMatrixCSC{Float64,Int64}, AV::SparseMatrixCSC{Float64,Int64},
#    C::SparseMatrixCSC{Float64,Int64},
#    B::Array{Float64,4},
#    phih, phim, combBasis,
#    Ψtensor, phihtensor, phimtensor, phiAtensor1, phiBtensor1, phiCtensor1;
#    alpha_lap=1.0)
#    # N = C.n
#    # Ψtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
#    # phihtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
#    # phimtensor = zeros(Float64, ntuple(x -> 2 * N, ne))
#    # phiAtensor1 = zeros(Float64, ntuple(x -> N, ne))
#    # phiBtensor1 = zeros(Float64, ntuple(x -> N, ne))
#    # phiCtensor1 = zeros(Float64, ntuple(x -> N, ne))
#    Ψtensor .= 0.0
#    phihtensor .= 0.0
#    phimtensor .= 0.0
#    phiAtensor1 .= 0.0
#    phiBtensor1 .= 0.0
#    phiCtensor1 .= 0.0

#    # indecies for the basis
#    # basis1body = 1:2*N
#    # combBasis = collect(combinations(basis1body, ne))
#    # phi = H⋅Ψ
#    @assert length(Ψ) == length(combBasis)
#    @assert ne > 1
#    # phih = zeros(size(Ψ))
#    # phih = zeros(length(Ψ))
#    phih .= 0.0
#    # @show phih == zeros(size(Ψ))
#    # phih = ones(size(Ψ))
#    # phim = zeros(size(Ψ))
#    phim .= 0.0

#    @show Ψ
#    # @show phih
#    # @show phim
#    # phim .= 0.
#    # @show phih

#    # computate the permutations and paritiy
#    v = 1:ne
#    p = collect(permutations(v))[:]
#    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
#    coulomb_which2 = collect(combinations(v, 2))
#    #ik = zeros(Int,ne)
#    # reshape the vector Ψ to the (antisymmetric) tensor
#    for j = 1:length(combBasis)
#       ij = combBasis[j]
#       for k = 1:length(p)
#          ik = ij[p[k][1]]
#          for l = 2:ne
#             ik += (ij[p[k][l]] - 1) * (2N)^(l - 1)
#          end
#          Ψtensor[ik] = Ψ[j] * ε[k]
#       end
#    end

#    A = 0.5 * alpha_lap * AΔ + AV

#    # loop through different spin configurations
#    sptr = zeros(Int64, ne)
#    m1 = zeros(Int64, ne)
#    mp1 = zeros(Int64, ne)
#    m2 = zeros(Int64, ne)
#    mp2 = zeros(Int64, ne)
#    W = zeros(N, N, N^(ne - 2))
#    M1 = zeros(N, N^(ne - 1))
#    MA1 = zeros(N, N^(ne - 1))
#    MC1 = zeros(N, N^(ne - 1))
#    for s = 1:2^ne
#       sp = sptr * N
#       np = ntuple(x -> sp[x]+1:sp[x]+N, ne)
#       for j = 1:ne # act A on the j-th particle
#          phiAtensor = getindex(Ψtensor, np...)
#          phiCtensor = copy(phiAtensor)
#          # perform C⊗⋯⊗C⊗A⊗C⊗⋯⊗C on the tensor ψtensor
#          for i = 1:ne
#             m1[1] = i
#             m1[2:end] = setdiff(1:ne, i)
#             MA = reshape(permutedims!(phiAtensor1, phiAtensor, m1), N, N^(ne - 1))
#             MC = reshape(permutedims!(phiCtensor1, phiCtensor, m1), N, N^(ne - 1))
#             if i == j
#                mul!(MA1, A, MA)
#             else
#                mul!(MA1, C, MA)
#             end
#             mul!(MC1, C, MC)
#             sortperm!(mp1, m1)
#             permutedims!(phiAtensor, reshape(MA1, ntuple(x -> N, ne)), mp1)
#             permutedims!(phiCtensor, reshape(MC1, ntuple(x -> N, ne)), mp1)
#          end
#          # assemble the value to phitensor
#          phihtensor[np...] += phiAtensor
#          phimtensor[np...] += phiCtensor
#       end

#       for j = 1:length(coulomb_which2) # act B on the k-th,l-th particle
#          k = coulomb_which2[j][1]
#          l = coulomb_which2[j][2]
#          phiBtensor = getindex(Ψtensor, np...)
#          # perform C⊗⋯⊗C⊗B⊗C⊗⋯⊗C on the tensor ψtensor
#          for i = 1:ne
#             if i != k && i != l
#                m2[1] = i
#                m2[2:end] = setdiff(1:ne, i)
#                M = reshape(permutedims!(phiBtensor1, phiBtensor, m2), N, N^(ne - 1))
#                mul!(M1, C, M)
#                permutedims!(phiBtensor, reshape(M1, ntuple(x -> N, ne)), sortperm!(mp2, m2))
#             elseif i == k
#                m2[1] = k
#                m2[2] = l
#                m2[3:end] = setdiff(1:ne, k, l)
#                M = reshape(permutedims!(phiBtensor1, phiBtensor, m2), N, N, N^(ne - 2))
#                @. W = 0.0
#                @tensor W[i, j, k] = B[i, j, a, b] * M[a, b, k]
#                permutedims!(phiBtensor, reshape(W, ntuple(x -> N, ne)), sortperm!(mp2, m2))
#             end
#          end
#          # assemble the value to phitensor
#          phihtensor[np...] += phiBtensor
#       end
#       # adjust sptr
#       sptr[1] += 1
#       for ℓ = 1:ne-1
#          if sptr[ℓ] == 2
#             sptr[ℓ] = 0
#             sptr[ℓ+1] += 1
#          end
#       end
#    end

#    for i = 1:length(combBasis)
#       il = combBasis[i]
#       l = il[1]
#       for j = 2:ne
#          l += (il[j] - 1) * (2N)^(j - 1)
#       end
#       # @show phihtensor[l]
#       phih[i] = phihtensor[l]
#       phim[i] = phimtensor[l]
#    end
#    @show phih
#    return phih, phim ./ ne
# end

# using TensorOperations

# M_Ψ(Ψ)


2
# N = 10
# A = rand(N,N)
# A = A'*A

# function allocate(N)
#    phi = zeros(N)
#    return phi
# end

# x0 = ones(N)


# phi = allocate(N)




# function Ax2!(x, phi)
#    phi .= 0 
#    return A*x, x
# end

# using KrylovKit
# kdim = 5
# maxiter = 100
# E, Ψt, cvinfo = geneigsolve(Ax2!, x0, 1, :SR; krylovdim=kdim, maxiter=maxiter, issymmetric=true,
# isposdef=true)

# function 


# M_Ψ(Ψ::Array{Float64,1}) = ham_free_tensor!(
#    ne, N, Ψ, ham.AΔ, ham.AV, ham.C, ham.Bee,
#    Φh, Φm, combBasis; 
#    alpha_lap=ham.alpha_lap)

# E, Ψt, cvinfo = geneigsolve(M_Ψ, x0, 1, :SR; krylovdim=kdim, maxiter=maxiter, issymmetric=true,
#    isposdef=true)