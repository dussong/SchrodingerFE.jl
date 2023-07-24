using SchrodingerFE
using Printf
using Test
using Arpack
using LinearAlgebra

include("check_lap_overlap.jl")
include("check_coulomb_full.jl")
include("check_density.jl")

# Check that the matrices from the old and new implementation match
L = 5.
N = 6

println("-------------------------------------------")
println(" Testing matching full hamiltonian matrices")
println("-------------------------------------------")
# Defining Coulomb
vee(x) = 1.0./abs.(x)

# ---------------------------------------------------------------
# 2 electrons case
# --------------------------------------------------------------
println("         Testing 2 electrons case          ")
println("-------------------------------------------")
println("        Laplace and Overlap matrix         ")
(mat_lap, mat_mass, rank) = mat_2d_fermion(L, N)
(Htot, mat_mass2, rank2, mat_lap2) = mat_2d_fermion_coul(1., L, N)
mat_coul = Htot - mat_lap2

# checking the 2 matlab implementations
@test (norm(mat_lap-mat_lap2) < 1e-10)
@test (norm(mat_mass-mat_mass2) < 1e-10)
@test (norm(rank-rank2) < 1e-10)

# Julia implementation
AΔ = oneB_lap(L, N)
C  = oneB_over(L, N)
Bee = twoB_V(L, N, x->vee(x),3,4)
# laplace matrix
HΔ = ham_1B_sp(2, AΔ, C)
# overlap matrix
M  = ham_1B_sp(2, C/2, C)
# Coulomb matrix
Vee = ham_2B_sp(2, Bee, C)

# computing eigenvalues
eigLap = sort(eigvals(Array(mat_lap),Array(mat_mass)))
eigLapJ = sort(eigvals(Array(HΔ),Array(M)))

eigCoul = sort(eigvals(Array(mat_lap+mat_coul),Array(mat_mass)))
eigCoulJ = sort(eigvals(Array(HΔ+Vee),Array(M)))

@test (norm(eigLap-eigLapJ,Inf) < 1e-7)
@test (norm(eigCoul-eigCoulJ,Inf) < 1e-2) #not enough ??

# @test (norm(HΔ-mat_lap,Inf) < 1e-7)
# @test (norm(Vee-mat_coul,Inf) < 1e-2)

# Checking the electronic density

# matlab version
Em, Ψm = eigs(mat_lap, mat_mass, nev = 1, which=:SM)
ρm,ρ2m = reshape_2d_fermion(real(Ψm[:,1]), N, rank)
ρm = (ρm * N * 2) / (norm(ρm, 1) * 2L);

# Julia version
Ej, Ψj = eigs(HΔ, M, nev = 1, which=:SM)
ham = ham1d(L, N)

x = collect(range(-L,L,length=N+1))[2:end-1]
ρj = SchrodingerFE.density(x,2, real(Ψj[:,1]), ham)
ρj = (ρj * N * 2) / (norm(ρj, 1) * 2L);

ρ2j = pair_density(hcat(x,x), L, 2,  real(Ψj[:,1]), ham.C)
@test norm(ρm-ρj) < 1e-9

# --------------------------------------------------------------
# 3 electrons case
# --------------------------------------------------------------
println("         Testing 3 electrons case")
(mat_lap, mat_mass, rank) = mat_3d_fermion(L, N)
(mat_lap2, mat_coul, mat_mass2, rank2) = mat_3d_fermion_coul(L, N)

# checking the 2 matlab implementations
@test (norm(mat_lap-6*mat_lap2) < 1e-10)
@test (norm(mat_mass-6*mat_mass2) < 1e-10)
@test (norm(rank-rank2) < 1e-10)

# Julia implementation
AΔ = oneB_lap(L, N)
C  = oneB_over(L, N)
Bee = twoB_V(L, N, x->vee(x),4,3)
# laplace matrix
HΔ = ham_1B_sp(3, AΔ, C)
# overlap matrix
M  = ham_1B_sp(3, C/3, C)
# Coulomb matrix
Vee = ham_2B_sp(3, Bee, C)

# computing eigenvalues
eigLap = eigvals(Array(mat_lap),Array(mat_mass))
eigLapJ = eigvals(Array(HΔ),Array(M))

eigCoul = sort(eigvals(Array(6*mat_coul),Array(mat_mass)))
eigCoulJ = sort(eigvals(Array(Vee),Array(M)))

@test (norm(eigLap-eigLapJ,Inf) < 1e-7)
@test (norm(eigCoul-eigCoulJ,Inf) < 1e-7)

# --------------------------------------------------------------
# 4 electrons case
# --------------------------------------------------------------
println("         Testing 4 electrons case")
@time (mat_lap, mat_mass, rank) = mat_4d_fermion(L,N)

# computing eigenvalues
eiglap = sort(eigvals(Array(mat_lap),Array(mat_mass)))

# Julia implementation
AΔ = oneB_lap(L, N)
C  = oneB_over(L, N)
# laplace matrix
HΔ = ham_1B_sp(4, AΔ, C)
# overlap matrix
M  = ham_1B_sp(4, C/4, C)

# computing eigenvalues
eiglapJ = sort(eigvals(Array(HΔ), Array(M)))

@test (norm(eiglap-eiglapJ,Inf) < 1e-8)



# --------------------------------------------------------------
# 2 electrons case - check the density
# --------------------------------------------------------------
println("         Testing 4 electrons case")
@time (mat_lap, mat_mass, rank) = mat_4d_fermion(L,N)

# computing eigenvalues
eiglap = sort(eigvals(Array(mat_lap),Array(mat_mass)))

# Julia implementation
AΔ = oneB_lap(L, N)
C  = oneB_over(L, N)
# laplace matrix
HΔ = ham_1B_sp(4, AΔ, C)
# overlap matrix
M  = ham_1B_sp(4, C/4, C)

# computing eigenvalues
eiglapJ = sort(eigvals(Array(HΔ), Array(M)))

@test (norm(eiglap-eiglapJ,Inf) < 1e-8)
