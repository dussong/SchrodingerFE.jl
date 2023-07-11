using KrylovKit
using LinearAlgebra
using SchrodingerFE

ne1 = [2,3];
N1 = [30,20];
L = 5.0;
for (ne, N) in zip(ne1, N1)
    ham = ham1d(L,N);
    n = N-1;
    x0 = rand(binomial(2n,ne));
    println("----------------------------------------------")
    println("\n $(ne)-particle","  N = $(N)")
    println("----------------------------------------------")
    println(" no matrix free:")
    @time begin
        H, M = hamiltonian(ne,ham)
        λ1, u1 = geneigsolve((H,M),1,:SR)
    end
    println(" matrix free:")
    function M_Ψ(Ψ::Array{Float64,1})
        HΨ,MΨ = ham_free_tensor(ne,Ψ,ham)
        return HΨ,MΨ
    end
    @time λ2, u2 = geneigsolve(M_Ψ, x0, 1, :SR;issymmetric = true,isposdef = true)
    e1 = norm(λ2-λ1)
    #@test(abs(e1) < 1e-8)
    println("  eigenvalue's error = $(e1)")
    u1 = u1[1] ./ norm(u1[1])
    u2 = u2[1] ./ norm(u2[1])
    e2 = min(norm(u2+u1), norm(u2-u1))
    #@test(abs(e2) < 1e-8)
    println("  eigenvector's error = $(e2)")
end
