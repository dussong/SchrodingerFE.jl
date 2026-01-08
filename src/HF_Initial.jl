#-------------------------------------------------------------------------------
# Solving Schrödinger Equation with Hatree Fock Method Based on Finite Element Discretization
# {ϕ_vs} v = 1,...,N;s = 0,1
# Ψ = |φ_1,0 ...φ_[ne/2],0 φ1,1...φ_ne-[ne/2],1>
# φ_ks = Σ_(v=1...N)u_v,ks * ϕ_vs
# H*φi = ϵi*φi -> H*Ui = ϵi*M*Ui
#-------------------------------------------------------------------------------

export HF

function genFock(AH, AF, ρ, B::Array{Float64,4})
    @tensor AH[i, j] = B[i, a, j, b] * ρ[a, b]
    @tensor AF[i, j] = B[a, b, i, j] * ρ[a, b]

    @. AH = (AH + AH') / 2
    @. AF = (AF + AF') / 2

    return AH, AF
end

function genFock(AH, AF, ρ, B::SparseMatrixCSC{Float64,Int64})
    colptr = B.colptr
    rowval = B.rowval
    N = size(AH,1)

    jptr = colptr[2:end] - colptr[1:end-1]
    count = 0
    @. AH = 0.
    @. AF = 0.
    for j = 1:length(jptr) #loop for the nonzero elements of column
        jtr = jptr[j]
        if jtr > 0
            jx = j % N == 0 ? N : j % N
            jy = div(j - jx, N) + 1
            for k = 1:jtr
                i = rowval[count+k]
                ix = i % N == 0 ? N : i % N
                iy = div(i - ix, N) + 1

                AH[ix, jx] += B[i, j] * ρ[iy, jy]
                AF[ix, jx] += B[i, j] * ρ[ix, iy]
            end
            count += jtr
        end
    end
           
    @. AH = (AH + AH') / 2
    @. AF = (AF + AF') / 2

    AH, AF
end

function scfHF(ne::Int64, ham::Hamiltonian, Norb::Int64, 
            max_iter::Int64, mixing::Float64, scf_tol::Float64)
    # ===== solve linear problem to initialize ρ =====#
    AΔ = ham.AΔ
    AV = ham.AV
    M = ham.C
    Bee = ham.Bee
    alpha_lap = ham.alpha_lap

    N = M.n
    AH = zeros(N, N)
    AF = zeros(N, N)
    H = zeros(N, N)

    A = 0.5 .* alpha_lap .* AΔ + AV
    λ, W = eigs(A, M; nev=Norb, which=:SR)
    ρ1 = zeros(N, N)
    @views for k = 1:Norb
        ρ1 += 2.0 * W[:, k] * W[:, k]'
    end
    ρ = copy(ρ1)

    # ===== start the SCF iterations =====#
    err = 1.0
    k1 = mixing
    k2 = 1.0 - k1  # charge mixing parameter
    for k = 1:max_iter
        if err < scf_tol
            break;
        end
        genFock(AH, AF, ρ, Bee)
        @. H = A + AH - 0.5 * AF

        λ, W = eigs(H, M; nev=Norb, which=:SR)
        ρ2 = zeros(N, N)
        @views for j = 1:Norb
            ρ2 += 2.0 * W[:, j] * W[:, j]'
        end
        ρ = k1 .* ρ1 + k2 .* ρ2
        err = norm(ρ2 - ρ1)
        ρ1 = ρ
        println(" step : $(k),  err : $(err)")
    end

    W
end

function HFtoFCI(ne::Int64, N::Int64, U::Array{Float64,2})
    ind = Int64[]
    val = Float64[]

    m = zeros(Int64, 2)
    m[1] = cld(ne, 2)
    m[2] = ne - m[1]

    basis1body = 1:N
    combBasis = map(x -> collect(combinations(basis1body, x)), m)
    b = zeros(length(combBasis[1]), 2)

    # loop for the spin
    for i = 1:2
        v = 1:m[i]
        p = collect(permutations(v))[:]
        ε = (-1) .^ [parity(p[l]) for l = 1:length(p)]
        ij = zeros(Int64, m[i])
        for j = 1:length(combBasis[i])
            @views ij = combBasis[i][j]
            for k = 1:length(p)
                a = ε[k]
                for l = 1:m[i]
                    ik = ij[p[k][l]]
                    a *= U[ik, l]
                end
                b[j, i] += a
            end
        end
    end

    # combine s=0 with s=1
    for i = 1:length(combBasis[1])
        for j = 1:length(combBasis[2])
            ij = vcat(combBasis[1][i], combBasis[2][j] .+ N)
            push!(ind, seq2num_ns(2N, ne, ij))
            push!(val, b[i, 1] * b[j, 2])
        end
    end

    c0 = sparsevec(ind, val, (2N)^ne)
    return c0
end

function HF_1B(ne::Int64, U::Array{Float64,2}, 
               A::SparseMatrixCSC{Float64,Int64},
               C::SparseMatrixCSC{Float64,Int64})
    N = size(U, 1)
    rowval = A.rowval
    jptr = A.colptr[2:end] - A.colptr[1:end-1]
    m1 = cld(ne, 2)
    Af = zeros(Float64, m1, m1)
    Cf = zeros(Float64, m1, m1)

    for k = 1:m1, l = 1:m1
        count = 0
        for j = 1:length(jptr) #loop for the nonzero elements of column
            jtr = jptr[j]
            for jk = 1:jtr
                i = rowval[count+jk]

                Af[k, l] += U[i, k] * U[j, l] * A[i, j]
                Cf[k, l] += U[i, k] * U[j, l] * C[i, j]
            end
            count += jtr
        end
    end

    return Af, Cf
end

function HF_2B(ne::Int64, U::Array{Float64,2}, B::Array{Float64,4})
    N = size(U, 1)
    m1 = cld(ne, 2)
    Bf = zeros(Float64, m1, m1, m1, m1)

    for i1 = 1:m1, i2 = 1:m1, j1 = 1:m1, j2 = 1:m1
        for k1 = 1:N, k2 = 1:N, l1 = 1:N, l2 = 1:N
            Bf[i1, i2, j1, j2] += U[k1, i1] * U[k2, i2] * U[l1, j1] * U[l2, j2] * B[k1, k2, l1, l2]
        end
    end

    return Bf
end

function HF_2B(ne::Int64, U::Array{Float64,2}, B::SparseMatrixCSC{Float64,Int64})
    N = size(U, 1)
    m1 = cld(ne, 2)
    rowval = B.rowval
    lptr = B.colptr[2:end] - B.colptr[1:end-1]
    Bf = zeros(Float64, m1, m1, m1, m1)

    for i1 = 1:m1, i2 = 1:m1, j1 = 1:m1, j2 = 1:m1
        count = 0
        for l = 1:length(lptr) #loop for the nonzero elements of column
            ltr = lptr[l]
            if ltr > 0
                l1 = l % N == 0 ? N : l % N
                l2 = div(l - l1, N) + 1
                for li = 1:ltr
                    k = rowval[count+li]
                    k1 = k % N == 0 ? N : k % N
                    k2 = div(k - k1, N) + 1

                    Bf[i1, i2, j1, j2] += U[k1, i1] * U[k2, i2] * U[l1, j1] * U[l2, j2] * B[k, l]
                end
                count += ltr
            end
        end
    end

    return Bf
end

function energyHF(ne::Int64, U::Array{Float64,2}, ham::Hamiltonian)
    A = 0.5 .* ham.alpha_lap .* ham.AΔ + ham.AV
    C = ham.C
    B = ham.Bee

    v = 1:ne
    p = collect(permutations(v))[:]
    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
    coulomb_which2 = collect(combinations(v, 2))
    valH = 0.0
    valM = 0.0
    m1 = cld(ne, 2)
    m2 = ne - m1

    i = vcat(collect(1:m1), collect(1:m2))
    s = vcat(zeros(Int, m1), ones(Int, m2))
    jp = zeros(Int, ne)
    tp = zeros(Int, ne)
    Cp = zeros(Float64, ne)

    Af, Cf = HF_1B(ne, U, A, C)
    Bf = HF_2B(ne, U, B)

    for k = 1:length(p)
        Av = 0.0
        Bv = 0.0
        for l in 1:ne
            tp[l] = s[p[k][l]]
            jp[l] = i[p[k][l]]
            Cp[l] = Cf[i[l], jp[l]]
        end
        Cv = prod(Cp)
        if s == tp && Cv != 0.0
            for l in 1:ne
                Av += Af[i[l], jp[l]] / Cp[l]
            end
            Av *= Cv

            for l = 1:length(coulomb_which2)
                ca = coulomb_which2[l][1]
                cb = coulomb_which2[l][2]
                Bv += Bf[i[ca], i[cb], jp[ca], jp[cb]] / (Cp[ca] * Cp[cb])
            end
            Bv *= Cv

            valH += ε[k] * (Av + Bv)
            valM += ε[k] * Cv
        end
    end

    return valH, valM
end

function HF(ne::Int64, ham::Hamiltonian; Norb=cld(ne, 2), 
            max_iter=100, mixing=0.8, scf_tol=1e-5)
    println("SCF time : ")
    @time U0 = scfHF(ne, ham, Norb, max_iter, mixing, scf_tol)
    U = real.(U0)
    @assert norm(U-U0) < 1e-8

    println("Turn into FCI time : ")
    N = ham.C.n
    @time c0 = HFtoFCI(ne, N, U)
    wfsp = WaveFunction_sp(ne, N, c0)

    println("Compute energy time : ")
    @time valH, valM = energyHF(ne, U, ham)
    println("Hartree-Fock energy : ", valH/valM)

    return wfsp, U, valH, valM
end
