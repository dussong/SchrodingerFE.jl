#-------------------------------------------------------------------------------
#   compute 1-body and 2-body properties from the ne-body wavefunction
#-------------------------------------------------------------------------------
export density_coef, pair_density_coef, density, pair_density, pair_density_spin_coef, pair_density_spin, pair_density_coef_2ne, pair_density_2ne
# PARAMETERS
# n : the number of electrons
# Ψ : ne-body wavefunction
# C : 1-body overlap
#
# RETURN
# ρ  : density
# ρ2 : pair density
# γ  : one-body reduced density matrix
# γ2 : two-body reduced density matrix
#-------------------------------------------------------------------------------


# return a tri-diagonal matrix that store the coefficients for ϕᵢ(x)⋅ϕⱼ(x)
function density_coef(n::Int64, Ψ::Array{Float64,1}, C::SparseMatrixCSC{Float64,Int64})
    N = C.n
    Ψtensor = zeros(Float64, ntuple(x -> 2 * N, n))
    basis1body = 1:2*N
    combBasis = collect(combinations(basis1body, n))
    # compute the permutations and paritiy
    v = 1:n
    p = collect(permutations(v))[:]
    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
    
    # integrate the n-1 variable out to obtain the coefficients of ρ
    indrow = [1:N; 1:N-1; 2:N]
    indcol = [1:N; 2:N; 1:N-1]
    val = zeros(Float64, 3 * N - 2)

    if n == 1
        @. val[1:N] = Ψ[1:N]^2 +  Ψ[N+1:2N]^2
        @. val[N+1:2N-1] = Ψ[1:N-1] * Ψ[2:N] + Ψ[N+1:2N-1] * Ψ[N+2:2N]
        @. val[2N:3N-2] = Ψ[1:N-1] * Ψ[2:N] + Ψ[N+1:2N-1] * Ψ[N+2:2N]
    else 
        # reshape the vector Ψ to the (antisymmetric) tensor
        for j = 1:length(combBasis)
            ij = combBasis[j]
            for k = 1:length(p)
                ik = seq2num_ns(2N, n, ij[p[k]])
                Ψtensor[ik] = Ψ[j] * ε[k]
            end
        end

        mass = overlap(n - 1, C)
        for k = 1:2*N
            sptr = zeros(Int, n - 1, 1)
            for s = 1:2^(n-1)
                sp = sptr * N
                uk = getindex(Ψtensor, k, ntuple(x -> sp[x]+1:sp[x]+N, n - 1)...)
                ukvec = reshape(uk, N^(n - 1), 1)[:]
                if k <= N
                    val[k] += dot(ukvec, mass, ukvec)
                else
                    val[k-N] += dot(ukvec, mass, ukvec)
                end
                if k < N
                    uk_right = getindex(Ψtensor, k + 1, ntuple(x -> sp[x]+1:sp[x]+N, n - 1)...)
                    ukvec_right = reshape(uk_right, N^(n - 1), 1)[:]
                    val[N+k] += dot(ukvec, mass, ukvec_right)
                    val[2*N+k-1] += dot(ukvec, mass, ukvec_right)
                end
                if k > N && k < 2 * N
                    uk_right = getindex(Ψtensor, k + 1, ntuple(x -> sp[x]+1:sp[x]+N, n - 1)...)
                    ukvec_right = reshape(uk_right, N^(n - 1), 1)[:]
                    val[k] += dot(ukvec, mass, ukvec_right)
                    val[N+k-1] += dot(ukvec, mass, ukvec_right)
                end
                # adjust sptr
                sptr[1] += 1
                if n >= 3
                    for ℓ = 1:n-2
                        if sptr[ℓ] == 2
                            sptr[ℓ] = 0
                            sptr[ℓ+1] += 1
                        end
                    end
                end # end if
            end
        end
    end
    ρcoef = sparse(indrow, indcol, val)
    return ρcoef * n / length(p)
end

# compute the value ρ(x) with the coefficients
function density(x::Float64, L::Float64, coef::SparseMatrixCSC{Float64,Int64})
    val = 0.0
    N = coef.n
    h = (2.0 * L) / (N + 1)
    j = floor(Int64, (x + L) / h)
    ϕleft = ((j + 1) * h - L - x) / h
    ϕright = (x + L - j * h) / h
    if j > 0 && j < N + 1
        val += ϕleft^2 * coef[j, j]
    end
    if j < N
        val += ϕright^2 * coef[j+1, j+1]
    end
    if j > 0 && j < N
        val += ϕleft * ϕright * coef[j, j+1] + ϕleft * ϕright * coef[j+1, j]
    end
    return val
end

density(x::Array{Float64}, L::Float64, coef::SparseMatrixCSC{Float64,Int64}) =
    [density(x[i], L, coef) for i = 1:length(x)]

density(x::Array{Float64}, n::Int64, Ψ::Array{Float64,1}, ham::ham1d) = density(x, ham.L, density_coef(n, Ψ, ham.C))


# compute the coefficients for pair pair_density
function pair_density_coef(n::Int64, Ψ::Array{Float64,1}, C::SparseMatrixCSC{Float64,Int64})
    N = C.n
    Ψtensor = zeros(Float64, ntuple(x -> 2 * N, n))
    basis1body = [i for i = 1:2*N]
    combBasis = collect(combinations(basis1body, n))
    # computate the permutations and paritiy
    v = [i for i = 1:n]
    p = collect(permutations(v))[:]
    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
    # reshape the vector Ψ to the (antisymmetric) tensor
    for j = 1:length(combBasis)
        ij = combBasis[j]
        for k = 1:length(p)
            ik = seq2num_ns(2N, n, ij[p[k]])
            Ψtensor[ik] = Ψ[j] * ε[k]
        end
    end
    # integrate the n-2 variable out to obtain the coefficients of ρ2
    # coefficients stored in a (2N+3)×(2N+3) matrix: N+2 for ϕ_iϕ_i and (N+1) for ϕ_iϕ_i+1
    coef = zeros(Float64, 2 * N + 3, 2 * N + 3)
    # only necessary to perform integration for more than 2-electron systems
    if n == 2
        for j = 1:2*N, k = 1:2*N
            jp = j < N + 1 ? j : j - N
            kp = k < N + 1 ? k : k - N
            coef[jp+1, kp+1] += Ψtensor[j, k]^2 #i1=i2,j1=j2
            if jp < N
                coef[N+2+jp+1, kp+1] += Ψtensor[j, k] * Ψtensor[j+1, k] #i2=i1+1,j1=j2
            end
            if kp < N
                coef[jp+1, N+2+kp+1] += Ψtensor[j, k] * Ψtensor[j, k+1] #i1=i2,j2=j1+1
            end
            if jp < N && kp < N
                coef[N+2+jp+1, N+2+kp+1] += Ψtensor[j, k] * Ψtensor[j+1, k+1] #i2=i1+1,j2=j1+1
            end
        end
    else
        mass = overlap(n - 2, C)
        for j = 1:2*N, k = 1:2*N
            jp = j < N + 1 ? j : j - N
            kp = k < N + 1 ? k : k - N
            sptr = zeros(Int, n - 2, 1)
            for s = 1:2^(n-2)
                sp = sptr * N
                u = getindex(Ψtensor, j, k, ntuple(x -> sp[x]+1:sp[x]+N, n - 2)...)
                uvec = reshape(u, N^(n - 2), 1)[:]
                coef[jp+1, kp+1] += dot(uvec, mass, uvec)
                if jp < N
                    u_xr = getindex(Ψtensor, j + 1, k, ntuple(x -> sp[x]+1:sp[x]+N, n - 2)...)
                    uvec_xr = reshape(u_xr, N^(n - 2), 1)[:]
                    coef[N+2+jp+1, kp+1] += dot(uvec_xr, mass, uvec)
                end
                if kp < N
                    u_yr = getindex(Ψtensor, j, k + 1, ntuple(x -> sp[x]+1:sp[x]+N, n - 2)...)
                    uvec_yr = reshape(u_yr, N^(n - 2), 1)[:]
                    coef[jp+1, N+2+kp+1] += dot(uvec_yr, mass, uvec)
                end
                if jp < N && kp < N
                    coef[N+2+jp+1, N+2+kp+1] += dot(uvec_xr, mass, uvec_yr)
                end
                # adjust sptr
                sptr[1] += 1
                if n >= 4
                    for ℓ = 1:n-3
                        if sptr[ℓ] == 2
                            sptr[ℓ] = 0
                            sptr[ℓ+1] += 1
                        end
                    end
                end # end if
            end
        end
    end # end if n == 2
    return coef * n * (n - 1) / 2 / length(p)
end

# compute the value ρ2(x) with the coefficients
function pair_density(x::Float64, y::Float64, L::Float64, coef::Array{Float64,2})
    val = 0.0
    N = floor(Int, (size(coef)[1] - 3) / 2)
    h = (2.0 * L) / (N + 1)
    j = floor(Int64, (x + L) / h)
    ϕx_l = ((j + 1) * h - L - x) / h
    ϕx_r = (x + L - j * h) / h
    vecx = [ϕx_l * ϕx_l, ϕx_l * ϕx_r, ϕx_r * ϕx_l, ϕx_r * ϕx_r]
    k = floor(Int64, (y + L) / h)
    ϕy_l = ((k + 1) * h - L - y) / h
    ϕy_r = (y + L - k * h) / h
    vecy = [ϕy_l * ϕy_l, ϕy_l * ϕy_r, ϕy_r * ϕy_l, ϕy_r * ϕy_r]
    if j < N + 1 && k < N + 1
        mat_coef = [coef[j+1, k+1] coef[j+1, N+2+k+1] coef[j+1, N+2+k+1] coef[j+1, k+2]
            coef[N+2+j+1, k+1] coef[N+2+j+1, N+2+k+1] coef[N+2+j+1, N+2+k+1] coef[N+2+j+1, k+2]
            coef[N+2+j+1, k+1] coef[N+2+j+1, N+2+k+1] coef[N+2+j+1, N+2+k+1] coef[N+2+j+1, k+2]
            coef[j+2, k+1] coef[j+2, N+2+k+1] coef[j+2, N+2+k+1] coef[j+2, k+2]
        ]
        val += vecx' * mat_coef * vecy
    end
    return val
end

pair_density(xy::Array{Float64,2}, L::Float64, coef::Array{Float64,2}) =
    [pair_density(xy[i, 1], xy[j, 2], L, coef) for i = 1:size(xy, 1) for j = 1:size(xy, 1)]

pair_density(xy::Array{Float64,2}, L::Float64, n::Int64, Ψ::Array{Float64,1}, C::SparseMatrixCSC{Float64,Int64}) = pair_density(xy, L, pair_density_coef(n, Ψ, C))


# compute the one-body reduced density matrix from Ψ
function one_body_DM(Ψ::Array{Float64,1})

end

# compute the coefficients for pair pair_density_spin
function pair_density_spin_coef(n::Int64, Ψ::Array{Float64,1}, C::SparseMatrixCSC{Float64,Int64}, s1::Int64, s2::Int64)
    N = C.n
    Ψtensor = zeros(Float64, ntuple(x -> 2 * N, n))
    basis1body = [i for i = 1:2*N]
    combBasis = collect(combinations(basis1body, n))
    # computate the permutations and paritiy
    v = [i for i = 1:n]
    p = collect(permutations(v))[:]
    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
    # reshape the vector Ψ to the (antisymmetric) tensor
    for j = 1:length(combBasis)
        ij = combBasis[j]
        for k = 1:length(p)
            ik = seq2num_ns(2N, n, ij[p[k]])
            Ψtensor[ik] = Ψ[j] * ε[k]
        end
    end
    # integrate the n-2 variable out to obtain the coefficients of ρ2
    # coefficients stored in a (2N+3)×(2N+3) matrix: N+2 for ϕ_iϕ_i and (N+1) for ϕ_iϕ_i+1
    coef = zeros(Float64, 2 * N + 3, 2 * N + 3)
    # only necessary to perform integration for more than 2-electron systems
    if n == 2
        for jp = 1:N, kp = 1:N
            j = jp + s1 * N
            k = kp + s2 * N
            coef[jp+1, kp+1] += Ψtensor[j, k]^2 #i1=i2,j1=j2
            if jp < N
                coef[N+2+jp+1, kp+1] += Ψtensor[j, k] * Ψtensor[j+1, k] #i2=i1+1,j1=j2
            end
            if kp < N
                coef[jp+1, N+2+kp+1] += Ψtensor[j, k] * Ψtensor[j, k+1] #i1=i2,j2=j1+1
            end
            if jp < N && kp < N
                coef[N+2+jp+1, N+2+kp+1] += Ψtensor[j, k] * Ψtensor[j+1, k+1] #i2=i1+1,j2=j1+1
            end
        end
    else
        mass = overlap(n - 2, C)
        for jp = 1:N, kp = 1:N
            j = jp + s1 * N
            k = kp + s2 * N
            sptr = zeros(Int, n - 2, 1)
            for s = 1:2^(n-2)
                sp = sptr * N
                u = getindex(Ψtensor, j, k, ntuple(x -> sp[x]+1:sp[x]+N, n - 2)...)
                uvec = reshape(u, N^(n - 2), 1)[:]
                coef[jp+1, kp+1] += dot(uvec, mass, uvec)
                if jp < N
                    u_xr = getindex(Ψtensor, j + 1, k, ntuple(x -> sp[x]+1:sp[x]+N, n - 2)...)
                    uvec_xr = reshape(u_xr, N^(n - 2), 1)[:]
                    coef[N+2+jp+1, kp+1] += dot(uvec_xr, mass, uvec)
                end
                if kp < N
                    u_yr = getindex(Ψtensor, j, k + 1, ntuple(x -> sp[x]+1:sp[x]+N, n - 2)...)
                    uvec_yr = reshape(u_yr, N^(n - 2), 1)[:]
                    coef[jp+1, N+2+kp+1] += dot(uvec_yr, mass, uvec)
                end
                if jp < N && kp < N
                    coef[N+2+jp+1, N+2+kp+1] += dot(uvec_xr, mass, uvec_yr)
                end
                # adjust sptr
                sptr[1] += 1
                if n >= 4
                    for ℓ = 1:n-3
                        if sptr[ℓ] == 2
                            sptr[ℓ] = 0
                            sptr[ℓ+1] += 1
                        end
                    end
                end # end if
            end
        end
    end # end if n == 2
    return coef * n * (n - 1) / 2 / length(p)
end

pair_density_spin(xy::Array{Float64,2}, s1::Int64, s2::Int64, L::Float64, n::Int64, Ψ::Array{Float64,1}, C::SparseMatrixCSC{Float64,Int64}) = pair_density(xy, L, pair_density_spin_coef(n, Ψ, C, s1, s2))

pair_density_spin(xy::Array{Float64,2}, s1::Int64, s2::Int64, Ψ::WaveFunction_full, ham::ham1d) = pair_density_spin(xy, s1, s2, ham.L, Ψ.ne, Ψ.wf, ham.C)

#------------------------------------------------------------------------------------------
# density for 2D
# compute the value ρ(r), r=(x,y) with the coefficients
function density(x::Float64, y::Float64, Lx::Float64, Ly::Float64, Nx::Int64, Ny::Int64, coef::SparseMatrixCSC{Float64,Int64})
    val = 0.0
    Nx = Nx - 1
    Ny = Ny - 1
    hx = (2.0 * Lx) / (Nx + 1)
    hy = (2.0 * Ly) / (Ny + 1)

    i = floor(Int64, (x + Lx) / hx)
    ϕileft = ((i + 1) * hx - Lx - x) / hx
    ϕiright = (x + Lx - i * hx) / hx

    j = floor(Int64, (y + Ly) / hy)
    ϕjleft = ((j + 1) * hy - Ly - y) / hy
    ϕjright = (y + Ly - j * hy) / hy

    if i > 0 && i < Nx + 1
        if j > 0 && j < Ny + 1
            val += ϕileft^2 * ϕjleft^2 * coef[i+(j-1)*Nx, i+(j-1)*Nx]
        elseif j < Ny
            val += ϕileft^2 * ϕjright^2 * coef[i+j*Nx, i+j*Nx]
        elseif j > 0 && j < Ny
            val += ϕileft^2 * (ϕjleft * ϕjright * coef[i+(j-1)*Nx, i+j*Nx] + ϕjleft * ϕjright * coef[i+j*Nx, i+(j-1)*Nx])
        end
    elseif i < Nx
        if j > 0 && j < Ny + 1
            val += ϕiright^2 * ϕjleft^2 * coef[i+1+(j-1)*Nx, i+1+(j-1)*Nx]
        elseif j < Ny
            val += ϕiright^2 * ϕjright^2 * coef[i+1+j*Nx, i+1+j*Nx]
        elseif j > 0 && j < Ny
            val += ϕiright^2 * (ϕjleft * ϕjright * coef[i+1+(j-1)*Nx, i+1+j*Nx] + ϕjleft * ϕjright * coef[i+1+j*Nx, i+1+(j-1)*Nx])
        end
    elseif i > 0 && i < Nx
        if j > 0 && j < Ny + 1
            val += ϕileft * ϕiright * ϕjleft^2 * coef[i+(j-1)*Nx, i+1+(j-1)*Nx] + ϕileft * ϕiright * ϕjleft^2 * coef[i+1+(j-1)*Nx, i+(j-1)*Nx]
        elseif j < Ny
            val += ϕileft * ϕiright * ϕjright^2 * coef[i+j*Nx, i+1+j*Nx] + ϕileft * ϕiright * ϕjright^2 * coef[i+1+j*Nx, i+j*Nx]
        elseif j > 0 && j < Ny
            val += ϕileft * ϕiright * (ϕjleft * ϕjright * coef[i+(j-1)*Nx, i+1+j*Nx] + ϕjleft * ϕjright * coef[i+j*Nx, i+1+(j-1)*Nx]) + ϕileft * ϕiright * (ϕjleft * ϕjright * coef[i+1+(j-1)*Nx, i+j*Nx] + ϕjleft * ϕjright * coef[i+1+j*Nx, i+(j-1)*Nx])
        end
    end

    return val
end

density(xy::Vector{Vector{Float64}}, Lx::Float64, Ly::Float64, Nx::Int64, Ny::Int64, coef::SparseMatrixCSC{Float64,Int64}) =
    [density(xy[1][i], xy[2][j], Lx, Ly, Nx, Ny, coef) for i = 1:length(xy[1]) for j = 1:length(xy[2])]

density(xy::Vector{Vector{Float64}}, n::Int64, Ψ::Array{Float64,1}, ham::ham2d) = density(xy, ham.L[1], ham.L[2], ham.N[1], ham.N[2], density_coef(n, Ψ, ham.C))


function pair_density_coef_2ne(Ψ::Array{Float64,1}, N::Vector{Int64})
    Nx = N[1] - 1
    Ny = N[2] - 1
    dof = Nx * Ny
    n = 2
    Ψtensor = zeros(Float64, ntuple(x -> 2 * dof, n))
    basis1body = [i for i = 1:2*dof]
    combBasis = collect(combinations(basis1body, n))
    # computate the permutations and paritiy
    v = [i for i = 1:n]
    p = collect(permutations(v))[:]
    ε = (-1) .^ [parity(p[i]) for i = 1:length(p)]
    # reshape the vector Ψ to the (antisymmetric) tensor
    for j = 1:length(combBasis)
        ij = combBasis[j]
        for k = 1:length(p)
            ik = seq2num_ns(2dof, n, ij[p[k]])
            Ψtensor[ik] = Ψ[j] * ε[k]
        end
    end

    I = Int[]
    J = Int[]
    vals = Float64[]
    label1 = zeros(Int, 4)
    label2 = zeros(Int, 4)
    for i = 1:2*dof, j = 1:2*dof
        ip = i < dof + 1 ? i : i - dof
        si = i < dof + 1 ? 0 : 1
        mi = ip % Nx == 0 ? Nx : ip % Nx
        ni = Int((ip - mi) / Nx) + 1

        jp = j < dof + 1 ? j : j - dof
        tj = j < dof + 1 ? 0 : 1
        mj = jp % Nx == 0 ? Nx : jp % Nx
        nj = Int((jp - mj) / Nx) + 1

        label1[1] = Nx * (ni - 1) + mi#(i,j)
        label1[2] = label1[1] + 1#(i+1,j)
        label1[3] = Nx * ni + mi#(i,j+1)
        label1[4] = label1[3] + 1#(i+1,j+1)
        label2[1] = Nx * (nj - 1) + mj#(i,j-1)
        label2[2] = label2[1] + 1#(i,j-1)
        label2[3] = Nx * nj + mj#(i,j)
        label2[4] = label2[3] + 1#(i,j)
        if ni == Ny
            label1[3] = -1
            label1[4] = -1
        end
        if mi == Nx
            label1[2] = -1
            label1[4] = -1
        end
        if nj == Ny
            label2[3] = -1
            label2[4] = -1
        end
        if mj == Nx
            label2[2] = -1
            label2[4] = -1
        end

        for k1 = 1:4, k2 = 1:4
            m1 = label1[k1]
            m2 = label2[k2]
            if m1 > 0 && m2 > 0
                val = Ψtensor[i, j] * Ψtensor[m1+si*dof, m2+tj*dof]
                if val != 0
                    push!(I, (m1 - 1) * dof + ip)
                    push!(J, (m2 - 1) * dof + jp)
                    push!(vals, val)
                end
            end
        end
    end

    return sparse(I, J, vals, dof^2, dof^2) / length(p)
end

function pair_density_2ne(x1::Float64, y1::Float64, x2::Float64, y2::Float64, L::Vector{Float64}, N::Vector{Int64}, coef::SparseMatrixCSC{Float64,Int64})

    val = 0.0
    Lx = L[1]
    Ly = L[2]
    Nx = N[1] - 1
    Ny = N[2] - 1
    dof = Nx * Ny
    hx = (2.0 * Lx) / (Nx + 1)
    hy = (2.0 * Ly) / (Ny + 1)

    ix1 = floor(Int64, (x1 + Lx) / hx)# Index of the basis function on the left of x1
    ϕx1_l = ((ix1 .+ 1) .* hx .- Lx .- x1) ./ hx
    ϕx1_r = (x1 .+ Lx .- ix1 .* hx) ./ hx
    vecx1 = 0 < ix1 < Nx ? [ϕx1_l * ϕx1_l, ϕx1_l * ϕx1_r, ϕx1_r * ϕx1_l, ϕx1_r * ϕx1_r] : (ix1 == 0 ? [ϕx1_r * ϕx1_r] : [ϕx1_l * ϕx1_l])
    indx1 = 0 < ix1 < Nx ? [[ix1, ix1], [ix1, ix1 + 1], [ix1, ix1 + 1], [ix1 + 1, ix1 + 1]] : (ix1 == 0 ? [[ix1 + 1, ix1 + 1]] : [[ix1, ix1]])

    jy1 = floor.(Int64, (y1 .+ Ly) ./ hy)# Index of the basis function on the left of y1
    ϕy1_l = ((jy1 .+ 1) .* hy .- Ly .- y1) ./ hy
    ϕy1_r = (y1 .+ Ly .- jy1 .* hy) ./ hy
    vecy1 = 0 < jy1 < Ny ? [ϕy1_l * ϕy1_l, ϕy1_l * ϕy1_r, ϕy1_r * ϕy1_l, ϕy1_r * ϕy1_r] : (jy1 == 0 ? [ϕy1_r * ϕy1_r] : [ϕy1_l * ϕy1_l])
    indy1 = 0 < jy1 < Ny ? [[jy1, jy1], [jy1, jy1 + 1], [jy1, jy1 + 1], [jy1 + 1, jy1 + 1]] : (jy1 == 0 ? [[jy1 + 1, jy1 + 1]] : [[jy1, jy1]])

    vecr1 = vecy1 .* vecx1[1]
    for vx in vecx1[2:end]
        append!(vecr1, vecy1 .* vx)
    end

    ind1 = Int[]
    for i in indx1, j in indy1
        push!(ind1, (i[1] + Nx * (j[1] - 1)) + dof * (i[2] + Nx * (j[2] - 1) - 1))
    end

    ix2 = floor(Int64, (x2 + Lx) / hx)# Index of the basis function on the left of x2
    ϕx2_l = ((ix2 .+ 1) .* hx .- Lx .- x2) ./ hx
    ϕx2_r = (x2 .+ Lx .- ix2 .* hx) ./ hx
    vecx2 = 0 < ix2 < Nx ? [ϕx2_l * ϕx2_l, ϕx2_l * ϕx2_r, ϕx2_r * ϕx2_l, ϕx2_r * ϕx2_r] : (ix2 == 0 ? [ϕx2_r * ϕx2_r] : [ϕx2_l * ϕx2_l])
    indx2 = 0 < ix2 < Nx ? [[ix2, ix2], [ix2, ix2 + 1], [ix2, ix2 + 1], [ix2 + 1, ix2 + 1]] : (ix2 == 0 ? [[ix2 + 1, ix2 + 1]] : [[ix2, ix2]])

    jy2 = floor.(Int64, (y2 .+ Ly) ./ hy)# Index of the basis function on the left of y2
    ϕy2_l = ((jy2 .+ 1) .* hy .- Ly .- y2) ./ hy
    ϕy2_r = (y2 .+ Ly .- jy2 .* hy) ./ hy
    vecy2 = 0 < jy2 < Ny ? [ϕy2_l * ϕy2_l, ϕy2_l * ϕy2_r, ϕy2_r * ϕy2_l, ϕy2_r * ϕy2_r] : (jy2 == 0 ? [ϕy2_r * ϕy2_r] : [ϕy2_l * ϕy2_l])
    indy2 = 0 < jy2 < Ny ? [[jy2, jy2], [jy2, jy2 + 1], [jy2, jy2 + 1], [jy2 + 1, jy2 + 1]] : (jy2 == 0 ? [[jy2 + 1, jy2 + 1]] : [[jy2, jy2]])

    vecr2 = vecy2 .* vecx2[1]
    for vx in vecx2[2:end]
        append!(vecr2, vecy2 .* vx)
    end

    ind2 = Int[]
    for i in indx2, j in indy2
        push!(ind2, (i[1] + Nx * (j[1] - 1)) + dof * (i[2] + Nx * (j[2] - 1) - 1))
    end

    if ix1 < Nx + 1 && jy1 < Ny + 1 && ix2 < Nx + 1 && jy2 < Ny + 1
        #println("x1 : $(x1), y1 : $(y1), x2 : $(x2), y2 : $(y2) | ind1 : $(ind1), ind2 : $(ind2)\n")
        val += vecr1' * coef[ind1, ind2] * vecr2
    end

    return val
end

pair_density_2ne(xy::Vector{Vector{Float64}}, L::Vector{Float64}, N::Vector{Int64}, coef::SparseMatrixCSC{Float64,Int64}) = [pair_density_2ne(xy[1][i], xy[2][j], xy[3][k], xy[4][l], L, N, coef) for i = 1:length(xy[1]) for j = 1:length(xy[2]) for k = 1:length(xy[3]) for l = 1:length(xy[4])]

pair_density_2ne(xy::Vector{Vector{Float64}}, Ψ::Array{Float64,1}, ham::ham2d) = pair_density_2ne(xy, ham.L, ham.N, pair_density_coef_2ne(Ψ, ham.N))
