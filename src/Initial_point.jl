#Newton-Raphson
using LinearAlgebra
using NLsolve
using ForwardDiff

export InitPT

# Newton-Raphson method
function NR(x::Vector{Float64}, f::Function; k=100)
    report(i, residual) = @info "$i-th iteration: residual = $residual"

    for i in 1:k
        #global x
        J = ForwardDiff.jacobian(f, x)
        F = lu(J, check=false)

        if F.info != 0 # singular Jacobian
            F = lu(J + sqrt(eps(maximum(J))) * I)
        end

        x -= F \ f(x)
        residual = norm(f(x), Inf)
        #residual <= 100eps() && (report(i, residual); break)
        residual <= 100eps() && break
    end
    return x
end

function grid_point(ne::Int64, ham::ham1d, Nc::Int64, ind::Vector{Int64})

    L = ham.L
    h = @. 2L / Nc

    xy = @. -L + h * (ind - 1)
end

function grid_point(ne::Int64, ham::ham2d, Nc::Vector{Int64}, ind::Vector{Int64})

    Lx = ham.L[1]
    Ly = ham.L[2]
    hx = 2Lx / Nc[1]
    hy = 2Ly / Nc[2]
    Nx = Nc[1] + 1
    Ny = Nc[2] + 1

    xy = zeros(2ne)
    for i = 1:ne
        ni = ind[i]
        lx = ni % Nx == 0 ? Nx : ni % Nx
        ly = (ni - lx) / Nx + 1
        xy[i] = lx
        xy[i+ne] = ly
    end

    @. xy[1:ne] = -Lx + hx * (xy[1:ne] - 1)
    @. xy[ne+1:2ne] = -Ly + hy * (xy[ne+1:2ne] - 1)

    return xy
end

function Init_coarse(ne::Int64, ham::Hamiltonian, F::Function; Nc = Nc)

    dof = prod(Nc .+ 1)
    basis1body = [i for i = 1:dof]
    comb = collect(combinations(basis1body, ne))

    gp = map(x -> grid_point(ne, ham, Nc, x), comb)
    l = collect(1:length(gp))

    return gp, l
end

# num : sample number 
# find global minimizers limited on [-L,L]
function InitPT(ne::Int64, ham::ham1d; num=500, a0 = nothing, Nc = cld.(ham.N,2))

    L = ham.L
    vee = ham.vee
    vext = ham.vext

    dvext(x) = @. ForwardDiff.derivative(x2 -> vext(x2), x)
    ddvext(x) = @. ForwardDiff.derivative(x2 -> dvext(x2), x)
    dvee(x) = @. ForwardDiff.derivative(x2 -> vee(x2), x)

    # derivate function
    function f(x; dvee=dvee, dvext=dvext)

        fvec = dvext(x)
        ne = length(x)
        for i = 1:ne
            for j = vcat(1:i-1, i+1:ne)
                fvec[i] += dvee(x[i] - x[j])
            end
        end
        fvec
    end

    # SCE function
    function F(x; ne=ne, vee=vee, vext=vext)
        Fv = sum(vext.(x))
        for i = 1:ne-1
            for j = i+1:ne
                Fv += vee(x[i] - x[j])
            end
        end
        Fv
    end

    if a0 == nothing 

        a0 = [(rand(ne) .- 0.5) .* 2L for i = 1:num]

        # find the local minimizers of vext
        a1 = find_zeros(dvext, -L, L)
        a1 = a1[findall(x -> x > 0.0, ddvext.(a1))]
        length(a1) == ne && push!(a0, a1)

    end

    r = sort!.(NR.(a0, f))
    unique!(r)

    l = Int64[]
    rm = zeros(ne - 1)
    for i = 1:length(r)
        ri = @views r[i]
        if norm(ri, Inf) <= L && abs(ri[1] - ri[2]) > 1e-4 && abs(ri[ne-1] - ri[ne]) > 1e-4
            @views r1 = ri[1:ne-1]
            @views r2 = ri[2:ne]
            @. rm = r1 - r2
            if minimum(abs.(rm)) > 1e-4
                push!(l, i)
            end
        end
    end

    if length(l) != 0
        Fv = F.(r[l])
        Fmin = findmin(Fv)[1]
    else
        @warn "Inaccurate initial point"
        for k = 1:10
            r, l = Init_coarse(ne, ham, F; Nc=Nc)
            Fv = F.(r)
            Fmin = findmin(Fv)[1]
            abs(Fmin) == Inf ? Nc .+= 1 : break
        end
    end

    i0 = findall(x -> x < Fmin + 0.001, Fv)
    l0 = l[i0]
    r0 = r[l0]

    return r0
end

function InitPT(ne::Int64, ham::ham2d; num=500, a0 = nothing, Nc = cld.(ham.N,2))

    Lx = ham.L[1]
    Ly = ham.L[2]
    vee = ham.vee
    vext = ham.vext

    dvext_dx(x, y) = ForwardDiff.derivative(x -> vext(x, y), x)
    dvext_dy(x, y) = ForwardDiff.derivative(y -> vext(x, y), y)
    dvee_dx(x, y) = ForwardDiff.derivative(x -> vee(x, y), x)
    dvee_dy(x, y) = ForwardDiff.derivative(y -> vee(x, y), y)

    # derivate function
    function f(x; ne=ne, dvee_dx=dvee_dx, dvee_dy=dvee_dy, dvext_dx=dvext_dx, dvext_dy=dvext_dy)

        fvec = vcat(dvext_dx.(x[1:ne], x[ne+1:2ne]), dvext_dy.(x[1:ne], x[ne+1:2ne]))
        for i = 1:ne
            for j = vcat(1:i-1, i+1:ne)
                fvec[i] += dvee_dx(x[i] - x[j], x[i+ne] - x[j+ne])
                fvec[i+ne] += dvee_dy(x[i] - x[j], x[i+ne] - x[j+ne])
            end
        end
        fvec
    end

    # SCE function
    function F(x; ne=ne, vee=vee, vext=vext)
        Fv = sum(vext.(x[1:ne], x[ne+1:2ne]))
        for i = 1:ne-1
            for j = i+1:ne
                Fv += vee(x[i] - x[j], x[i+ne] - x[j+ne])
            end
        end
        Fv
    end
    
    if a0 == nothing
        a0 = [vcat((rand(ne) .- 0.5) .* 2Lx, (rand(ne) .- 0.5) .* 2Ly) for i = 1:num]

        # find the local minimizers of vext
        df(x, y) = SVector(dvext_dx(x, y), dvext_dy(x, y))
        df(X) = df(X...)
        X = (-Lx .. Lx) × (-Ly .. Ly)
        rts = IntervalRootFinding.roots(df, X, IntervalRootFinding.Bisection, 1e-3)
        rts = IntervalRootFinding.roots(df, rts, IntervalRootFinding.Bisection)
        rr = [[rts[i].interval[1].lo, rts[i].interval[2].lo] for i = 1:length(rts)]
        J = map(x -> ForwardDiff.jacobian(df, x), rr)
        rr = rr[findall(x -> isone(prod(x .> 0.0)), eigvals.(J))]
        if length(rr) == ne
            a1 = zeros(ne,2)
            for i = 1:ne
                a1[i,:] = rr[i]
            end 
            push!(a0, a1[:])
        end
    end
    r = NR.(a0, f)
    unique!(r)

    l = Int64[]
    rm = zeros(ne - 1)
    for i = 1:length(r)
        ri = @views r[i]
        if norm(ri[1:ne], Inf) <= Lx && norm(ri[ne+1:2ne], Inf) <= Ly && abs(ri[1] - ri[2]) > 1e-4 && abs(ri[ne-1] - ri[ne]) > 1e-4
            @views r1 = ri[1:ne-1]
            @views r2 = ri[2:ne]
            @. rm = r1 - r2
            if minimum(abs.(rm)) > 1e-4
                push!(l, i)
            end
        end
    end

    if length(l) != 0
        Fv = F.(r[l])
        Fmin = findmin(Fv)[1]
    else
        @warn "Inaccurate initial point"
        for k = 1:10      
            r, l = Init_coarse(ne, ham, F; Nc=Nc)
            Fv = F.(r)
            Fmin = findmin(Fv)[1]
            abs(Fmin) == Inf ? Nc .+= 1 : break
        end
    end

    i0 = findall(x -> x < Fmin + 0.001, Fv)
    l0 = l[i0]
    r0 = r[l0]

    return r0
end