# Implementation from Matlab translated to Julia
using SparseArrays


function mat_2d_fermion_coul(epsilon, L, N)
    # generate the stiff matrix for \epsilon\Delta+1/|x-y|
    # note that the wavefunctions for fermions are antisymmetric, we use basis functions
    # \phi_i(x)\phi_j(y) - \phi_j(x)\phi_i(y)
    # number of basis functions
    #  _____________________________
    # |     0        | (N-1)(N-2)/2 |
    # |______________|______________|
    # | (N-1)(N-2)/2 |  (N-1)(N-1)  |
    # |______________|______________|

    h = 2.0 * L / N;
    n = 2 * N;                       # consider the spin, dof is double in each direction
    dof = (N-1) * (2 * N - 3);       # number of basis functions
    mat_lap  = zeros(dof, dof);
    HLap  = zeros(dof, dof);
    mat_mass = zeros(dof, dof);

    rank = zeros(Int,dof, 2);
    #Label matrix: save labels and coordinates for 4 points in each element
    label_matrix = zeros(Int,n+1,n+1);
    is_p = zeros(Int,n+1,n+1);
    count = 0;
    for i=2:n
        for j=(i+1):n
            if j!=(N+1) && i!=(N+1)
                count = count+1;
                label_matrix[i,j] = count;
                label_matrix[j,i] = count;
                is_p[i,j] = 1.0;
                is_p[j,i] = -1.0;
                rank[count,:] = [i,j];
            end
        end
    end

    NPX = 4;
    NPY = 3;
    gauss_x = [-0.8611363116 -0.3399810435 0.3399810435 0.8611363116];
    weight_x = [0.3478548451 0.6521451548 0.6521451548 0.3478548451];
    gauss_y = [-0.7745966692 0.0 0.7745966692];
    weight_y = [0.5555555555 0.8888888889 0.5555555555];

    weight = weight_y'*weight_x;
    weight = reshape(weight,NPX*NPY,1);
    wt = weight * h^2/4.0;

    #points with respect to [-1,1]
    x = ones(NPY,1)*gauss_x;
    x = reshape(x,NPX*NPY,1);
    y = gauss_y'*ones(1,NPX);
    y = reshape(y,NPX*NPY,1);

    # value and gradient of basis functions at (x,y)
    f  = hcat((x.-1.0).*(y.-1.0)/4.0, .-(x.+1.0).*(y.-1.0)/4.0,
          .-(x.-1.0).*(y.+1.0)/4.0, (x.+1.0).*(y.+1.0)/4.0);
    gy = hcat((y.-1.0)/4.0, .-(y.-1.0)/4.0, .-(y.+1.0)/4.0, (y.+1.0)/4.0) * 2.0/h;
    gx = hcat((x.-1.0)/4.0, .-(x.+1.0)/4.0, .-(x.-1.0)/4.0, (x.+1.0)/4.0 ) * 2.0/h;


    for i = 1 : n
        for j = 1 : n
            #----------- label the 4 nodes in each elements ----------#
            label = [label_matrix[i,j], label_matrix[i,j+1],
                     label_matrix[i+1,j],label_matrix[i+1,j+1]];
            isp = [is_p[i,j], is_p[i,j+1], is_p[i+1,j], is_p[i+1,j+1]];
            ind = findall(x->x!=0,label);

            # coordinates
            xx = -L .+ h * (j.-0.5.+x/2 .- N*(j>N) );
            yy = -L .+ h * (i.-0.5.+y/2 .- N*(i>N) );
            r = abs.(xx - yy);

            #--------------- calculate stiff elements -----------------#
            for k = ind
                for l = ind
                    mx = label[k];
                    my = label[l];
                    ix = isp[k];
                    iy = isp[l];
                    HLap[mx,my] = HLap[mx,my] + ix * iy *
                             sum( (gx[:,k] .* gx[:,l] + gy[:,k] .* gy[:,l]) .* wt );
                    mat_lap[mx,my] = mat_lap[mx,my] + ix * iy *
                             sum( (epsilon * (gx[:,k] .* gx[:,l] + gy[:,k] .* gy[:,l]) +
                             1.0 ./r .* f[:,k] .* f[:,l]) .* wt );
                    mat_mass[mx,my] = mat_mass[mx,my] + ix * iy * sum( f[:,k] .* f[:,l] .* wt );
                end
            end
        end
    end

    HLap  = sparse(HLap);
    mat_lap  = sparse(mat_lap);
    mat_mass = sparse(mat_mass);

    return mat_lap, mat_mass, rank, HLap
end



function mat_3d_fermion_coul(L, N)
    # generate the mass matrix for 3d
    # number of basis functions
    #  y____________________________
    # |     0        | (N-1)(N-2)/2 |
    # |______________|______________|
    # | (N-3)(N-2)/2 |  (N-2)(N-1)  |
    # |______________|______________x
    #  z____________________________
    # |     0        | (N-3)(N-2)/2 |
    # |______________|______________|
    # | (N-2)(N-1)/2 |  (N-2)(N-1)  |
    # |______________|______________y
    # dof =  1/2 * [ N(N+1) + .. + 1(1+2) ] = 1/6 (2n-2)(2n-3)(2n-4) = C_{2n-2}^3

    h   = 2.0*L/N;
    n   = 2*N;                                  # consider the spin, dof is double in each direction
    dof = Int((2*N-2) * (2*N-3) * (2*N-4) /6);       # number of basis functions
    s_size = 64;                        # sparse size, each row and colum at most 27 nonzero elements
    IA     = ones(Int, dof * s_size);     # storage indexs for sparse matrix
    JA     = ones(Int, dof * s_size);
    A_mass = zeros(dof * s_size);    # for mass matrix
    A_lap  = zeros(dof * s_size);    # for stiff matrix
    A_v    = zeros(dof * s_size);    # for stiff matrix
    label  = zeros(Int, 8);                # save labels and coordinates for 4 points in each element, i,e,(n,x,y)
    rank   = zeros(Int, dof, 3);             # give the ijk order of each dof

    NP = 3;
    gaussP  = [-0.7745966692, 0.0, 0.7745966692];
    weightP = [0.5555555555, 0.8888888889, 0.5555555555];
    NQ = 4;
    gaussQ  = [-0.8611363116, -0.3399810435, 0.3399810435, 0.8611363116];
    weightQ = [0.3478548451, 0.6521451548, 0.6521451548, 0.3478548451];

    # make a label from points to fermionic points
    labelP2P = zeros((n-1)^3,1);
    kk = 1;
    for nk = 1:n-1
        for nj = 1:n-1
            for ni = 1:n-1
                m = (n-1)^2 * (nk-1) + (n-1) * (nj-1) + (ni-1) + 1;
                if nk < nj && nj < ni && nk != N && nj != N && ni != N
                    labelP2P[m] = kk;
                    # evaluate the rank
                    rank[kk,:] = [nk, nj, ni];
                    kk = kk + 1;
                else
                    labelP2P[m] = -1;
                end
            end
        end
    end
    # loop the elements
    for nk = 1:n
        for nj = nk:n
            for ni = nj:n
                #----------- label the 8 nodes in each elements ----------#
                label[1] = (n-1)^2 * (nk-2) + (n-1) * (nj-2) + (ni-2) + 1;
                label[2] = label[1] + 1;
                label[3] = label[1] + (n-1);
                label[4] = label[3] + 1;
                label[5] = label[1] + (n-1)^2;
                label[6] = label[5] + 1;
                label[7] = label[5] + (n-1);
                label[8] = label[7] + 1;

                if ni==1
                    label[1] = -1;  label[3] = -1;  label[5] = -1;  label[7] = -1;
                end
                if ni==n
                    label[2] = -1;  label[4] = -1;  label[6] = -1;  label[8] = -1;
                end
                if nj==1
                    label[1] = -1;  label[2] = -1;  label[5] = -1;  label[6] = -1;
                end
                if nj==n
                    label[3] = -1;  label[4] = -1;  label[7] = -1;  label[8] = -1;
                end
                if nk==1
                    label[1] = -1;  label[2] = -1;  label[3] = -1;  label[4] = -1;
                end
                if nk==n
                    label[5] = -1;  label[6] = -1;  label[7] = -1;  label[8] = -1;
                end

                for k = 1:8
                    if label[k] > 0
                        label[k] = labelP2P[label[k]];
                    end
                end
                #--------------- calculate stiff elements -----------------#
                for px = 1:NP
                    for py = 1:NP
                        for pz = 1:NQ
                            wt = weightP[px] * weightP[py] * weightQ[pz] * h^3/8.0;
                            x = gaussP[px];
                            y = gaussP[py];
                            z = gaussQ[pz];
                            # value and gradient of basis functions at (x,y)
                            f = 1.0/8 * hcat((1.0-x)*(1.0-y)*(1.0-z), (x+1.0)*(1.0-y)*(1.0-z),
                                         (1.0-x)*(y+1.0)*(1.0-z), (x+1.0)*(y+1.0)*(1.0-z),
                                         (1.0-x)*(1.0-y)*(1.0+z), (x+1.0)*(1.0-y)*(1.0+z),
                                         (1.0-x)*(y+1.0)*(1.0+z), (x+1.0)*(y+1.0)*(1.0+z) );
                            gx = 1.0/8 * hcat( -(1.0-y)*(1.0-z), (1.0-y)*(1.0-z),
                                           -(y+1.0)*(1.0-z), (y+1.0)*(1.0-z),
                                           -(1.0-y)*(1.0+z), (1.0-y)*(1.0+z),
                                           -(y+1.0)*(1.0+z), (y+1.0)*(1.0+z) ) * 2.0/h;
                            gy = 1.0/8 * hcat( -(1.0-x)*(1.0-z), -(x+1.0)*(1.0-z),
                                           (1.0-x)*(1.0-z), (x+1.0)*(1.0-z),
                                           -(1.0-x)*(1.0+z), -(x+1.0)*(1.0+z),
                                           (1.0-x)*(1.0+z), (x+1.0)*(1.0+z) ) * 2.0/h;
                            gz = 1.0/8 * hcat( -(1.0-x)*(1.0-y), -(x+1.0)*(1.0-y),
                                           -(1.0-x)*(y+1.0), -(x+1.0)*(y+1.0),
                                           (1.0-x)*(1.0-y), (x+1.0)*(1.0-y),
                                           (1.0-x)*(y+1.0), (x+1.0)*(y+1.0) ) * 2.0/h;
                            # coordinates
                            if ni <= N
                                rx = -L + h * (ni-0.5+x/2);
                            else
                                rx = -L + h * (ni-0.5+x/2 - N);
                            end
                            if nj <= N
                                ry = -L + h * (nj-0.5+y/2);
                            else
                                ry = -L + h * (nj-0.5+y/2 - N);
                            end
                            if nk <=N
                                rz = -L + h * (nk-0.5+z/2);
                            else
                                rz = -L + h * (nk-0.5+z/2 - N);
                            end
                            val = 1.0/abs(rx-rz) + 1.0/abs(ry-rz);

                            for k = 1:8
                                for l = 1:8
                                    mk = label[k];
                                    ml = label[l];
                                    index = 64*(mk-1) + 8*(k-1) + l;
                                    if mk>0 && ml>0
                                        IA[index] = mk;
                                        JA[index] = ml;
                                        A_v[index] = A_v[index] + val * f[k] * f[l] * wt;
                                        A_mass[index] = A_mass[index] + f[k] * f[l] * wt;
                                        A_lap[index] = A_lap[index] + ( gx[k] * gx[l] + gy[k] * gy[l] + gz[k] * gz[l] ) * wt;
                                    end
                                end
                            end
                            # deal with where pertubation basis at same element. note that it is not possible that ni = nj = nk
                            if ni == nj
                                if label[2]>0
                                    m = label[2];
                                    index = 64*(m-1) + 8*(2-1) + 2;
                                    A_v[index] = A_v[index] - val * f[2] * f[3] * wt;
                                    A_mass[index] = A_mass[index] - f[2] * f[3] * wt;
                                    A_lap[index] = A_lap[index] - ( gx[2] * gx[3] + gy[2] * gy[3] + gz[2] * gz[3] ) * wt;
                                    if label[6]>0
                                        index = 64*(m-1) + 8*(2-1) + 6;
                                        A_v[index] = A_v[index] - val * f[2] * f[7] * wt;
                                        A_mass[index] = A_mass[index] - f[2] * f[7] * wt;
                                        A_lap[index] = A_lap[index] - ( gx[2] * gx[7] + gy[2] * gy[7] + gz[2] * gz[7] ) * wt;
                                    end
                                end
                                if label[6]>0
                                    m = label[6];
                                    index = 64*(m-1) + 8*(6-1) + 6;
                                    A_v[index] = A_v[index] - val * f[6] * f[7] * wt;
                                    A_mass[index] = A_mass[index] - f[6] * f[7] * wt;
                                    A_lap[index] = A_lap[index] - ( gx[6] * gx[7] + gy[6] * gy[7] + gz[6] * gz[7] ) * wt;
                                    if label[2]>0
                                        index = 64*(m-1) + 8*(6-1) + 2;
                                        A_v[index] = A_v[index] - val * f[6] * f[3] * wt;
                                        A_mass[index] = A_mass[index] - f[6] * f[3] * wt;
                                        A_lap[index] = A_lap[index] - ( gx[6] * gx[3] + gy[6] * gy[3] + gz[6] * gz[3] ) * wt;
                                    end
                                end
                            end
                            if nj == nk
                                 if label[3]>0
                                    m = label[3];
                                    index = 64*(m-1) + 8*(3-1) + 3;
                                    A_v[index] = A_v[index] - val * f[3] * f[5] * wt;
                                    A_mass[index] = A_mass[index] - f[3] * f[5] * wt;
                                    A_lap[index] = A_lap[index] - ( gx[3] * gx[5] + gy[3] * gy[5] + gz[3] * gz[5] ) * wt;
                                    if label[4]>0
                                        index = 64*(m-1) + 8*(3-1) + 4;
                                        A_v[index] = A_v[index] - val * f[3] * f[6] * wt;
                                        A_mass[index] = A_mass[index] - f[3] * f[6] * wt;
                                        A_lap[index] = A_lap[index] - ( gx[3] * gx[6] + gy[3] * gy[6] + gz[3] * gz[6] ) * wt;
                                    end
                                end
                                if label[4]>0
                                    m = label[4];
                                    index = 64*(m-1) + 8*(4-1) + 4;
                                    A_v[index] = A_v[index] - val * f[4] * f[6] * wt;
                                    A_mass[index] = A_mass[index] - f[4] * f[6] * wt;
                                    A_lap[index] = A_lap[index] - ( gx[4] * gx[6] + gy[4] * gy[6] + gz[4] * gz[6] ) * wt;
                                    if label[3]>0
                                        index = 64*(m-1) + 8*(4-1) + 3;
                                        A_v[index] = A_v[index] - val * f[4] * f[5] * wt;
                                        A_mass[index] = A_mass[index] - f[4] * f[5] * wt;
                                        A_lap[index] = A_lap[index] - ( gx[4] * gx[5] + gy[4] * gy[5] + gz[4] * gz[5] ) * wt;
                                    end
                                end
                            end
                        end
                    end
                end
                # ------ add 1/|x-y| to the stiff matrix ----- #
                for px = 1:NP
                    for py = 1:NQ
                        for pz = 1:NP
                            wt = weightP[px] * weightQ[py] * weightP[pz] * h^3/8.0;
                            x = gaussP[px];
                            y = gaussQ[py];
                            z = gaussP[pz];
                            # value and gradient of basis functions at (x,y)
                            f = 1.0/8 * hcat((1.0-x)*(1.0-y)*(1.0-z), (x+1.0)*(1.0-y)*(1.0-z),
                                         (1.0-x)*(y+1.0)*(1.0-z), (x+1.0)*(y+1.0)*(1.0-z),
                                         (1.0-x)*(1.0-y)*(1.0+z), (x+1.0)*(1.0-y)*(1.0+z),
                                         (1.0-x)*(y+1.0)*(1.0+z), (x+1.0)*(y+1.0)*(1.0+z) );
                            # coordinates
                            if ni <= N
                                rx = -L + h * (ni-0.5+x/2);
                            else
                                rx = -L + h * (ni-0.5+x/2 - N);
                            end
                            if nj <= N
                                ry = -L + h * (nj-0.5+y/2);
                            else
                                ry = -L + h * (nj-0.5+y/2 - N);
                            end
                            val = 1/abs(rx-ry);

                            for k = 1:8
                                for l = 1:8
                                    mk = label[k];
                                    ml = label[l];
                                    index = 64*(mk-1) + 8*(k-1) + l;
                                    if mk>0 && ml>0
                                        IA[index] = mk;
                                        JA[index] = ml;
                                        A_v[index] = A_v[index] + val * f[k] * f[l] * wt;
                                    end
                                end
                            end
                            # deal with where pertubation basis at same element. note that it is not possible that ni = nj = nk
                            if ni == nj
                                if label[2]>0
                                    m = label[2];
                                    index = 64*(m-1) + 8*(2-1) + 2;
                                    A_v[index] = A_v[index] - val * f[2] * f[3] * wt;
                                    if label[6]>0
                                        index = 64*(m-1) + 8*(2-1) + 6;
                                        A_v[index] = A_v[index] - val * f[2] * f[7] * wt;
                                    end
                                end
                                if label[6]>0
                                    m = label[6];
                                    index = 64*(m-1) + 8*(6-1) + 6;
                                    A_v[index] = A_v[index] - val * f[6] * f[7] * wt;
                                    if label[2]>0
                                        index = 64*(m-1) + 8*(6-1) + 2;
                                        A_v[index] = A_v[index] - val * f[6] * f[3] * wt;
                                    end
                                end
                            end
                            if nj == nk
                                 if label[3]>0
                                    m = label[3];
                                    index = 64*(m-1) + 8*(3-1) + 3;
                                    A_v[index] = A_v[index] - val * f[3] * f[5] * wt;
                                    if label[4]>0
                                        index = 64*(m-1) + 8*(3-1) + 4;
                                        A_v[index] = A_v[index] - val * f[3] * f[6] * wt;
                                    end
                                end
                                if label[4]>0
                                    m = label[4];
                                    index = 64*(m-1) + 8*(4-1) + 4;
                                    A_v[index] = A_v[index] - val * f[4] * f[6] * wt;
                                    if label[3]>0
                                        index = 64*(m-1) + 8*(4-1) + 3;
                                        A_v[index] = A_v[index] - val * f[4] * f[5] * wt;
                                    end
                                end
                            end
                        end
                    end
                end

            end
        end
    end

    mat_v = sparse(IA, JA, A_v);
    mat_lap = sparse(IA, JA, A_lap);
    mat_mass = sparse(IA, JA, A_mass);

    return mat_lap, mat_v, mat_mass, rank

end
