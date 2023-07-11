# Implementation from Matlab translated to Julia
using SparseArrays


function mat_2d_fermion(L, N)
    #  _____________________________
    # |     0        | (N-1)(N-2)/2 |
    # |______________|______________|
    # | (N-1)(N-2)/2 |  (N-1)(N-1)  |
    # |______________|______________|

    h = 2.0 * L / N;
    n = 2 * N;
    dof = (N-1) * (2 * N - 3);
    mat_lap  = zeros(dof, dof);
    mat_mass = zeros(dof, dof);

    rank = zeros(Int,dof, 2);
    label_matrix = zeros(Int,n+1,n+1);
    is_p = zeros(Int,n+1,n+1);
    count = 0;
    for i in 2:n
        for j in (i+1):n
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
    gy = hcat((y.-1.0)/4.0 * 2.0/h, .-(y.-1.0)/4.0 * 2.0/h,
                .-(y.+1.0)/4.0 * 2.0/h, (y.+1.0)/4.0 * 2.0/h) ;
    gx = hcat((x.-1.0)/4.0 * 2.0/h, .-(x.+1.0)/4.0 * 2.0/h,
                .-(x.-1.0)/4.0 * 2.0/h, (x.+1.0)/4.0 * 2.0/h) ;


    for i = 1 : n
        for j = 1 : n
            #----------- label the 4 nodes in each elements ----------#
            label = [label_matrix[i,j], label_matrix[i,j+1],
                     label_matrix[i+1,j],label_matrix[i+1,j+1]];
            isp = [is_p[i,j], is_p[i,j+1], is_p[i+1,j], is_p[i+1,j+1]];
            ind = findall(x->(x!=0),label);

            #--------------- calculate stiff elements -----------------#
            for k = ind
                for l = ind
                    mx = label[k];
                    my = label[l];
                    ix = isp[k];
                    iy = isp[l];
                    mat_lap[mx,my] = mat_lap[mx,my] + ix * iy * sum( (gx[:,k] .* gx[:,l] + gy[:,k] .* gy[:,l]) .* wt );
                    mat_mass[mx,my] = mat_mass[mx,my] + ix * iy * sum( f[:,k] .* f[:,l] .* wt );
                end
            end
        end
    end
    return (sparse(mat_lap), sparse(mat_mass), rank)
end


function mat_3d_fermion(L, N)
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

    h   = 2.0*L/N;
    n   = 2*N;
    dof = Int((2*N-2) * (2*N-3) * (2*N-4) /6);
    s_size = 64;
    IA     = ones(Int, dof * s_size);
    JA     = ones(Int, dof * s_size);
    A_mass = zeros(dof * s_size);
    A_lap  = zeros(dof * s_size);
    label  = zeros(Int,8,1);
    rank   = zeros(Int,dof, 3);

    NP = 3;
    gaussP  = [-0.7745966692 0.0 0.7745966692];
    weightP = [0.5555555555 0.8888888889 0.5555555555];
    NQ = 4;
    gaussQ  = [-0.8611363116 -0.3399810435 0.3399810435 0.8611363116];
    weightQ = [0.3478548451 0.6521451548 0.6521451548 0.3478548451];

    labelP2P = zeros(Int,(n-1)^3,1);
    kk = 1;
    for nk in 1:n-1
        for nj in 1:n-1
            for ni in 1:n-1
                m = (n-1)^2 * (nk-1) + (n-1) * (nj-1) + (ni-1) + 1;
                if nk < nj && nj < ni && nk != N && nj != N && ni != N
                    labelP2P[m] = kk;
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
                    label[1] = -1;
                    label[3] = -1;
                    label[5] = -1;
                    label[7] = -1;
                end
                if ni==n
                    label[2] = -1;
                    label[4] = -1;
                    label[6] = -1;
                    label[8] = -1;
                end
                if nj==1
                    label[1] = -1;
                    label[2] = -1;
                    label[5] = -1;
                    label[6] = -1;
                end
                if nj==n
                    label[3] = -1;
                    label[4] = -1;
                    label[7] = -1;
                    label[8] = -1;
                end
                if nk==1
                    label[1] = -1;
                    label[2] = -1;
                    label[3] = -1;
                    label[4] = -1;
                end
                if nk==n
                    label[5] = -1;
                    label[6] = -1;
                    label[7] = -1;
                    label[8] = -1;
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
                            gx = 1.0/8 * 2.0/h * hcat( -(1.0-y)*(1.0-z), (1.0-y)*(1.0-z), -(y+1.0)*(1.0-z), (y+1.0)*(1.0-z),
                               -(1.0-y)*(1.0+z), (1.0-y)*(1.0+z),
                               -(y+1.0)*(1.0+z), (y+1.0)*(1.0+z) ) ;
                            gy = 1.0/8 * hcat( -(1.0-x)*(1.0-z), -(x+1.0)*(1.0-z), (1.0-x)*(1.0-z), (x+1.0)*(1.0-z),
                           -(1.0-x)*(1.0+z), -(x+1.0)*(1.0+z),
                           (1.0-x)*(1.0+z), (x+1.0)*(1.0+z) ) * 2.0/h;
                            gz = 1.0/8 * hcat( -(1.0-x)*(1.0-y), -(x+1.0)*(1.0-y),-(1.0-x)*(y+1.0), -(x+1.0)*(y+1.0),
                           (1.0-x)*(1.0-y), (x+1.0)*(1.0-y),
                           (1.0-x)*(y+1.0), (x+1.0)*(y+1.0) ) * 2.0/h;

                            for k = 1:8
                                for l = 1:8
                                    mk = label[k];
                                    ml = label[l];
                                    index = 64*(mk-1) + 8*(k-1) + l;
                                    if mk>0 && ml>0
                                        IA[index] = mk;
                                        JA[index] = ml;
                                      A_mass[index] = A_mass[index] + f[k] * f[l] * wt;
                                      A_lap[index]  = A_lap[index] + ( gx[k] * gx[l] + gy[k] * gy[l] + gz[k] * gz[l] ) * wt;
                                    end
                                end
                            end
                            # deal with where pertubation basis at same element. note that it is not possible that ni = nj = nk
                            if ni == nj
                                if label[2]>0
                                    m = label[2];
                                    index = 64*(m-1) + 8*(2-1) + 2;
                                    A_mass[index] = A_mass[index] - f[2] * f[3] * wt;
                                    A_lap[index] = A_lap[index] - ( gx[2] * gx[3] + gy[2] * gy[3] + gz[2] * gz[3] ) * wt;
                                    if label[6]>0
                                        index = 64*(m-1) + 8*(2-1) + 6;
                                        A_mass[index] = A_mass[index] - f[2] * f[7] * wt;
                                        A_lap[index] = A_lap[index] - ( gx[2] * gx[7] + gy[2] * gy[7] + gz[2] * gz[7] ) * wt;
                                    end
                                end
                                if label[6]>0
                                    m = label[6];
                                    index = 64*(m-1) + 8*(6-1) + 6;
                                    A_mass[index] = A_mass[index] - f[6] * f[7] * wt;
                                    A_lap[index] = A_lap[index] - ( gx[6] * gx[7] + gy[6] * gy[7] + gz[6] * gz[7] ) * wt;
                                    if label[2]>0
                                        index = 64*(m-1) + 8*(6-1) + 2;
                                        A_mass[index] = A_mass[index] - f[6] * f[3] * wt;
                                        A_lap[index]  = A_lap[index] - ( gx[6] * gx[3] + gy[6] * gy[3] + gz[6] * gz[3] ) * wt;
                                    end
                                end
                            end
                            if nj == nk
                                 if label[3]>0
                                    m = label[3];
                                    index = 64*(m-1) + 8*(3-1) + 3;
                                    A_mass[index] = A_mass[index] - f[3] * f[5] * wt;
                                    A_lap[index]  = A_lap[index] - ( gx[3] * gx[5] + gy[3] * gy[5] + gz[3] * gz[5] ) * wt;
                                    if label[4]>0
                                        index = 64*(m-1) + 8*(3-1) + 4;
                                        A_mass[index] = A_mass[index] - f[3] * f[6] * wt;
                                        A_lap[index]  = A_lap[index] - ( gx[3] * gx[6] + gy[3] * gy[6] + gz[3] * gz[6] ) * wt;
                                    end
                                end
                                if label[4]>0
                                    m = label[4];
                                    index = 64*(m-1) + 8*(4-1) + 4;
                                    A_mass[index] = A_mass[index] - f[4] * f[6] * wt;
                                    A_lap[index]  = A_lap[index] - ( gx[4] * gx[6] + gy[4] * gy[6] + gz[4] * gz[6] ) * wt;
                                    if label[3]>0
                                        index = 64*(m-1) + 8*(4-1) + 3;
                                        A_mass[index] = A_mass[index] - f[4] * f[5] * wt;
                                        A_lap[index]  = A_lap[index] - ( gx[4] * gx[5] + gy[4] * gy[5] + gz[4] * gz[5] ) * wt;
                                    end
                                end
                            end
                        end
                    end
                end

            end
        end
    end

    mat_lap  = sparse(IA, JA, A_lap) * 6;
    mat_mass = sparse(IA, JA, A_mass) * 6;

    return (mat_lap, mat_mass, rank)
end



function mat_4d_fermion(L, N)

    h = 2.0*L/N;
    n = 2*N;
    dof = Int((2*N-2) * (2*N-3) * (2*N-4) * (2*N-5) /24);
    s_size = 256;
    IA     = ones(Int, dof * s_size);
    JA     = ones(Int, dof * s_size);
    A_mass = zeros(dof * s_size);
    A_lap  = zeros(dof * s_size);
    rank   = zeros(Int,dof, 4);

    NP = 2;
    NQ = 3;
    gauss   = [-0.577350269189625 0.577350269189625];
    weight  = [1.0 1.0];
    gaussQ  = [-0.7745966692 0.0 0.7745966692];
    weightQ = [0.5555555555 0.8888888889 0.5555555555];

    labelP2P = zeros(Int,(n-1)^4, 1);
    kk = 1;
    for n1 = 1:n-1
        for n2 = 1:n-1
            for n3 = 1:n-1
                for n4 = 1:n-1
                    m = (n-1)^3 * (n1-1) + (n-1)^2 * (n2-1) + (n-1) * (n3-1) + n4;
                    if n1 < n2 && n2 < n3 && n3 < n4 && n1 != N && n2 != N && n3 != N && n4 != N
                        labelP2P[m] = kk;
                        # evaluate the rank
                        rank[kk,:] = [n1, n2, n3, n4];
                        kk = kk + 1;
                    else
                        labelP2P[m] = -1;
                    end
                end
            end
        end
    end

    for n1 = 1:n
        for n2 = n1:n
            for n3 = n2:n
                for n4 = n3:n
                    # label the nodes in each element
                    (label, is_out, is_p) = label4d_fermion(N, n1, n2, n3, n4);
                    for k = 1:16
                        if label[k] > 0
                            label[k] = labelP2P[label[k]];
                        end
                    end

                    for p1 = 1:NP
                        for p2 = 1:NP
                            for p3 = 1:NQ
                                for p4 = 1: NQ
                                    # weight and gauss integral points in a reference element
                                    wt = weight[p1] * weight[p2] * weightQ[p3] * weightQ[p4] * (h/2.0)^4;
                                    x1 = gauss[p1];
                                    x2 = gauss[p2];
                                    x3 = gaussQ[p3];
                                    x4 = gaussQ[p4];
                                    # value and gradient of basis functions at (x,y)
                                    (f, g1, g2, g3, g4) = shape4d(x1, x2, x3, x4, h);

                                    for k = 1:16
                                        for l = 1:16
                                            mk = label[k];
                                            ml = label[l];
                                            index = 256*(mk-1) + 16*(k-1) + l;
                                            if mk>0 && ml>0 && is_out[k]>0
                                                IA[index] = mk;
                                                JA[index] = ml;
                    A_mass[index] = A_mass[index] + is_p[k] * is_p[l] * f[k] * f[l] * wt;
                    A_lap[index]  = A_lap[index] + is_p[k] * is_p[l] *  ( g1[k] * g1[l] + g2[k] * g2[l] + g3[k] * g3[l] + g4[k] * g4[l] ) * wt;
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end

                end
            end
        end
    end

    mat_mass = sparse(IA,JA,A_mass);
    mat_lap  = sparse(IA,JA,A_lap);

    return (mat_lap, mat_mass, rank)

end


function shape4d(x1, x2, x3, x4, h)
# give the shape function at gauss point (x1,x2,x3,x4)

# first the basis function in each directions
    basis = [ (.-(x1.-1.0)) (x1.+1.0);
              (.-(x2.-1.0)) (x2.+1.0);
              (.-(x3.-1.0)) (x3.+1.0);
              (.-(x4.-1.0)) (x4.+1.0) ] / 2.0;
    gradient = hcat( -1, 1 ) / 2.0;

    f = zeros(16);
    g1 = zeros(16);
    g2 = zeros(16);
    g3 = zeros(16);
    g4 = zeros(16);

    for k1 = 1:2
        for k2 = 1:2
            for k3 = 1:2
                for k4 = 1:2
                    k = k1 + 2 * (k2-1) + 4 * (k3-1) + 8 * (k4-1);
                    f[k] = basis[1,k1] * basis[2,k2] * basis[3,k3] * basis[4,k4];
                    g1[k] = gradient[k1] * basis[2,k2] * basis[3,k3] * basis[4,k4] * 2.0/h;
                    g2[k] = basis[1,k1] * gradient[k2] * basis[3,k3] * basis[4,k4] * 2.0/h;
                    g3[k] = basis[1,k1] * basis[2,k2] * gradient[k3] * basis[4,k4] * 2.0/h;
                    g4[k] = basis[1,k1] * basis[2,k2] * basis[3,k3] * gradient[k4] * 2.0/h;
                end
            end
        end
    end

    return (f, g1, g2, g3, g4)
end

function label4d_fermion(N, n1, n2, n3, n4)
    # give the label of 16 nodes(4d element) in the element (n1,n2,n3,n4)

    n = 2*N;
    label = zeros(Int,16);
    is_out = ones(Int,16);
    is_p = ones(Int,16);

    labb = zeros(Int,16,4);
    #----------- label the 16 nodes in each elements ----------#
    labb[1,:] = [n1, n2, n3, n4];
    labb[2,:] = [n1, n2, n3, n4+1];
    labb[3,:] = [n1, n2, n3+1, n4];
    labb[4,:] = [n1, n2, n3+1, n4+1];
    labb[5,:] = [n1, n2+1, n3, n4];
    labb[6,:] = [n1, n2+1, n3, n4+1];
    labb[7,:] = [n1, n2+1, n3+1, n4];
    labb[8,:] = [n1, n2+1, n3+1, n4+1];

    labb[9,:] = [n1+1, n2, n3, n4];
    labb[10,:] = [n1+1, n2, n3, n4+1];
    labb[11,:] = [n1+1, n2, n3+1, n4];
    labb[12,:] = [n1+1, n2, n3+1, n4+1];
    labb[13,:] = [n1+1, n2+1, n3, n4];
    labb[14,:] = [n1+1, n2+1, n3, n4+1];
    labb[15,:] = [n1+1, n2+1, n3+1, n4];
    labb[16,:] = [n1+1, n2+1, n3+1, n4+1];

    for k = 1:16
        # is_out
        if labb[k,1]>labb[k,2] || labb[k,2]>labb[k,3] || labb[k,3]>labb[k,4]
            is_out[k] = -1;
        end
        # is_pertubation
        m = 0;
        if labb[k,4]<labb[k,3]      # 4-3
            temp = labb[k,3];
            labb[k,3] = labb[k,4];
            labb[k,4] = temp;
            m = m+1;
        end
        if labb[k,3]<labb[k,2]      #3-2
            temp = labb[k,2];
            labb[k,2] = labb[k,3];
            labb[k,3] = temp;
            m = m+1;
        end
        if labb[k,2]<labb[k,1]      #2-1
            temp = labb[k,1];
            labb[k,1] = labb[k,2];
            labb[k,2] = temp;
            m = m+1;
        end
        if labb[k,4]<labb[k,3]      #4-3
            temp = labb[k,3];
            labb[k,3] = labb[k,4];
            labb[k,4] = temp;
            m = m+1;
        end
        if labb[k,3]<labb[k,2]      #3-2
            temp = labb[k,2];
            labb[k,2] = labb[k,3];
            labb[k,3] = temp;
            m = m+1;
        end
        if labb[k,4]<labb[k,3]      #4-3
            temp = labb[k,3];
            labb[k,3] = labb[k,4];
            labb[k,4] = temp;
            m = m+1;
        end

        is_p[k] = (-1)^m;
        # label
        label[k] = (n-1)^3 * (labb[k,1]-2) + (n-1)^2 * (labb[k,2]-2) + (n-1) * (labb[k,3]-2) + (labb[k,4]-2) + 1;
    end

    #------- label the boundary nodes as -1 ------------#
    if n1 == 1 || n1 == N+1
        for k = 1:8
            label[k] = -1;
        end
    end
    if n1 == N || n1 == 2*N
        for k = 9:16
            label[k] = -1;
        end
    end

    if n2 == 1 || n2 == N+1
        for k = 1:4
            label[k] = -1;
            label[k+8] = -1;
        end
    end
    if n2 == N || n2 == 2*N
        for k = 5:8
            label[k] = -1;
            label[k+8] = -1;
        end
    end

    if n3 == 1 || n3 == N+1
        for k = 1:4
            label[4*k-3] = -1;
            label[4*k-2] = -1;
        end
    end
    if n3 == N || n3 == 2*N
        for k = 1:4
            label[4*k-1] = -1;
            label[4*k] = -1;
        end
    end

    if n4 == 1 || n4 == N+1
        for k = 1:8
            label[2*k-1] = -1;
        end
    end
    if n4 == N || n4 == 2*N
        for k = 1:8
            label[2*k] = -1;
        end
    end
    return (label, is_out, is_p)
end




# function mat_2d_ne2(epsilon, L, N)
#     # generate the stiff matrix for \epsilon\Delta+1/|x-y|
#     #with Gaussian quadrature
#     # taken from the matlab code
#     h = 2.0*L/(N-1);
#     dof = (N-1)*(N-1);
#     mat_lap = zeros(dof, dof);
#     mat_mass = zeros(dof, dof);
#
#     #Label matrix # save labels and coordinates for 4 points in each element, i,e,(n,x,y)
#     label_matrix = zeros(Int,N+1,N+1);
#     label_matrix[2:end-1,2:end-1] = reshape(1:dof,N-1,N-1)';
#
#     NPX = 4;
#     NPY = 3;
#     gauss_x = [-0.8611363116 -0.3399810435 0.3399810435 0.8611363116];
#     weight_x = [0.3478548451 0.6521451548 0.6521451548 0.3478548451];
#     gauss_y = [-0.7745966692 0.0 0.7745966692];
#     weight_y = [0.5555555555 0.8888888889 0.5555555555];
#
#     #gauss = gauss_y'*gauss_x;
#     #gauss = reshape(gauss,NPX*NPY,1);
#     weight = weight_y'*weight_x;
#     weight = reshape(weight,NPX*NPY,1);
#     wt = weight * h^2/4.0;
#
#     #points with respect to [-1,1]
#     x = ones(NPY,1)*gauss_x;
#     x = reshape(x,NPX*NPY,1);
#     y = gauss_y'*ones(1,NPX);
#     y = reshape(y,NPX*NPY,1);
#
#     # value and gradient of basis functions at (x,y)
#     f =hcat((x.-1.0).*(y.-1.0)/4.0, .-(x.+1.0).*(y.-1.0)/4.0, .-(x.-1.0).*(y.+1.0)/4.0, (x.+1.0).*(y.+1.0)/4.0);
#     gy = hcat((y.-1.0)/4.0 * 2.0/h, .-(y.-1.0)/4.0 * 2.0/h, .-(y.+1.0)/4.0 * 2.0/h, (y.+1.0)/4.0 * 2.0/h);
#     gx = hcat((x.-1.0)/4.0 * 2.0/h, .-(x.+1.0)/4.0 * 2.0/h, .-(x.-1.0)/4.0 * 2.0/h, (x.+1.0)/4.0 * 2.0/h) ;
#
#     for i=1:N
#         for j=1:N
#             label = [label_matrix[i,j],label_matrix[i,j+1],label_matrix[i+1,j],label_matrix[i+1,j+1]];
#             ind = findall(x->(x !=0), label);
#
#             # coordinates
#             xx = -L .+ h * (j.-0.5.+x/2);
#             yy = -L .+ h * (i.-0.5.+y/2);
#             r = abs.(xx - yy);
#
#             for k = ind
#                 for l = ind
#                     mx = label[k];
#                     my = label[l];
#                     mat_lap[mx,my] = mat_lap[mx,my] + sum( (epsilon * (gx[:,k].*gx[:,l] + gy[:,k] .* gy[:,l])+ 1.0./r .*f[:,k].*f[:,l]) .* wt);
#                     mat_mass[mx,my] = mat_mass[mx,my] + sum(f[:,k] .* f[:,l] .* wt);
#                 end
#             end
#         end
#     end
#
#     # mat_lap = sparse(mat_lap);
#     # mat_mass = sparse(mat_mass);
#
#     return mat_lap, mat_mass
# end
