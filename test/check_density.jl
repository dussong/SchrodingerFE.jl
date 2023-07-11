# Check density

function reshape_2d_fermion(u, N, rank)
# reshape the eigenvector into the 2-dim vector

    dof = (N-1) * (2 * N - 3);
    p2 = zeros(2*(N-1),2*(N-1));

    for k = 1:dof
        nn = rank[k,:];
        nj = nn[1];
        ni = nn[2];
        if ni > N
            ni = ni - 2;
        else
            ni = ni - 1;
        end
        if nj > N
            nj = nj - 2;
        else
            nj = nj - 1;
        end

        p2[ni,nj] = u[k];
        p2[nj,ni] = -u[k];
        #p2(nj,ni) = p2(nj,ni) + u[k]^2;
        #p2(ni,nj) = p2(ni,nj) + (-u[k])^2;
    end

    # return the mass matrix for 1d
    h = 2.0*L/N;
    mat_mas = zeros(N-1,N-1);
    #NP = 4; #number of quadrature points (Gauss)
    gauss = [-0.8611363116 -0.3399810435 0.3399810435 0.8611363116];
    weight = [0.3478548451 0.6521451548 0.6521451548 0.3478548451];
    wt = weight*h/2;
    fminus = (gauss.+1)./2;
    fplus = (1.0.-gauss)./2;

    mat_mas[1,1] = mat_mas[1,1] + sum(fminus.*fminus.*wt);
    for i = 2:N-1
        mat_mas[i,i] = mat_mas[i,i]+sum(fminus.*fminus.*wt);
        mat_mas[i-1,i-1] = mat_mas[i-1,i-1]+sum(fplus.*fplus.*wt);
        mat_mas[i,i-1] = mat_mas[i,i-1] + sum(fminus.*fplus.*wt);
        mat_mas[i-1,i] = mat_mas[i-1,i] + sum(fminus.*fplus.*wt);
    end
    mat_mas[N-1,N-1] = mat_mas[N-1,N-1] + sum(fplus.*fplus.*wt);


    rho = zeros(N-1)
    for k = 1:N-1
        rho[k] = p2[1:N-1,k]'*mat_mas*p2[1:N-1,k] + p2[N:2*(N-1),k]'*mat_mas*p2[N:2*(N-1),k] + p2[1:N-1,k]'*mat_mas*p2[1:N-1,k] + p2[N:2*(N-1),k+N-1]'*mat_mas*p2[N:2*(N-1),k+N-1];
    end
    rho = rho * 4.0;

    rho2 = p2[1:N-1,1:N-1].^2 + p2[1:N-1,N:2*(N-1)].^2 + p2[N:2*(N-1),1:N-1].^2 + p2[N:2*(N-1),N:2*(N-1)].^2
    rho2 = rho2./norm(rho2,2);

    return rho, rho2
end
