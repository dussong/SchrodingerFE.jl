# Function replicated from Matlab in order to compare

function mat_2d_coulomb(L, N)
 # generate the stiff matrix for 1/|x-y|

   h = 2.0 * L / N;
   mat_coulomb = zeros(N-1, N-1, N-1, N-1);

   NPX = 4;
   NPY = 3;
   gauss_x = [-0.8611363116 -0.3399810435 0.3399810435 0.8611363116];
   weight_x = [0.3478548451 0.6521451548 0.6521451548 0.3478548451];
   gauss_y = [-0.7745966692 0.0 0.7745966692];
   weight_y = [0.5555555555 0.8888888889 0.5555555555];
   weight = weight_y'*weight_x;
   weight = reshape(weight,NPX*NPY,1);
   wt = weight * h^2/4.0;

   # points with respect to [-1,1]
   x = ones(NPY,1)*gauss_x;
   x = reshape(x,NPX*NPY,1);
   y = gauss_y'*ones(1,NPX);
   y = reshape(y,NPX*NPY,1);

   # value and gradient of basis functions at (x,y)
   f  = hcat((x.-1.0).*(y.-1.0)/4.0,
             .-(x.+1.0).*(y.-1.0)/4.0,
             .-(x.-1.0).*(y.+1.0)/4.0,
             (x.+1.0).*(y.+1.0)/4.0);

   for j = 1 : N
       for i = 1 : N
            # label
           xlabel = [i-1, i, i-1, i];
           ylabel = [j-1, j-1, j, j];
            # coordinate
           xx = .-L .+ h * (i.-0.5.+x/2);
           yy = .-L .+ h * (j.-0.5.+y/2);
           r = abs.(xx - yy);
            # calculate stiff elements
           for k = 1:4
               for l = 1:4
                   ix = xlabel[k];
                   iy = ylabel[k];
                   jx = xlabel[l];
                   jy = ylabel[l];
                   if ix>0 && ix<N && iy>0 && iy<N && jx>0 && jx<N && jy>0 && jy<N
                       mat_coulomb[ix,iy,jx,jy] = mat_coulomb[ix,iy,jx,jy] +
                               sum( 1.0 ./r .* f[:,k] .* f[:,l] .* wt );
                   end
               end
           end
       end
   end
   return mat_coulomb
end
