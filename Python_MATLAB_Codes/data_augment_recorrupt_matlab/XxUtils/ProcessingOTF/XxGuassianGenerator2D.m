function kernel = XxGuassianGenerator2D(Nx, Ny, sigma_xy)
kernel = fspecial('gaussian', [Nx, Ny], sigma_xy);
kernel = kernel / sum(kernel(:));
end