function kernel = XxGuassianGenerator3D(Nx, Ny, Nz, sigma_xy, sigma_z)

kernel_xy = fspecial('gaussian', [Nx, Ny], sigma_xy);
kernel_z = fspecial('gaussian', Nz, sigma_z);
kernel_z = kernel_z / max(kernel_z);
kernel = repmat(kernel_xy, [1, 1, Nz]);
for i = 1:Nz
    kernel(:, :, i) = kernel(:, :, i) * kernel_z(i);
end
kernel = kernel / sum(kernel(:));
end