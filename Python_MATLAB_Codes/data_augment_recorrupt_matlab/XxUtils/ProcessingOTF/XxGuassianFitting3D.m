function [sigma_xy, sigma_z] = XxGuassianFitting3D(psf)

addpath(genpath('../'));

[Ny, ~, Nz] = size(psf);
y = double(1:Ny);
z = double(1:Nz);
psf_z = squeeze(max(max(psf,[] , 2), [], 1));
[~, I] = max(psf_z);
psf_xy = psf(:, :, round(I));

% calculate psf_y
[~, I] = max(sum(psf_xy,2));
psf_y = XxNorm(psf_xy(I,:),0,100);
% calulate psf_z
[my, mx] = find(psf_xy == max(psf_xy(:)));
psf_z = XxNorm(squeeze(psf(my, mx, :)), 0, 100);
% calculate sigma_rsf
fun_gs = fittype('A*exp(-(x-mu)^2/(2*sigma^2))');
[cf_y, ~] = fit(y(:),psf_y(:),fun_gs,'Start',[1, Ny/2, 3]);
[cf_z, ~] = fit(z(:),psf_z(:),fun_gs,'Start',[1, Nz/2, 3]);
sigma_xy = cf_y.sigma;
sigma_z = cf_z.sigma;

end