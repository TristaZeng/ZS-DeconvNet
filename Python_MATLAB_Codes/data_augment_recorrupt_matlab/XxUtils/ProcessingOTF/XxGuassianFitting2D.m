function sigma = XxGuassianFitting2D(psf)

addpath(genpath('../'));

[~, Nx] = size(psf);
x = double(1:Nx);
[~, I] = max(sum(psf,2));
psf_y = XxNorm(psf(I,:),0,100);
fun_gs = fittype('A*exp(-(x-mu)^2/(2*sigma^2))');
[cf_y, ~] = fit(x(:),psf_y(:),fun_gs,'Start',[1,  Nx/2, 3]);
sigma = cf_y.sigma;

end