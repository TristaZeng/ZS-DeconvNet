function [img_clean,img_noisy,img_sim] = create_simu_beads(system_params, smpl_params, img_params, otf_params)

% % necessary field
% system_params = {};
% system_params.NA_ex = 1.17;                                   % NA of the excitation beam
% system_params.lambda_ex = 0.488;                              % wavelength of the excitation beam
% system_params.angle_k0 = [1.48, 2.5272, -2.706114];           % directions of the excitation patterns   
% system_params.n_imm = 1.406;                                  % refractive index of immersion oil  
% 
% smpl_params = {};
% smpl_params.signal_intensity = 1000;                          % intensity of the beads beam
% smpl_params.noise_sigma = 3.5;                                % std of gaussian noise
% smpl_params.background = 100;                                 % mean value of gaussian noise 
% smpl_params.n_beads_max = 100;                                % max number of beads in a volume
% smpl_paramsbeads_radius = 50 * 1e-3;                          % radius of the beads
% 
% img_params = {};
% img_params.img_size_xy = 256;                                 % pixel number in xy dimension
% img_params.img_size_z = 21;                                   % pixel number in z dimension
% img_params.pxl_size_xy_raw = 0.061;                           % pixel size (raw images) in xy dimension
% img_params.pxl_size_z = 0.160;                                % pixel size in z dimension                               
% 
% otf_params = {};
% otf_params.file_type = 1;                                     % 1: otf in '.mrc' file, 2: psf in '.tif' file                 
% otf_params.file_path = '/home/bbnc/Documents/harddrive/MatlabCode/SIM-R2R/OTFs/OTF-3D/3D-488-OTF-1.17NA-ang2-0.160step-20-25-3.mrc';


addpath(genpath('XxMatlabUtils'));

% system params
NA_ex = system_params.NA_ex; 
lambda_ex = system_params.lambda_ex;
angle_k0 = system_params.angle_k0;
n_imm = system_params.n_imm;

% sample params
signal_intensity = smpl_params.signal_intensity;
noise_sigma = smpl_params.noise_sigma; % std of gaussian noise
background = smpl_params.background; % mean value of gaussian noise 
n_beads_max = smpl_params.n_beads_max;
beads_radius = smpl_params.beads_radius;

% image params
img_size_xy = img_params.img_size_xy;
img_size_z = img_params.img_size_z;
pxl_size_xy = img_params.pxl_size_xy_raw/2; % pixel size of SR images
pxl_size_z = img_params.pxl_size_z;


% otf params
file_type = otf_params.file_type; % 1: otf in '.mrc' file; 2: psf in '.tif' file
file_path = otf_params.file_path;

Nxy_hr = img_size_xy * 2;
dkxy = 1 / (Nxy_hr * pxl_size_xy);
dkz = 1 / (img_size_z * pxl_size_z);

if file_type == 1
    otf_lr = XxReadOTF3D(file_path, Nxy_hr, Nxy_hr, img_size_z, dkxy, dkxy, dkz);
    psf_lr = XxNorm(abs(fftshift(fftn(otf_lr))),0,100);
    psf_lr = psf_lr / sum(psf_lr(:));
    otf_lr = otf_lr / max(otf_lr(:));

elseif file_type == 2
    z_num_psf = numel(imfinfo(file_path));
    psf_all = [];
    for zz = 1: 1: z_num_psf
        psf_all(:,:,zz) = double(imread(file_path,zz));
    end
    psf_lr = psf_all(:,:,ceil(z_num_psf/2)-floor(img_size_z/2):ceil(z_num_psf/2)+floor(img_size_z/2));
    psf_lr = psf_lr / sum(psf_lr(:));
    otf_lr = abs(fftshift(fftn(psf_lr)));
    otf_lr = otf_lr / max(otf_lr(:));
end

% ideal sim otf
[sigma_xy, sigma_z] = XxGuassianFitting3D(psf_lr);
psf_hr = XxGuassianGenerator3D(Nxy_hr, Nxy_hr, img_size_z, sigma_xy/2, sigma_z/2);
otf_hr = XxNorm(abs(fftshift(fftn(psf_hr))));

% ideal beads otf
sigma_beads_xy = beads_radius / 2.355 / pxl_size_xy;
sigma_beads_z = beads_radius / 2.355 / pxl_size_z;
psf_beads = XxGuassianGenerator3D(Nxy_hr, Nxy_hr, img_size_z*2-1, sigma_beads_xy, sigma_beads_z);
otf_beads = XxNorm(abs(fftshift(fftn(psf_beads))));


% define SIM parameters
ndirs = length(angle_k0);
nphases = 5;
dtheaX = 2 * pi / nphases;
xx = pxl_size_xy * (-Nxy_hr/2:Nxy_hr/2-1);
yy = pxl_size_xy * (-Nxy_hr/2:Nxy_hr/2-1);
[X,Y] = meshgrid(xx,yy);
z = (-img_size_z/2:1:img_size_z/2-1)*pxl_size_z;

% generate SIM pattern
patterns_ex = zeros(Nxy_hr,Nxy_hr,nphases,img_size_z,ndirs);
for i = 1:img_size_z
    for d = 0:ndirs-1
        alpha = angle_k0(d+1);
        add_phase = 0;
        kxL = 2 * pi / lambda_ex * NA_ex * cos(alpha);
        kyL = 2 * pi / lambda_ex * NA_ex * sin(alpha);
        kzL = sqrt((2*pi/lambda_ex*n_imm)^2 - (2*pi/lambda_ex*NA_ex)^2);
        kxR = -2 * pi / lambda_ex*NA_ex*cos(alpha);
        kyR = -2 * pi / lambda_ex*NA_ex*sin(alpha);
        kzR = kzL;
        kxC = 0;
        kyC = 0;
        kzC = 2 * pi / lambda_ex * n_imm;
        for n = 0:nphases-1
            phOffsetL = n * dtheaX + add_phase/2;
            phOffsetR = -n * dtheaX - add_phase/2;
            interBeam = exp(1i*(kxL*X + kyL*Y + kzL*z(i) + phOffsetL)) + ...
                exp(1i*(kxR*X + kyR*Y + kzR*z(i) + phOffsetR)) + ...
                exp(1i*(kxC*X + kyC*Y + kzC*z(i)));
            pattern = abs(interBeam) .^ 2;
            patterns_ex(:,:,n+1,i,d+1) = pattern / max(pattern(:));
        end
    end
end
patterns_ex = XxNorm(patterns_ex);

% generate random beads map
n_beads = randi(round(n_beads_max/2)) + round(n_beads_max/2);
cordinate_xy = randi(Nxy_hr,n_beads, 2);
cordinate_z = randi(img_size_z*2-1,n_beads, 1);

beads_map_all = zeros(Nxy_hr,Nxy_hr,img_size_z*2-1);
for k = 1:n_beads
    beads_map_all(cordinate_xy(k,1),cordinate_xy(k,2),cordinate_z(k)) = 1;
end

beads_map_all = fftshift(fftn(beads_map_all)) .* otf_beads;
beads_map_all = abs(ifftn(ifftshift(beads_map_all))) / 0.2918;
beads_map = beads_map_all(:, :, ceil(img_size_z/2):ceil(img_size_z/2)+img_size_z-1);

% ideal sim images
img_sim = fftshift(fftn(beads_map)) .* otf_hr;
img_sim = abs(ifftn(ifftshift(img_sim)));
img_sim = 2^15 * XxNorm(imresize3(img_sim, [Nxy_hr, Nxy_hr, img_size_z]));

% generate raw SIM images
img_clean = zeros(img_size_xy, img_size_xy, img_size_z*ndirs*nphases); % noise-free raw SIM images
img_noisy = zeros(img_size_xy, img_size_xy, img_size_z*ndirs*nphases); % noisy raw SIM images

for z = 1:img_size_z
    cur_beads_map = beads_map_all(:,:,z:z+img_size_z-1);
    for d = 1:ndirs
        for p = 1:nphases
            id = (z-1)*ndirs*nphases + (d-1)*nphases + p;
            pattern = squeeze(patterns_ex(:,:,p,:,d));
            cur_volume = cur_beads_map .* pattern;

            cur_volume = fftshift(fftn(cur_volume)) .* otf_lr;
            cur_volume = abs(ifftn(ifftshift(cur_volume)));
            cur_volume = imresize3(cur_volume, [img_size_xy, img_size_xy, img_size_z]);
            
            img_beads = signal_intensity * cur_volume(:,:,ceil(img_size_z/2)) / 0.0129;

            img_clean(:, :, id) = 2^10 * img_beads / signal_intensity;
            img_noisy(:, :, id) = round(poissrnd(img_beads) +...
                normrnd(background, noise_sigma, size(img_beads)));
        end
    end
end


end