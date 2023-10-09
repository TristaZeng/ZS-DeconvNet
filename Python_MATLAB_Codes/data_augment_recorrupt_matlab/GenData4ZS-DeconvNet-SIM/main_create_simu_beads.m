clear;
close all;
cd 'C:\Users\Admin\Git\repo\ZS-DeconvNet\Python_MATLAB_Codes\data_augment_recorrupt_matlab'; % change this path to your cloned path
addpath(genpath('./XxUtils'));
addpath(genpath('./GenData4ZS-DeconvNet-SIM'))

% necessary field
system_params = {};
system_params.NA_ex = 1.17;                                   % NA of the excitation beam
system_params.lambda_ex = 0.488;                              % wavelength of the excitation beam
system_params.angle_k0 = [1.48, 2.5272, -2.706114];           % directions of the excitation patterns   
system_params.n_imm = 1.406;                                  % refractive index of immersion oil  

smpl_params = {};
smpl_params.signal_intensity = 1000;                          % intensity of the beads beam
smpl_params.noise_sigma = 3.5;                                % std of gaussian noise
smpl_params.background = 100;                                 % mean value of gaussian noise 
smpl_params.n_beads_max = 100;                                % max number of beads in a volume
smpl_params.beads_radius = 50 * 1e-3;                          % radius of the beads

img_params = {};
img_params.img_size_xy = 256;                                 % pixel number in xy dimension
img_params.img_size_z = 21;                                   % pixel number in z dimension
img_params.pxl_size_xy_raw = 0.061;                           % pixel size (raw images) in xy dimension
img_params.pxl_size_z = 0.160;                                % pixel size in z dimension                               

otf_params = {};
otf_params.file_type = 1;                                     % 1: otf in '.mrc' file, 2: psf in '.tif' file                 
otf_params.file_path = '3D-488-OTF-1.17NA-ang2-0.160step-20-25-3.mrc';

% system params
NA_ex = system_params.NA_ex;
lambda_ex = system_params.lambda_ex;
angle_k0 = system_params.angle_k0;
n_imm = system_params.n_imm;


[img_clean,img_noisy,img_sim] = create_simu_beads(system_params, smpl_params, img_params, otf_params);
