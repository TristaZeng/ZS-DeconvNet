clc, clear;
addpath('OTF');
addpath('XxUtils');

%% calculate OTF0
OTFPath = 'Detection_OTF_488.mrc';
OTFPath_Simul = '/media/zkyd/New Volume1/IsotropicRecon/B23/OTFs/OTF0.tif';
SaveOTF0 = '/media/zkyd/New Volume1/IsotropicRecon/B23/OTFs_Exp/OTF0.tif';
SaveOTF1 = '/media/zkyd/New Volume1/IsotropicRecon/B23/OTFs_Exp/OTF1.tif';

nx = 512;
ny = 512;
nz = 151;
dxy = 0.0926;
dz = 0.2;

dkx = 1 / (nx * dxy);
dky = 1 / (nx * dxy);
dkz = 1 / (nz * dz);

OTF = XxReadOTF3D(OTFPath, nx, ny, nz, dkx, dky, dkz);
OTF = OTF(:,:,ceil(nz/2));
OTF_simul = imread(OTFPath_Simul);

figure(1);
subplot(1,2,1), imshow(OTF,[]);
subplot(1,2,2), imshow(OTF_simul,[]);

imwrite(uint16(65535 * OTF), SaveOTF0);

%% Calculate OTF1
lambda = 0.488;
angle = 1.58;
NA_exc = 0.4;
dist = NA_exc / lambda;
deltax = fix(dist * cos(angle) / dkx);
deltay = fix(dist * sin(angle) / dky);

OTF1 = circshift(OTF, [deltay, deltax]) + circshift(OTF, [-deltay, -deltax]);
OTF1 = XxNorm(OTF1,0,100);

figure(2);
imshow(OTF1, []);

imwrite(uint16(65535 * OTF1), SaveOTF1);




