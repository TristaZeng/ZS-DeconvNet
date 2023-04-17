clc, clear;
addpath('/home/li-lab/Documents/rDL-LLS-SIM-Matlab/GenerateDataset/XxUtils');

%% save otf as mrc file
% read psf
psfPath = 'Simulated_PSF_642.tif';
% psfPath = 'Illum642_Cyc1_Ch2_St1.mrc';
% psfPath = 'TIRF560_Cyc1_Ch2_St1.mrc';
% psfPath = 'Sheetscan488_Cyc1_Ch1_St1.mrc';
% psfPath = 'TestDetectionOTF/TIRF560_Cyc1_Ch2_St1.mrc';

lambda = 0.670;
NA = 1.1;

if strcmp(psfPath(end-2:end),'tif')
    psf = XxReadTiffSmallerThan4GB(psfPath,'uint16');
    psf = double(psf);
    % initialization
    dxy = 0.0926;
    dz = 0.100;
    [ny, nx, nz] = size(psf);
elseif strcmp(psfPath(end-2:end),'mrc')
    [header, psf] = XxReadMRC(psfPath);
    nx = single(header(1));
    ny = single(header(2));
    nz = single(header(3));
    psf = double(reshape(psf,[nx,ny,nz]));
    dxy = single(typecast(header(11),'single'));
    dz = single(typecast(header(13),'single'));
end

% sub background and shift
bg = 100;
psf = psf - bg;
psf(psf<0) = 0;
psf = psf / max(psf(:));
[my, ~] = find(psf == max(psf(:)));
psf_y = squeeze(psf(my,:,:));
[mx, mz] = find(psf_y == max(psf_y(:)));
z_prf = XxNorm(squeeze(psf(my,mx,:))) .^ 1;
psf_shift = circshift(psf,ceil(nz/2)-mz,3);

psf_y = squeeze(psf_shift(my,:,:));
[mx, mz_shift] = find(psf_y == max(psf_y(:)));
z_prf_shift = XxNorm(squeeze(psf_shift(my,mx,:))) .^ 1;

fprintf('Before shift, mz: %d, after shift, mz: %d\n', mz, mz_shift);

figure(1);
subplot(2,1,1), plot(z_prf); set(gca,'XLim',[1,nz]);
subplot(2,1,2), plot(z_prf_shift); set(gca,'XLim',[1,nz]);

% calculate otf
otf = fftshift(fftn(psf_shift));
otf_slice = squeeze(otf(floor(nx/2)+1:end,floor(ny/2),:));
otf_slice = otf_slice' / max(abs(otf_slice(:)));
otf_slice = fftshift(otf_slice,1);
[nx, ny] = size(otf_slice);
otf_abs = abs(otf_slice);
mask = log(otf_abs+1) > 1e-2;
otf_slice = otf_slice .* mask;

figure();
subplot(2,1,1), imshow(fftshift(abs(otf_slice),1),[]);
subplot(2,1,2), imshow(log(fftshift(abs(otf_slice),1)),[]);

% write otf
headerfile = 'otf_sys.mrc';
header = XxReadMRCHeader(headerfile);
header(1) = int32(nx);
header(2) = int32(ny);
header(3) = int32(1);
header(4) = int32(4);
header(11) = typecast(single(1/(dz*nz)),'int32');
header(12) = typecast(single(1/(dxy*ny*2)),'int32');
header(46) = typecast([int16(1), int16(1)], 'int32');
header(50) = typecast([int16(1), int16(lambda*1000)], 'int32');
% outputName = [psfPath(1:end-7) '_OTF.mrc'];
outputName = 'Simulated_OTF_642.mrc';
handle = fopen(outputName,'w+');
XxWriteMRC(handle,single(otf_slice),header);
fclose(handle);






