clc, clear;
addpath(genpath('D:\QcData\XxMatlabUtils'));

%% calculate OTF0
% OTFPath = 'Detection_OTF_488_NAo0p6_NAi0p505.mrc';
OTFPath = 'Detection_OTF_560_NAo0p6_NAi0p505.mrc';
% OTFPath = 'Detection_OTF_642_NAo0p6_NAi0p505.mrc';
SaveOTF2d = [OTFPath(1:end-4) '_2D.mrc'];

% parameter of 3D OTF files
nx = 512;
ny = 512;
nz = 151;
dxy = 0.0926;
dz = 0.2;
NA = 1;

dkx = 1 / (nx * dxy);
dky = 1 / (nx * dxy);
dkz = 1 / (nz * dz);

[headerotf, rawOTF] = XxReadMRC(OTFPath);
nxotf = single(headerotf(1));
nyotf = single(headerotf(2));
nzotf = single(headerotf(3));
dkzotf = single(typecast(headerotf(11),'single'));
dkrotf = single(typecast(headerotf(12),'single'));
dkr = min(dkx,dky);
rawOTF = complex(rawOTF(1:2:end),rawOTF(2:2:end));
rawOTF = reshape(rawOTF,[nxotf,nyotf,nzotf]);
diagdist = ceil(sqrt((nx/2)^2+(ny/2)^2)+1);

temp=typecast(headerotf(50),'int16');
nwaves=single(temp(2)); 
lamda=single(temp(1))/1000;

rawOTF = rawOTF(:,:,1);
rawOTF = fftshift(rawOTF,1);
OTF_abs_y = sum(abs(rawOTF),2);
ind = find(OTF_abs_y==max(OTF_abs_y));
OTF2d = abs(rawOTF(ind,:,:));
for i = 15:-1:1
    OTF2d(i) = 2*OTF2d(i+1)-OTF2d(i+2);
end
% OTF2d = OTF2d(2,:);
OTF2d = XxNorm(smooth(OTF2d,10));
krcutoff = 2*NA/lamda/dkr + 0.5;
OTF2d(krcutoff:end) = 0;

figure();
subplot(2,1,1), imshow(log(abs(rawOTF)),[]);
subplot(2,1,2), plot(OTF2d);

%% save mrc files
header_out = headerotf;
header_out(1) = int32(nyotf);
header_out(2) = int32(1);
header_out(3) = int32(1);
header_out(11) = int32(typecast(dkrotf, 'int32'));
header_out(12) = int32(typecast(dkrotf, 'int32'));
header_out(13) = int32(typecast(dkzotf, 'int32'));
OTF2d = complex(single(OTF2d),single(zeros(size(OTF2d))));
handle_out = fopen(SaveOTF2d,'w+');
handle_out = XxWriteMRC_SmallEndian(handle_out, OTF2d, header_out);
fclose(handle_out);






