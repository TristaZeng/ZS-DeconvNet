function curOTF = XxReadOTF3D(OTFPath, nx, ny, nz, dkx, dky, dkz)
addpath(genpath('../'));

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

% dzotf = 1 / (dkzotf * nxotf);
rawOTF = abs(rawOTF(:,:,1));
rawOTF = fftshift(rawOTF,1);

% figure();
% subplot(2,1,1), imshow(log(abs(rawOTF)),[]);

z = double((0:dkzotf:(nxotf-1)*dkzotf+1e-6)' - dkzotf*floor(nxotf/2));
zi = double((0:dkz:(nz-1)*dkz)' - dkz*floor(nz/2));
x = double((0:dkrotf:(nyotf-1)*dkrotf+1e-6)');
xi = double((0:dkr:nyotf*dkrotf)');
[X,Z] = meshgrid(x,z);
[Xi,Zi] = meshgrid(xi,zi);

rawOTF = interp2(X,Z,rawOTF,Xi,Zi,'spline');
rawOTF(isnan(rawOTF)) = 0;
rawOTF(:,size(rawOTF,2)+1:diagdist) = 0;
[~,otflen] = size(rawOTF);
OTF = rawOTF;

% subplot(2,1,2), imshow(log(abs(OTF)),[]);

x = (-nx/2:1:nx/2-1)*dkx;
y = (-ny/2:1:ny/2-1)*dky;
[X,Y] = meshgrid(x,y);
rdist = sqrt(X.^2+Y.^2);

curOTF = zeros(ny,nx,nz);
otflen = otflen - 1;
for z = 1:nz
    OTFz = OTF(z,:);
    curOTF(:,:,z) = interp1(0:dkr:otflen*dkr, OTFz, rdist, 'spline');
end

curOTF = abs(curOTF);
curOTF = curOTF / max(curOTF(:));
