function curOTF = XxReadOTF2D(otfPath, nx, ny, dkx, dky)

addpath(genpath('../'));

[headerotf, rawOTF] = XxReadMRC(otfPath);
nxotf = single(headerotf(1));
dkrotf = single(typecast(headerotf(11),'single'));
dkr = min(dkx,dky);
diagdist = ceil(sqrt((nx/2)^2+(ny/2)^2)+1);
OTF = complex(rawOTF(1:2:end),rawOTF(2:2:end));
x = (0:dkrotf:(nxotf-1)*dkrotf)';
xi = (0:dkr:(nxotf-1)*dkrotf)';
OTF = interp1(x,OTF,xi,'spline');
sizeOTF = max(size(OTF));
OTF(sizeOTF+1:diagdist) = 0;

dx = (-nx/2:1:nx/2-1)*dkx;
dy = (-ny/2:1:ny/2-1)*dky;
[dX,dY] = meshgrid(dx,dy);
rdist = sqrt(dX.^2+dY.^2);
otflen = max(size(OTF))-1;
OTF = interp1(0:dkr:otflen*dkr, OTF, rdist, 'spline');
curOTF = XxNorm(OTF);
