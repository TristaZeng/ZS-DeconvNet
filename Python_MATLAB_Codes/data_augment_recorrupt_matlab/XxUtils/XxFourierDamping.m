function img_out = XxFourierDamping(img_raw, ksize, kstd, Center)

if nargin < 4
    Center = [372,528,670,653,496,354; 613,685,585,412,341,441];
end

[Ny, Nx] = size(img_raw);
img_fft = fftshift(fft2(img_raw));

yc = Center(1,:);
xc = Center(2,:);

G = fspecial('gaussian',[ksize ksize],kstd);
G = XxNorm(G,0,100);

Mask = zeros(Nx,Ny);
for i = 1:length(xc)
    Mask(yc(i)-(ksize-1)/2:yc(i)+(ksize-1)/2,xc(i)-(ksize-1)/2:xc(i)+(ksize-1)/2) = G;
end

img_fft_fd = img_fft .* (1-Mask);
img_out = abs(ifft2(ifftshift(img_fft_fd)));
