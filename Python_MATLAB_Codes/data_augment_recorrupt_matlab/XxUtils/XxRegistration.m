function img_regist = XxRegistration(img, ref, Regis_scale, maxshift, fix_y, fix_x, ShowRegisFlag)

if nargin < 7, ShowRegisFlag = 0; end
if nargin < 6, fix_x = 0; end
if nargin < 5, fix_y = 0; end
if nargin < 4, maxshift = uint16(2e15); end
if nargin < 3, Regis_scale = 3; end

maxshiftn = maxshift * Regis_scale;
% calculate cross correlation
img_r = imresize(img,Regis_scale,'bicubic');
ref_r = imresize(ref,Regis_scale,'bicubic');
F_img = fft2(img_r);
F_gt = fft2(ref_r);
crosscorr = abs(fftshift(ifft2(conj(F_img) .* F_gt)));
[my, mx] = find(crosscorr == max(crosscorr(:)));

center = round(size(ref,1) / 2);
offset_y = my(1) - (center*Regis_scale+1);
offset_x = mx(1) - (center*Regis_scale+1);
if fix_x == 1, offset_x = 0; end
if fix_y == 1, offset_y = 0; end
if offset_y > maxshiftn, offset_y = maxshiftn; end
if offset_y < -maxshiftn, offset_y = -maxshiftn; end
if offset_x > maxshiftn, offset_x = maxshiftn; end
if offset_x < -maxshiftn, offset_x = -maxshiftn; end
img_r_shift = circshift(img_r,[offset_y, offset_x]);

% repeat shifted line
if offset_y > 0
    img_r_shift(1:offset_y,:) = repmat(img_r(1,:),[offset_y,1]);
elseif offset_y < 0
    img_r_shift(end + offset_y + 1:end,:) = repmat(img_r(end,:),[-offset_y,1]);
end
if offset_x > 0
    img_r_shift(:,1:offset_x) = repmat(img_r(:,1),[1,offset_x]);
elseif offset_x < 0
    img_r_shift(:,end + offset_x + 1:end) = repmat(img_r(:,end),[1,-offset_x]);
end

if ShowRegisFlag == 1
    F_img_shift = fft2(img_r_shift);
    crosscorr_shift = abs(fftshift(ifft2(conj(F_img_shift) .* F_gt)));
    [my_shift, mx_shift] = find(crosscorr_shift == max(crosscorr_shift(:)));
    disp(['y,x=' num2str(my) ',' num2str(mx)]);
    disp(['After registration: y,x=' num2str(my_shift) ',' num2str(mx_shift)]);
end
img_regist = imresize(img_r_shift,1/Regis_scale,'bicubic');