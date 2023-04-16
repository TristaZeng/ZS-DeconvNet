function [data_seg_l,data_seg_h,gt_seg] = XxDataSeg_ForTrain(data_l, data_h, gt, num_seg, Nx, Ny, RotFlag)

addpath(genpath('./XxUtils'));

[r,l,~] = size(data_l);
[r_h,~,~] = size(gt);
sr_ratio = round(r_h / r);

if RotFlag == 0
    new_r = Nx;
    new_l = Ny;
else
    new_r = ceil(Nx * 1.5);
    new_l = ceil(Ny * 1.5);
    if new_r > r || new_l > l
        new_r = r;
        new_l = l;
        RotFlag = 0;
    end
end

% calculate foreground mask
thresh_mask = 0.07*max(gt(:));
ksize = 3;
gt_cal_mask = imresize(gt, 1/sr_ratio, 'bicubic');
mask = XxCalMask(gt_cal_mask,ksize,thresh_mask);
while sum(mask(:)) < 1e3
    thresh_mask = thresh_mask * 0.8;
    mask = XxCalMask(gt_cal_mask,ksize,thresh_mask);
end

% figure(1);
% subplot(1,2,1), imshow(gt,[]);
% subplot(1,2,2), imshow(mask,[]);

[X,Y] = meshgrid(1:l,1:r);
point_list = zeros(sum(mask(:)),2);
point_list(:,1) = Y(mask(:));
point_list(:,2) = X(mask(:));
l_list = size(point_list,1);

halfx = round(new_r / 2);
halfy = round(new_l / 2);

data_seg_l = [];
data_seg_h = [];
gt_seg = [];

%% Crop patches
% thresh is another method to ensure the content of patches except for mask
thresh_ar = 0;
thresh_ar_gt = 0;
thresh_sum_gt = 0;
count = 0;

for i = 1:num_seg
    % pick patch center points according to the mask
    p = randi(l_list,1);
    y1 = point_list(p, 2) - halfy + 1;
    y2 = point_list(p, 2) + halfy;
    x1 = point_list(p, 1) - halfx + 1;
    x2 = point_list(p, 1) + halfx;
    count_rand = 0;
    while (y1<2 || y2>l || x1<2 || x2>r)
        p = randi(l_list,1);
        y1 = point_list(p, 2) - halfy + 1;
        y2 = point_list(p, 2) + halfy;
        x1 = point_list(p, 1) - halfx + 1;
        x2 = point_list(p, 1) + halfx;
        count_rand = count_rand + 1;
        if count_rand > 1e3, break; end
    end
    if count_rand > 1e3, break; end
    
    if RotFlag >= 1 % if random rotate, rotate the chosen patch
        degree = randi(360, 1);
        patch_l = imrotate(sum(data_l(x1:x2,y1:y2,:), 3),degree,'bilinear','crop');
        patch_h = imrotate(sum(data_h(x1:x2,y1:y2,:), 3),degree,'bilinear','crop');
        tx = round(new_r/2)-round(Nx/2);
        ty = round(new_l/2)-round(Ny/2);
        patch_l = patch_l(tx+1:tx+Nx,ty+1:ty+Ny);
        patch_h = patch_h(tx+1:tx+Nx,ty+1:ty+Ny);
        
        patch_gt = imrotate(gt(sr_ratio*x1-1:sr_ratio*x2,sr_ratio*y1-1:sr_ratio*y2,1),degree,'bilinear','crop');
        tx_gt = round(new_r*sr_ratio/2)-round(Nx*sr_ratio/2);
        ty_gt = round(new_l*sr_ratio/2)-round(Ny*sr_ratio/2);
        patch_gt = patch_gt(tx_gt+1:tx_gt+sr_ratio*Nx,ty_gt+1:ty_gt+sr_ratio*Ny);
    else % not random rotate
        patch_l = sum(data_l(x1:x2,y1:y2,:), 3);
        patch_h = sum(data_h(x1:x2,y1:y2,:), 3);
        patch_gt = gt(sr_ratio*x1-1:sr_ratio*x2,sr_ratio*y1-1:sr_ratio*y2,1);
    end
    
    sum_patch_gt = sum(patch_gt(:));
    active_range = double(prctile(patch_l(:),99.9)) / double(prctile(patch_l(:),0.1)+1);
    active_range_gt = double(prctile(patch_gt(:),99.9)) / double(prctile(patch_gt(:),0.1)+1);
    
    while active_range_gt < thresh_ar_gt || sum_patch_gt < thresh_sum_gt || active_range < thresh_ar
        x1 = randi([2,r - new_r + 1], 1);
        x2 = x1 + new_r - 1;
        y1 = randi([2,l - new_l + 1], 1);
        y2 = y1 + new_l - 1;
        
        if RotFlag >= 1 % if random rotate
            degree = randi(360, 1);
            patch_l = imrotate(sum(data_l(x1:x2,y1:y2,:), 3),degree,'bilinear','crop');
            patch_h = imrotate(sum(data_h(x1:x2,y1:y2,:), 3),degree,'bilinear','crop');
            tx = round(new_r/2)-round(Nx/2);
            ty = round(new_l/2)-round(Ny/2);
            patch_l = patch_l(tx+1:tx+Nx,ty+1:ty+Ny);
            patch_h = patch_h(tx+1:tx+Nx,ty+1:ty+Ny);
            
            patch_gt = imrotate(gt(sr_ratio*x1-1:sr_ratio*x2,sr_ratio*y1-1:sr_ratio*y2,1),degree,'bilinear','crop');
            tx_gt = round(new_r*sr_ratio/2)-round(Nx*sr_ratio/2);
            ty_gt = round(new_l*sr_ratio/2)-round(Ny*sr_ratio/2);
            patch_gt = patch_gt(tx_gt+1:tx_gt+sr_ratio*Nx,ty_gt+1:ty_gt+sr_ratio*Ny);
        else % not random rotate
            patch_l = sum(data_l(x1:x2,y1:y2,:), 3);
            patch_h = sum(data_h(x1:x2,y1:y2,:), 3);
            patch_gt = gt(sr_ratio*x1-1:sr_ratio*x2,sr_ratio*y1-1:sr_ratio*y2,1);
        end
        active_range = double(prctile(patch_l(:),99.9)) / double(prctile(patch_l(:),0.1)+1e-2);
        sum_patch_gt = sum(patch_gt(:));
        active_range_gt = double(prctile(patch_gt(:),99.9)) / double(prctile(patch_gt(:),0.1)+1);

        count = count + 1;
        if count > 1e2
            thresh_ar_gt = thresh_ar_gt * 0.9;
            thresh_sum_gt = thresh_sum_gt * 0.9;
            thresh_ar = thresh_ar * 0.9;
            count = 0;
        end
        
%         fprintf('ar_gt    = %.2f\n',active_range_gt);
%         fprintf('sum_gt   = %.2f\n',sum_patch_gt);
%         fprintf('ar_data  = %.2f\n',active_range);
%         figure(1);
%         subplot(1,2,1), imagesc(patch(:,:,1)); axis image, axis off;
%         subplot(1,2,2), imagesc(patch_gt(:,:,1)); axis image, axis off;
    end
    
    if RotFlag >= 1
        tdata_l = imrotate(data_l(x1:x2,y1:y2,:),degree,'bilinear','crop');
        tdata_h = imrotate(data_h(x1:x2,y1:y2,:),degree,'bilinear','crop');
        if sr_ratio==1
            tgt = imrotate(gt(x1:x2,y1:y2,:),degree,'bilinear','crop');
        else
            tgt = imrotate(gt(sr_ratio*x1-1:sr_ratio*x2,sr_ratio*y1-1:sr_ratio*y2,:),degree,'bilinear','crop');
        end
        
        if isempty(data_seg_l)
            data_seg_l = tdata_l(tx+1:tx+Nx,ty+1:ty+Ny,:);
        else
            h  = size(tdata_l, 3);
            data_seg_l(:,:,end+1:end+h) = tdata_l(tx+1:tx+Nx,ty+1:ty+Ny,:);
        end
        if isempty(data_seg_h)
            data_seg_h = tdata_h(tx+1:tx+Nx,ty+1:ty+Ny,:);
        else
            h  = size(tdata_h, 3);
            data_seg_h(:,:,end+1:end+h) = tdata_h(tx+1:tx+Nx,ty+1:ty+Ny,:);
        end
        if isempty(gt_seg)
            gt_seg = tgt(tx_gt+1:tx_gt+sr_ratio*Nx,ty_gt+1:ty_gt+sr_ratio*Ny,:);
        else
            h_gt = size(tgt, 3);
            gt_seg(:,:,end+1:end+h_gt) = tgt(tx_gt+1:tx_gt+sr_ratio*Nx,ty_gt+1:ty_gt+sr_ratio*Ny,:);
        end
    else
        if isempty(data_seg_l)
            data_seg_l = data_l(x1:x2,y1:y2,:);
        else
            h  = size(data_l, 3);
            data_seg_l(:,:,end+1:end+h) = data_l(x1:x2,y1:y2,:);
        end
        if isempty(data_seg_h)
            data_seg_h = data_h(x1:x2,y1:y2,:);
        else
            h  = size(data_h, 3);
            data_seg_h(:,:,end+1:end+h) = data_h(x1:x2,y1:y2,:);
        end
        if isempty(gt_seg)
            gt_seg = gt(sr_ratio*x1-1:sr_ratio*x2,sr_ratio*y1-1:sr_ratio*y2,:);
        else
            h_gt = size(gt, 3);
            gt_seg(:,:,end+1:end+h_gt) = gt(sr_ratio*x1-1:sr_ratio*x2,sr_ratio*y1-1:sr_ratio*y2,:);
        end
    end
end

end