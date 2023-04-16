function DataAugmFor3D(DataPath,SavePath,SegX,SegZ)

addpath(genpath('./XxUtils'));
if nargin < 4, SegZ = 13; end
if nargin < 3, SegX = 64; end
if nargin < 2, SavePath = '../your_augmented_datasets/';end
% ------------------------------------------------------------------------
%   generate 3D dataset for ZS-DeconvNet
%   SavePath   -- the path of data to be saved
%   DataPath   -- the path of data to be augmented
%   SegX       -- height of augmented stacks
%   SegZ       -- depth of augmented stacks
% ------------------------------------------------------------------------

%% set path and segmentation parameters
SegY = SegX;
thresh_mask = 1e-2;
active_range_thresh = 0.5;
sum_thresh = 0.01*SegX*SegY*SegZ;
SegNum = 10000;
Rot_flag = 1; % =0, no rotation, =1, random rotation

save_training_path = [SavePath 'Zsize' num2str(SegZ) '_Xsize' num2str(SegX)];
mkdir('../your_augmented_datasets');
if ~exist(SavePath, 'dir'), mkdir(SavePath); end
if ~exist(save_training_path, 'dir'), mkdir(save_training_path); end
input_path = [save_training_path filesep 'input'];
gt_path = [save_training_path filesep 'gt'];
if ~exist(input_path,'dir')
    mkdir(input_path);
    mkdir(gt_path);
end

%% generating data
if Rot_flag
    halfx = floor(SegX*1.5/2);
    halfy = floor(SegY*1.5/2);
    tx = halfx-round(SegX/2);
    ty = halfy-round(SegY/2);
else
    halfx = floor(SegX/2);
    halfy = floor(SegY/2);
end
fileList = XxSort(XxDir(DataPath,'*'));   
cellnum  = length(fileList);
n_per_stack = max(ceil(SegNum/cellnum),1);
n_total = 0;
for t = 1:cellnum
    fprintf('Processing File %d/%d\n',t,cellnum);
    if contains(fileList{t},'mrc')
        [header, data] = XxReadMRC(fileList{t});
        data = reshape(data,header(1),header(2),header(3));
    else
        data = XxReadTiffSmallerThan4GB(fileList{t});
    end
    data = single(data);
    data(data<0) = 0;
    data = XxNorm(data);

    % crop data
    cur_thresh_mask = thresh_mask;
    mask = XxCalMask(data,10,cur_thresh_mask);
    ntry = 0;
    while sum(mask(:)) < n_per_stack
        cur_thresh_mask = cur_thresh_mask * 0.8;
        mask = XxCalMask(data,10,cur_thresh_mask);
        ntry = ntry + 1;
        if ntry > 1e3, break; end
    end
  
    [X,Y,Z] = meshgrid(1:size(data,2),1:size(data,1),1:size(data,3)); 
    point_list = zeros(sum(mask(:)),3);
    point_list(:,1) = Y(mask(:));
    point_list(:,2) = X(mask(:));
    point_list(:,3) = Z(mask(:));
    l_list = size(point_list,1);
    
    n_left = n_per_stack;
    ntry = 0;
    while n_left >= 1
        ntry = ntry + 1;
        if ntry > 1e5, break; end
        p = randi(l_list,1);
        x1 = point_list(p, 1) - halfx + 1;
        x2 = point_list(p, 1) + halfx;
        y1 = point_list(p, 2) - halfy + 1;
        y2 = point_list(p, 2) + halfy;
        z1 = point_list(p, 3) - SegZ + 1;
        z2 = point_list(p, 3) + SegZ;
        if x1 < 1 || y1 < 1 || z1 < 1, continue; end
        if x2 > size(data,1) || y2 > size(data,2) || z2 > size(data,3), continue; end
        input_crop = data(x1:x2,y1:y2,z1+1:2:z2);
        active_range = double(prctile(input_crop(:),99.9)) / double(prctile(input_crop(:),0.1)+1e-2);
        if active_range < active_range_thresh, continue; end
        sum_value = sum(input_crop(:));
        if sum_value < sum_thresh, continue; end
        
        input_crop = data(x1:x2,y1:y2,z1+1:2:z2);
        gt_crop = data(x1:x2,y1:y2,z1:2:z2);
    
        % save the pair
        if Rot_flag
            degree = randi(360, 1);
            input_crop_tmp = input_crop;
            input_crop = zeros(SegX,SegY,SegZ);
            gt_crop_tmp = gt_crop;
            gt_crop = zeros(SegX,SegY,SegZ);
            for z_ind = 1:size(input_crop,3)
                tmp = imrotate(input_crop_tmp(:,:,z_ind),degree,'bilinear','crop');
                input_crop(:,:,z_ind) = tmp(tx+1:tx+SegX,ty+1:ty+SegY);
                tmp = imrotate(gt_crop_tmp(:,:,z_ind),degree,'bilinear','crop');
                gt_crop(:,:,z_ind) = tmp(tx+1:tx+SegX,ty+1:ty+SegY);
            end
        end
        n_total = n_total + 1;
        n_left = n_left - 1;
        fprintf('Producing stack %d\n',n_total);
        imwriteTFSK(uint16(65535*input_crop), [input_path filesep num2str(n_total) '.tif']);
        imwriteTFSK(uint16(65535*gt_crop), [gt_path filesep num2str(n_total) '.tif']);

        input_crop = flipud(data(x1:x2,y1:y2,z1+1:2:z2));
        gt_crop = flipud(data(x1:x2,y1:y2,z1:2:z2));
    
        % save the pair
        if Rot_flag
            degree = randi(360, 1);
            input_crop_tmp = input_crop;
            input_crop = zeros(SegX,SegY,SegZ);
            gt_crop_tmp = gt_crop;
            gt_crop = zeros(SegX,SegY,SegZ);
            for z_ind = 1:size(input_crop,3)
                tmp = imrotate(input_crop_tmp(:,:,z_ind),degree,'bilinear','crop');
                input_crop(:,:,z_ind) = tmp(tx+1:tx+SegX,ty+1:ty+SegY);
                tmp = imrotate(gt_crop_tmp(:,:,z_ind),degree,'bilinear','crop');
                gt_crop(:,:,z_ind) = tmp(tx+1:tx+SegX,ty+1:ty+SegY);
            end
        end
        n_total = n_total + 1;
        n_left = n_left - 1;
        fprintf('Producing stack %d\n',n_total);
        imwriteTFSK(uint16(65535*input_crop), [input_path filesep num2str(n_total) '.tif']);
        imwriteTFSK(uint16(65535*gt_crop), [gt_path filesep num2str(n_total) '.tif']);

        input_crop = fliplr(data(x1:x2,y1:y2,z1+1:2:z2));
        gt_crop = fliplr(data(x1:x2,y1:y2,z1:2:z2));
    
        % save the pair
        if Rot_flag
            degree = randi(360, 1);
            input_crop_tmp = input_crop;
            input_crop = zeros(SegX,SegY,SegZ);
            gt_crop_tmp = gt_crop;
            gt_crop = zeros(SegX,SegY,SegZ);
            for z_ind = 1:size(input_crop,3)
                tmp = imrotate(input_crop_tmp(:,:,z_ind),degree,'bilinear','crop');
                input_crop(:,:,z_ind) = tmp(tx+1:tx+SegX,ty+1:ty+SegY);
                tmp = imrotate(gt_crop_tmp(:,:,z_ind),degree,'bilinear','crop');
                gt_crop(:,:,z_ind) = tmp(tx+1:tx+SegX,ty+1:ty+SegY);
            end
        end
        n_total = n_total + 1;
        n_left = n_left - 1;
        fprintf('Producing stack %d\n',n_total);
        imwriteTFSK(uint16(65535*input_crop), [input_path filesep num2str(n_total) '.tif']);
        imwriteTFSK(uint16(65535*gt_crop), [gt_path filesep num2str(n_total) '.tif']);

        input_crop = flipud(fliplr(data(x1:x2,y1:y2,z1+1:2:z2)));
        gt_crop = flipud(fliplr(data(x1:x2,y1:y2,z1:2:z2)));
    
        % save the pair
        if Rot_flag
            degree = randi(360, 1);
            input_crop_tmp = input_crop;
            input_crop = zeros(SegX,SegY,SegZ);
            gt_crop_tmp = gt_crop;
            gt_crop = zeros(SegX,SegY,SegZ);
            for z_ind = 1:size(input_crop,3)
                tmp = imrotate(input_crop_tmp(:,:,z_ind),degree,'bilinear','crop');
                input_crop(:,:,z_ind) = tmp(tx+1:tx+SegX,ty+1:ty+SegY);
                tmp = imrotate(gt_crop_tmp(:,:,z_ind),degree,'bilinear','crop');
                gt_crop(:,:,z_ind) = tmp(tx+1:tx+SegX,ty+1:ty+SegY);
            end
        end
        n_total = n_total + 1;
        n_left = n_left - 1;
        fprintf('Producing stack %d\n',n_total);
        imwriteTFSK(uint16(65535*input_crop), [input_path filesep num2str(n_total) '.tif']);
        imwriteTFSK(uint16(65535*gt_crop), [gt_path filesep num2str(n_total) '.tif']);

    end
end