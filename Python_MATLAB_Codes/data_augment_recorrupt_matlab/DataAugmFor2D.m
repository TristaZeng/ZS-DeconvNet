function DataAugmFor2D(DataPath,beta1,beta2,alpha)

if nargin<4,alpha=[0.5,1.5];end
if nargin<3,beta2=[10,15];end
if nargin<2,beta1=[0.5,1.5];end

% ------------------------------------------------------------------------
%   generate 2D dataset for ZS-DeconvNet
%   DataPath              -- the path of data to be augmented
%   beta1, beta2, alpha   -- hyper-parameters of re-corruption, typically set to
%                            1, sigma^2 and 1, where sigma^2 is the background variance. 
%                            Can also be a range, e.g., beta1=[0.5,1.5],beta2=[10,15],
%                            alpha=[1,2]
% ------------------------------------------------------------------------

%% set path and segmentation parameters 
RotFlag = 2; % =0, no rotation, =1, with rotation, =2, rotation with flipping
bg = 100; % background offset, used in re-corruption
SegX = 128; % height of augmented data pairs
SegY = 128; % width of augmented data pairs
TotalSegNum = 20000; % number of augmented data

if length(beta1)>1
    beta1_descript = [num2str(beta1(1)),'-',num2str(beta1(2))];
else
    beta1_descript = num2str(beta1);
end
if length(beta2)>1
    beta2_descript = [num2str(beta2(1)),'-',num2str(beta2(2))];
else
    beta2_descript = num2str(beta2);
end
if length(alpha)>1
    alpha_descript = [num2str(alpha(1)) '-' num2str(alpha(2))];
else
    alpha_descript = num2str(alpha);
end
SavePath = ['../your_augmented_datasets/beta1_' beta1_descript '_beta2_' beta2_descript '_alpha' alpha_descript '_SegNum' num2str(TotalSegNum) '/'];
save_input_path = [SavePath 'input/'];
save_gt_path = [SavePath 'gt/'];
mkdir('../your_augmented_datasets');
if ~exist(SavePath, 'dir'), mkdir(SavePath); end
if isfolder(save_input_path)
    disp('input directory already exists, check carefully.');
else
    mkdir(save_input_path); 
end
if isfolder(save_gt_path) 
    disp('gt directory already exists, check carefully.');
else
    mkdir(save_gt_path); 
end

%% Generate training patches
ImgList = XxSort(XxDir(DataPath, '*'));
cell_num = length(ImgList);
for i = 1:cell_num
    fprintf('Processing File %d/%d\n',i,cell_num);
    % read data
    if contains(ImgList{i},'mrc')
        [header, data]  = XxReadMRC(ImgList{i});
        data = reshape(data, header(1), header(2), header(3));
    else
        data = XxReadTiffSmallerThan4GB(ImgList{i});
    end
    data = single(data);
    data_gt = data;
    data_comp = data;
    
    % augment
    if RotFlag < 2
        [data_seg, data_gt_seg, ~] = XxDataSeg_ForTrain(data, data_gt, data_comp,...
            max(round(TotalSegNum/cell_num),1) , SegX, SegY, RotFlag);
        num_image = size(data_gt_seg, 3);
    else
        [data_seg, data_gt_seg, ~] = XxDataSeg_ForTrain(data, data_gt, data_comp,...
            max(round(TotalSegNum/cell_num/4),1) , SegX, SegY, RotFlag);
        % flip axis y
        curdata = flipud(data);
        curdata_gt = flipud(data_gt);
        curdata_comp = flipud(data_comp);
        [data_seg_ap, data_gt_seg_ap, ~] = XxDataSeg_ForTrain(curdata, curdata_gt, curdata_comp,...
            max(round(TotalSegNum/cell_num/4),1), SegX, SegY, RotFlag);
        num_image = size(data_gt_seg_ap, 3);
        data_seg(:, :, end+1:end+num_image) = data_seg_ap;
        data_gt_seg(:, :, end+1:end+num_image) = data_gt_seg_ap;
        % flip axis x
        curdata = fliplr(data);
        curdata_gt = fliplr(data_gt);
        curdata_comp = fliplr(data_comp);
        [data_seg_ap, data_gt_seg_ap, ~] = XxDataSeg_ForTrain(curdata, curdata_gt, curdata_comp,...
            max(round(TotalSegNum/cell_num/4),1), SegX, SegY, RotFlag);
        num_image = size(data_gt_seg_ap, 3);
        data_seg(:, :, end+1:end+num_image) = data_seg_ap;
        data_gt_seg(:, :, end+1:end+num_image) = data_gt_seg_ap;
        % flip axis x and y
        curdata = flipud(curdata);
        curdata_gt = flipud(curdata_gt);
        curdata_comp = flipud(curdata_comp);
        [data_seg_ap, data_gt_seg_ap, ~] = XxDataSeg_ForTrain(curdata, curdata_gt, curdata_comp,...
            max(round(TotalSegNum/cell_num/4),1), SegX, SegY, RotFlag);
        num_image = size(data_gt_seg_ap, 3);
        data_seg(:, :, end+1:end+num_image) = data_seg_ap;
        data_gt_seg(:, :, end+1:end+num_image) = data_gt_seg_ap;
        num_image = size(data_gt_seg, 3);
    end

    for n = 1:num_image
        
        img_gt = data_gt_seg(:,:,n);
        img_input = data_seg(:,:,n);

        % re-corrupt
        if length(beta1)>1
            beta1_cur = beta1(1)+(beta1(2)-beta1(1))*rand;
        else
            beta1_cur = beta1;
        end  
        if length(beta2)>1
            beta2_cur = beta2(1)+(beta2(2)-beta2(1))*rand;
        else
            beta2_cur = beta2;
        end
        if length(alpha)>1
            alpha_cur = alpha(1)+(alpha(2)-alpha(1))*rand;
        else
            alpha_cur = alpha;
        end
        mean_filter = fspecial('average',5);
        sigma = sqrt(max(beta1_cur*max(conv2(img_gt,mean_filter,'same')-bg,0)+beta2_cur,0));
        z = normrnd(0,1,size(img_gt));
        img_gt = img_gt-sigma.*z/alpha_cur;
        img_input = img_input+sigma.*z*alpha_cur;
        
        % normalize
        img_input = XxNorm(img_input);
        img_gt = XxNorm(img_gt);
        img_input = uint16(img_input * 65535); 
        img_gt = uint16(img_gt * 65535); 

        % save image
        fprintf('Producing stack %d in cell %d\n',n,i);
        imwrite(img_input,[save_input_path '/cell' num2str(i,'%02d') '_' num2str(n,'%08d') '.tif']);
        imwrite(img_gt,[save_gt_path '/cell' num2str(i,'%02d') '_' num2str(n,'%08d') '.tif']); 
        
    end

end