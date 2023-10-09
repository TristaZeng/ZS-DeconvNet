clear;
close all;
cd 'C:\Users\Admin\Git\repo\ZS-DeconvNet\Python_MATLAB_Codes\data_augment_recorrupt_matlab'; % change this path to your cloned path
addpath(genpath('./XxUtils'));
addpath(genpath('./GenData4ZS-DeconvNet-SIM'))

% It should be noted that, the raw data loading code of this demo is based on the file structure of one particular dataset, and may not
% be consistent with your raw data layout.
%% customized setting
repeat_time = 3;
save_dir_root = 'F:/DATA_part1/3DSIM';
root_dir = 'F:/DATA_part1/3DSIM';

file_list = dir([root_dir,'/*Ensconsin.mrc']);
file_count = length(file_list);
if ~exist(save_dir_root,'dir')
    mkdir(save_dir_root);
end

%% basic setting
n_phases = 5;
n_dirs = 3;
img_baseline = 100;
alpha_v_min = 0.1; 
alpha_v_max = 0.2;
beta_1_min = 1.5;
beta_1_max = 3;
beta_2_min = 20;
beta_2_max = 35;

%% Gaussian filter
gaussian_filter = fspecial('gaussian',[3 3],1.2);

%% parallel computing
% CoreNum = 8;
% if isempty(gcp('nocreate'))
%     parpool(CoreNum);
% end


%% final processing
id_min = 1;
id_max = 3;
for cur_id = id_min: id_max
    fprintf(['************************************************************\n']);
    fprintf(['************************************************************\n']);
    fprintf(['Processing snr ', num2str(cur_id,'%02d'), '!\n']);
    fprintf(['************************************************************\n']);
    fprintf(['************************************************************\n']);
    
    save_dir = [save_dir_root,'/','snr_',num2str(cur_id)];
    if ~exist(save_dir,'dir')
        mkdir(save_dir);
    end
    
    for cell_id = 1: file_count  
        fprintf(['Processing cell ', num2str(cell_id,'%03d'), '!\n']);
        
        file_name = file_list(cell_id).name;
        img_file = [root_dir,'/',file_name];
        
        [header, data] = XxReadMRC(img_file);

        Nx = double(single(header(1)));
        Ny = double(single(header(2)));
        N_slice = double(single(header(3)));
        temp = typecast(header(46),'int16');
        Nt = double(temp(1));
        Nz = N_slice / (n_phases*n_dirs) / Nt;
        data = double(reshape(data,[Nx, Ny, N_slice/Nt, Nt]));
        figure,imshow(data(:,:,1,1),[]);
        
        
        header_out = header;
        header_out(3) = int32(N_slice/Nt);
        header_out(46) = typecast([int16(1),int16(1)],'int32');
        
        img_raw = data(:,:,:,cur_id);
        
        file_name_raw = ['Cell_',num2str(cell_id,'%02d'),'-raw','.mrc'];
        handle = fopen([save_dir,'/',file_name_raw],'w+');
        handle = XxWriteMRC_SmallEndian(handle, uint16(img_raw), header_out);
        fclose(handle);
        
        
        h = size(img_raw,1);
        w = size(img_raw,2);
        slice_num = size(img_raw,3);
        
        for repeat_id = 1: 1: repeat_time
            alpha_v = (alpha_v_max-alpha_v_min)*rand() + alpha_v_min;
            D = 1/alpha_v * ones(h,w);
            D_1 = alpha_v * ones(h,w);

            beta_1 = (beta_1_max-beta_1_min)*rand()+beta_1_min;
            beta_2 = (beta_2_max-beta_2_min)*rand()+beta_2_min;

            img_raw1 = zeros(h,w,slice_num);
            img_raw2 = zeros(h,w,slice_num);

            img_tmp = max(img_raw-img_baseline,0);
            for ss = 1: 1: slice_num
                img_tmp(:,:,ss) = imfilter(img_tmp(:,:,ss),gaussian_filter,'replicate','same');
            end
            for ss = 1: 1: slice_num
                zz = normrnd(0,1,[h,w]);

                Sigma_x = max(beta_1 * img_tmp(:,:,ss) + beta_2,0);

                img_raw1(:,:,ss) = img_raw(:,:,ss) + D .* sqrt(Sigma_x) .* zz;
                img_raw2(:,:,ss) = img_raw(:,:,ss) - D_1 .* sqrt(Sigma_x) .* zz;
            end

            % input
            img_raw1 = uint16(img_raw1);       
            file_name_input = ['Cell_',num2str(cell_id,'%02d'),'-input',num2str(repeat_id),'.mrc']; 
            handle = fopen([save_dir,'/',file_name_input],'w+');
            handle = XxWriteMRC_SmallEndian(handle, img_raw1, header_out);
            fclose(handle);

            % target
            img_raw2 = uint16(img_raw2);   
            file_name_target = ['Cell_',num2str(cell_id,'%02d'),'-target',num2str(repeat_id),'.mrc']; 
            handle = fopen([save_dir,'/',file_name_target],'w+');
            handle = XxWriteMRC_SmallEndian(handle, img_raw2, header_out);
            fclose(handle);
        end

    end
end
% delete(gcp('nocreate'));





