clear;
close all;
cd 'C:\Users\Admin\Git\repo\ZS-DeconvNet\Python_MATLAB_Codes\data_augment_recorrupt_matlab'; % change this path to your cloned path
addpath(genpath('./XxUtils'));
addpath(genpath('./GenData4ZS-DeconvNet-SIM'));

% It should be noted that, the raw data loading code of this demo is based on the file structure of BioSR dataset, and may not
% be consistent with your raw data layout.
%% customized setting
save_root_dir = 'F:/DATA_part1/BioSR/Microtubules_corrupt/'; % where to save re-corrupted data
root_dir = 'F:/DATA_part1/BioSR/Microtubules/'; % where to load raw data

%% SIM img param
n_dirs = 3;
n_phases = 3;
img_baseline = 100;

%% r2r parameters
repeat_time = 3;
alpha_v_min = 0.2; 
alpha_v_max = 0.5;
beta_1_min = 1.5;
beta_1_max = 3;
beta_2_min = 20;
beta_2_max = 35;

%% sample
sub_dirs = dir([root_dir]);
sub_dirs = sub_dirs(3:end-1);
sub_dir_count = length(sub_dirs);

if ~exist(save_root_dir,'dir')
    mkdir(save_root_dir);
end

%% filter
gaussian_filter = fspecial('gaussian',[3 3],1.2);

%% re-corrupt
for smpl_id = 1: 1: sub_dir_count
    fprintf(['--------------------------------------------------\n']);
    fprintf(['Process cell ',num2str(smpl_id),' !\n']);
    fprintf(['--------------------------------------------------\n']);
    
    smpl_dir = [root_dir,'/',sub_dirs(smpl_id).name];
    
    for snr_id = 1: 1: 8
        
        smpl_file = ['RawSIMData_level_',num2str(snr_id,'%02d'),'.mrc'];
        [header, data] = XxReadMRC([smpl_dir,'/',smpl_file]);

        Nx = double(header(1));
        Ny = double(header(2));
        N_slice = double(header(3));
        temp = typecast(header(46),'int16');
        Nt = double(temp(1));
        Nz = N_slice / (n_phases*n_dirs) / Nt;

        data = double(reshape(data,[Nx, Ny, n_phases*n_dirs*Nz, Nt]));
        
        img_raw = data;
        file_name_raw = ['smpl_',num2str(smpl_id,'%02d'),'_snr_',num2str(snr_id,'%02d'),'-raw','.mrc'];
        handle = fopen([save_root_dir,'/',file_name_raw],'w+');
        handle = XxWriteMRC_SmallEndian(handle, uint16(img_raw), header);
        fclose(handle);            

        h = size(img_raw, 1);
        w = size(img_raw, 2);
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
                z_n = normrnd(0,1,[h,w]);

                Sigma_x = max(beta_1 * img_tmp(:,:,ss) + beta_2,0);

                img_raw1(:,:,ss) = img_raw(:,:,ss) + D .* sqrt(Sigma_x) .* z_n;
                img_raw2(:,:,ss) = img_raw(:,:,ss) - D_1 .* sqrt(Sigma_x) .* z_n;
            end
            img_raw1 = uint16(img_raw1);
            file_name_input= ['smpl_',num2str(smpl_id,'%02d'),'_snr_',num2str(snr_id,'%02d'),'-input',num2str(repeat_id),'.mrc']; 
            handle = fopen([save_root_dir,'/',file_name_input],'w+');
            handle = XxWriteMRC_SmallEndian(handle, img_raw1, header);
            fclose(handle);

            % target
            img_raw2 = uint16(img_raw2);
            file_name_target = ['smpl_',num2str(smpl_id,'%02d'),'_snr_',num2str(snr_id,'%02d'),'-target',num2str(repeat_id),'.mrc']; 
            handle = fopen([save_root_dir,'/',file_name_target],'w+');
            handle = XxWriteMRC_SmallEndian(handle, img_raw2, header);
            fclose(handle);
        end  
    end
end



