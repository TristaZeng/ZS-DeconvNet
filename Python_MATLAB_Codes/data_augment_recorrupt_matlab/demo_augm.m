cd 'C:\Users\Admin\Git\repo\ZS-DeconvNet\Python_MATLAB_Codes\data_augment_recorrupt_matlab';
addpath(genpath('./XxUtils'));

data_folder = 'F:/ZS-DeconvNet datasets/2D data/Lysosome/train_data'; % set data_folder to the folder you keep your raw data
bg = 100; % background offset, used in re-corruption
DataAugmFor2D(data_folder,bg);

data_folder = 'F:/ZS-DeconvNet datasets/3D data/LLSM/Mitochondria';
DataAugmFor3D(data_folder);