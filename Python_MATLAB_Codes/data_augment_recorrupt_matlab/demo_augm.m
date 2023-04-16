cd 'C:/Users/Admin/Git/repo/ZS-DeconvNet-master/data_augment_recorrupt_matlab';
addpath(genpath('./XxUtils'));
% set data_folder to the folder you keep your raw data
% data_folder should contain only raw data files
data_folder = 'D:/ZS_DeconvNet/1stRevision/ZS-DeconvNet datasets/2D data/Lysosome/train_data';
DataAugmFor2D(data_folder);
data_folder = 'F:/ZS-DeconvNet datasets/3D data/LLSM/Mitochondria';
DataAugmFor3D(data_folder);