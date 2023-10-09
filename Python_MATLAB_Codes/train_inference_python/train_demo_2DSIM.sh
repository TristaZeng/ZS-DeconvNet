
# ------------------------------- training arguments -------------------------------

# # models
# --conv_block_num: number of upsampling or downsampling block in U-Net
# --conv_num: number of convolution layer in each conv_block
# --upsample_flag: 0 or 1, whether the network upsamples the image

# # training settings
# --gpu_id: the gpu device you want to use in current task
# --gpu_memory_fraction: upper bound of gpu memory fraction that can be used
# --mixed_precision_training: whether use mixed precision training or not
# --iterations: total training iterations
# --test_interval: iteration interval of testing
# --valid_interval: iteration interval of validation
# --load_init_model_iter: loading weights interations

# # lr decay
# --start_lr: initial learning rate of training, typically set as 5e-5
# --lr_decay_factor: learning rate decay factor, typically set as 0.5

# # about data
# --psf_path: path of corresponding PSF
# --dxypsf: (only needed when psf_src_mode==1, can be read from TIFF) lateral sampling interval (um) of raw PSF, if dxypsf is not equal to dx, interpolation will be performed
# --dx: sampling interval in x direction (um), needs to be 1/2 of raw data dx when upsample_flag=1
# --dy: sampling interval in y direction (um), needs to be 1/2 of raw data dy when upsample_flag=1

# You need to arrange your data in the format: data_dir+folder+'/input/' and data_dir+folder+'/gt/'
# --data_dir: the root directory of training data folder
# --augment_flag: 1 or 0, whether or not to augment the data in data_dir+folder+'/input/' and data_dir+folder+'/gt/'
# --norm_flag: different normalization techniques
# --folder: the name of training data folder
# --test_images_path: the root path of test data
# --save_weights_dir: root directory where model weights will be saved in
# --save_weights_suffix: suffix for the folder name when saving models

# --batch_size: batch size for training
# --input_y: the height of input image patches
# --input_x: the width of input image patches
# --insert_xy: padded blank margin in pixels
# --input_y_test: the height of test image
# --input_x_test: the width of test image

# --valid_num: number of sampled images from training set

# # about loss functions
# --mse_flag: 0 for mae, 1 for mse
# --denoise_loss_weight: the weighting factor for denoising loss term
# --l1_rate: the weighting factor for L1 regularization term
# --TV_rate: the weighting factor for TV regularization term
# --Hess_rate: the weighting factor for Hessian regularization term

# ------------------------------- examples -------------------------------

cd C:/Users/Admin/Git/repo/ZS-DeconvNet/Python_MATLAB_Codes/train_inference_python
python Train_ZSDeconvNet_2DSIM.py --psf_path 'C:/Users/Admin/Git/repo/attachments_to_repos/SIM/SIM_code/psf/PSF-2D-SIM.tif' \
                                --data_dir 'D:/ZS_DeconvNet/R2R/TestSIM/' \
                                --folder 'ER' \
                               --test_images_path 'D:/ZS_DeconvNet/R2R/20220312_MainFig1/Lamp1/test_3addup/input/cell02.tif'


