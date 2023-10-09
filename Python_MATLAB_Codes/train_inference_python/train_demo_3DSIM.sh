
# ------------------------------- training arguments -------------------------------

# models
# --model: "twostage_RCAN3D", "twostage_RCAN3D_compact" or "twostage_RCAN3D_compact2"
# --upsample_flag: 0 or 1, whether the network upsamples the image

# training settings
# --load_all_data: 1 for load all training set into memory before training for faster computation, 0 for not
# --gpu_id: the gpu device you want to use in current task
# --gpu_memory_fraction: upper bound of gpu memory fraction that can be used
# --mixed_precision_training: whether use mixed precision training or not
# --iterations: total training iterations
# --test_interval: iteration interval of testing
# --valid_interval: iteration interval of validation
# --load_init_model_iter: loading weights interations

# lr decay
# --start_lr: initial learning rate of training, typically set as 1e-4
# --lr_decay_factor: learning rate decay factor, typically set as 0.5

# about data
# You need to arrange your data in the format: data_dir+folder+'/input/' and data_dir+folder+'/gt/'
# --psf_path: path of corresponding PSF. Do not support .mrc for now, may add the function later.
# --data_dir: the root directory of training data folder
# --augment_flag: 1 or 0, whether or not to augment the data in data_dir+folder+'/input/' and data_dir+folder+'/gt/'
# --norm_flag: different normalization techniques
# --folder: the name of training data folder
# --test_images_path: the root path of test data
# --save_weights_dir: root directory where model weights will be saved in
# --save_weights_suffix: suffix for the folder name when saving models
# --background: set to the value you want to extract in test image

# --input_y: the height of input image patches
# --input_x: the width of input image patches
# --insert_xy: padded blank margin in pixels
# --input_z: the depth of input image patches
# --insert_z: padded blank margin in axial direction (how many slices)
# --batch_size: batch size for training
# --dx: sampling interval in x direction (um) for training data, needs to be 1/2 of raw data dx when upsample_flag=1
# --dz: sampling interval in z direction (um) for training data, needs to be 2 times the dz of raw data
# --dxpsf: sampling interval in x direction (um) of raw PSF, if dxpsf is not equal to dx, interpolation will be performed. Can be read from TIFF.
# --dzpsf: sampling interval in z direction (um) of raw PSF, if dzpsf is not equal to dz, interpolation will be performed. Can be read from TIFF.
# --wavelength: excitation wavelength (nm)

# about loss functions
# --mse_flag: 0 for mae, 1 for mse
# --TV_rate: the weighting factor for TV regularization term
# --Hess_rate: the weighting factor for Hessian regularization term

cd C:/Users/Admin/Git/repo/ZS-DeconvNet/Python_MATLAB_Codes/train_inference_python
# ------------------------------- examples -------------------------------

python Train_ZSDeconvNet_3DSIM.py --psf_path 'C:/Users/Admin/Git/repo/attachments_to_repos/SIM/SIM_code/psf/PSF-3D-SIM.tif' \
--data_dir 'D:/ZS_DeconvNet/R2R/TestSIM/' \
                                --folder 'Ensconsin_augmented' --augment_flag 0 \
--test_images_path 'D:/ZS_DeconvNet/R2R/TestSIM/Ensconsin/00000001.tif' \

