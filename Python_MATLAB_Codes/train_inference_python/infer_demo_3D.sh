
# ------------------------------- inference arguments -------------------------------

# --bs: batch size, a tuple of length n for n input images. e.g. 3 7 for 2 input images
# --num_seg_window_x: number of segmented patch along x axis for each input image, a tuple of length n for n input images. e.g. 3 7 for 2 input images
# --num_seg_window_y: number of segmented patch along y axis for each input image, a tuple of length n for n input images
# --num_seg_window_z: number of segmented patch along z axis for each input image, a tuple of length n for n input images
# --overlap_x: overlapping in x direction for each input image, a tuple of length n for n input images. NOTICE: if num_seg_window>1, overlap should be big enough to avoid segmentation gap
# --overlap_y: overlapping in y direction for each input image, a tuple of length n for n input images.
# --overlap_z: overlapping in z direction for each input image, a tuple of length n for n input images.

# --input_dir: the root path of test data
# --load_weights_path: root path where models weights are loaded
# --background: the background value you want to extract in input images

# --Fourier_damping_flag: if performing Fourier damping to remove fixed pattern noise, usu. set to 1 for LLS and 0 for other modalities
# --Fourier_damping_length: the length of the upper and lower bar in Fourier damping mask
# --Fourier_damping_width: 2*Fourier_damping_width+1 will be the actual Fourier damping mask width 

# --insert_z: padded blank margin in each side of axial direction (how many slices)
# --model: "twostage_RCAN3D" or "twostage_Unet3D"
# --insert_xy: padded blank margin in pixels in each side 
# --upsample_flag: 0 or 1, whether the network upsamples the image

cd C:/Users/Admin/Git/repo/ZS-DeconvNet/Python_MATLAB_Codes/train_inference_python
# ------------------------------- examples1 3D LLS -------------------------------

python Infer_3D.py --num_seg_window_z 2 --num_seg_window_x 2 --num_seg_window_y 1 \
                   --background 100 --input_dir '../saved_models/LLS3D_Mitochondria/test_data/NoisyWF.tif'  \
                   --load_weights_path '../saved_models/LLS3D_Mitochondria/saved_model/weights_10000.h5'

# # ------------------------------- examples2 3D Confocal -------------------------------

python Infer_3D.py --num_seg_window_z 2 --num_seg_window_x 2 --num_seg_window_y 2 \
                   --background 0 --input_dir '../saved_models/confocal3D_ActinRing/test_data/*' \
                   --load_weights_path '../saved_models/confocal3D_ActinRing/saved_model/weights_10000.h5' \
                   --Fourier_damping_flag 0 --upsample_flag 1

python Infer_3D.py --num_seg_window_z 2 --num_seg_window_x 2 --num_seg_window_y 2 \
                   --background 0 --input_dir '../saved_models/confocal3D_Microtubule/test_data/*' \
                   --load_weights_path '../saved_models/confocal3D_Microtubule/saved_model/weights_10000.h5' \
                   --Fourier_damping_flag 0 --upsample_flag 1

# ------------------------------- examples3 3D LLS-SIM -------------------------------
python Infer_3D.py --input_dir '../saved_models/LLS-SIM3D_F-actin/test_data/*.tif' \
                   --load_weights_path '../saved_models/LLS-SIM3D_F-actin/saved_model/Lifeact_model.h5' \
                    --num_seg_window_x 3 --num_seg_window_z 3 --num_seg_window_y 3 --Fourier_damping_flag 0