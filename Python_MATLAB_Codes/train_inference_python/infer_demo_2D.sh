
# ------------------------------- inference arguments -------------------------------

# --bs: batch size, a tuple of length n for n input images. e.g. 3 7 for 2 input images
# --num_seg_window_x: number of segmented patch along x axis for each input image, a tuple of length n for n input images. e.g. 3 7 for 2 input images
# --num_seg_window_y: number of segmented patch along y axis for each input image, a tuple of length n for n input images
# --overlap_x: overlapping in x direction for each input image, a tuple of length n for n input images. NOTICE: if num_seg_window>1, overlap should be big enough to avoid segmentation gap
# --overlap_y: overlapping in y direction for each input image, a tuple of length n for n input images.
# --predict_iter: loading weights iterations
# --input_dir: the path of test data
# --load_weights_path: root directory where models weights are loaded
# --insert_xy: padded blank margin in pixels. The applied blank margin may be larger than your given value due to size requirement of network. 
# --upsample_flag: 0 or 1, whether the network upsamples the image

# ------------------------------- examples -------------------------------

cd C:/Users/Admin/Git/repo/ZS-DeconvNet-master/train_inference_python
# WF lysosome
python Infer_2D.py --input_dir '../saved_models/WF2D_Lysosome/test_data/NoisyInput.tif' \
                   --load_weights_path '../saved_models/WF2D_Lysosome/saved_model/weights_20000.h5' \
                    --num_seg_window_x 1

# # SIM CCP
python Infer_2D.py --input_dir '../saved_models/SIM2D_CCP/test_data/*.tif' \
                   --load_weights_path '../saved_models/SIM2D_CCP/saved_model/weights_50000.h5' --insert_xy 0

# SIM MT
python Infer_2D.py --input_dir '../saved_models/SIM2D_Microtubule/test_data/*.tif' \
                   --load_weights_path '../saved_models/SIM2D_Microtubule/saved_model/Microtubules_50000.h5' --insert_xy 0

