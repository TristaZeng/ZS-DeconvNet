import argparse
from models import twostage_RCAN3D, twostage_Unet3D
from tensorflow.keras import optimizers
import glob
import numpy as np
from utils.utils import prctile_norm
from utils.loss import read_mrc
import tifffile as tiff
import os
import tensorflow as tf
import math

os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.95)
tf.compat.v1.Session(config=tf.compat.v1.compat.v1.ConfigProto(gpu_options=gpu_options))

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=tuple, default=[1], nargs='*')
parser.add_argument("--num_seg_window_z", type=tuple, default=[4], nargs='*')
parser.add_argument("--overlap_z", type=tuple, default=[4], nargs='*')
parser.add_argument("--num_seg_window_x", type=tuple, default=[4], nargs='*')
parser.add_argument("--overlap_x", type=tuple, default=[20], nargs='*')
parser.add_argument("--num_seg_window_y", type=tuple, default=[4], nargs='*')
parser.add_argument("--overlap_y", type=tuple, default=[20], nargs='*')

parser.add_argument("--input_dir", type=str, default='../saved_models/LLS3D_Mitochondria/test_data/NoisyWF.tif')
parser.add_argument("--load_weights_path", type=str, default='../saved_models/LLS3D_Mitochondria/saved_model/weights_10000.h5')
parser.add_argument("--background", type=int, default=100)

parser.add_argument("--Fourier_damping_flag", type=int, default=1)
parser.add_argument("--Fourier_damping_length", type=int, default=450)
parser.add_argument("--Fourier_damping_width", type=int, default=1)

parser.add_argument("--model", type=str, default="twostage_RCAN3D")
parser.add_argument("--insert_z", type=int, default=2)
parser.add_argument("--insert_xy", type=int, default=8)
parser.add_argument("--upsample_flag", type=int, default=0)

args = parser.parse_args()

bs = args.bs
num_seg_window_z = args.num_seg_window_z
overlap_z = args.overlap_z
num_seg_window_x = args.num_seg_window_x
overlap_x = args.overlap_x
num_seg_window_y = args.num_seg_window_y
overlap_y = args.overlap_y

input_dir = args.input_dir
load_weights_path = args.load_weights_path
background = args.background

model = args.model
insert_z = args.insert_z
insert_xy = args.insert_xy
upsample_flag = args.upsample_flag

sep_ind = load_weights_path.rfind('/')
save_path = load_weights_path[0:sep_ind] +'/Inference/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
    
# --------------------------------------------------------------------------------
#                           select models and optimizer
# --------------------------------------------------------------------------------
modelFns = {'twostage_RCAN3D': twostage_RCAN3D.RCAN3D,
            'twostage_Unet3D': twostage_Unet3D.Unet,
            'twostage_RCAN3D_SIM':twostage_RCAN3D.RCAN3D_SIM,
            'twostage_RCAN3D_SIM_compact':twostage_RCAN3D.RCAN3D_SIM_compact,
            'twostage_RCAN3D_SIM_compact2':twostage_RCAN3D.RCAN3D_SIM_compact2}
modelFN = modelFns[model]
optimizer_g = optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, decay=1e-5)

# --------------------------------------------------------------------------------
#                                  predict
# --------------------------------------------------------------------------------

file_path = glob.glob(input_dir)
num_tif = len(file_path)
seg_window_z = [0 for _ in range(num_tif)]
seg_window_x = [0 for _ in range(num_tif)]
seg_window_y = [0 for _ in range(num_tif)]

for tif_ind in range(num_tif):

    # load input stack
    if 'tif' in file_path[tif_ind]:
        image = tiff.imread(file_path[tif_ind]).astype('float')
        image = np.flip(image,axis=1)
    else:
        header,image = read_mrc(file_path[tif_ind])
        image = image.astype('float')
        image = image.transpose((2,1,0))
    image = image-background
    image [image<0] = 0
    image = prctile_norm(image)
    image_name = '%02d'%tif_ind
    inp_z,inp_x,inp_y = image.shape
    seg_window_z[tif_ind]=math.ceil((inp_z+(num_seg_window_z[tif_ind]-1)*overlap_z[tif_ind])/num_seg_window_z[tif_ind])
    seg_window_x[tif_ind]=math.ceil((inp_x+(num_seg_window_x[tif_ind]-1)*overlap_x[tif_ind])/num_seg_window_x[tif_ind])
    seg_window_y[tif_ind]=math.ceil((inp_y+(num_seg_window_y[tif_ind]-1)*overlap_y[tif_ind])/num_seg_window_y[tif_ind])

    p = modelFN([seg_window_x[tif_ind] + 2 * insert_xy, seg_window_y[tif_ind] + 2 * insert_xy, seg_window_z[tif_ind] + 2 * insert_z, 1],
                upsample_flag=upsample_flag, insert_z=insert_z, insert_xy=insert_xy)
    p.compile(loss=None, optimizer=optimizer_g)
    
    # segment
    segmented_inp = []

    zz_list = list(range(0,inp_z-seg_window_z[tif_ind]+1,seg_window_z[tif_ind]-overlap_z[tif_ind]))
    if zz_list[-1] != inp_z-seg_window_z[tif_ind]:
        zz_list.append(inp_z-seg_window_z[tif_ind])

    rr_list = list(range(0,inp_x-seg_window_x[tif_ind]+1,seg_window_x[tif_ind]-overlap_x[tif_ind]))
    if rr_list[-1] != inp_x-seg_window_x[tif_ind]:
        rr_list.append(inp_x-seg_window_x[tif_ind])

    cc_list = list(range(0,inp_y-seg_window_y[tif_ind]+1,seg_window_y[tif_ind]-overlap_y[tif_ind]))
    if cc_list[-1] != inp_y-seg_window_y[tif_ind]:
        cc_list.append(inp_y-seg_window_y[tif_ind])

    print('segmenting...')
    for zz in zz_list:
        for rr in rr_list:
            for cc in cc_list:
                segmented_inp.append(image[zz:zz+seg_window_z[tif_ind],rr:rr+seg_window_x[tif_ind],cc:cc+seg_window_y[tif_ind]])
    segmented_inp = np.array(segmented_inp).astype(np.float32)
    segmented_inp = np.transpose(segmented_inp, (0,2,3,1))
    segmented_inp = segmented_inp[...,np.newaxis]
    seg_num = segmented_inp.shape[0]
    insert_shape_z = np.zeros([seg_num,seg_window_x[tif_ind],seg_window_y[tif_ind],insert_z,1]).astype(np.float32)
    insert_shape_x = np.zeros([seg_num,insert_xy,seg_window_y[tif_ind],seg_window_z[tif_ind]+2*insert_z,1]).astype(np.float32)
    insert_shape_y = np.zeros([seg_num,seg_window_x[tif_ind]+2*insert_xy,insert_xy,seg_window_z[tif_ind]+2*insert_z,1]).astype(np.float32)
    segmented_inp = np.concatenate((insert_shape_z,segmented_inp,insert_shape_z),axis=3)
    segmented_inp = np.concatenate((insert_shape_x,segmented_inp,insert_shape_x),axis=1)
    segmented_inp = np.concatenate((insert_shape_y,segmented_inp,insert_shape_y),axis=2)

    cur_bs = bs[tif_ind]
    bs_list = list(range(0,seg_num-cur_bs+1,cur_bs))
    if bs_list[-1] != seg_num-cur_bs:
        last_bs = seg_num-bs_list[-1]
    else:
        last_bs = cur_bs
    
    # predict
    p.load_weights(load_weights_path)
        
    dec_list = np.zeros([seg_num,seg_window_z[tif_ind],seg_window_x[tif_ind]*(1+upsample_flag),seg_window_y[tif_ind]*(1+upsample_flag)], dtype=np.float32)
    den_list = np.zeros([seg_num,seg_window_z[tif_ind],seg_window_x[tif_ind],seg_window_y[tif_ind]], dtype=np.float32)
    for batch_ind_start in bs_list:
        if batch_ind_start == bs_list[-1]:
            print('predicting patches %03d'%(batch_ind_start+1)+'-%03d'%(batch_ind_start+last_bs)+' out of '+'%03d'%(seg_num))
            pred = p.predict(segmented_inp[batch_ind_start:batch_ind_start+last_bs,...])
        else:
            print('predicting patches %03d'%(batch_ind_start+1)+'-%03d'%(batch_ind_start+cur_bs)+' out of '+'%03d'%(seg_num))
            pred = p.predict(segmented_inp[batch_ind_start:batch_ind_start+cur_bs,...])
        pred1 = np.squeeze(pred[0],axis=4).astype(np.float32)
        pred1 = np.transpose(pred1,(0,3,1,2))
        den_list[batch_ind_start:batch_ind_start+pred1.shape[0],...]=pred1
        pred2 = np.squeeze(pred[1],axis=4).astype(np.float32)
        pred2 = pred2[:,insert_xy*(1+upsample_flag):(seg_window_x[tif_ind]+insert_xy)*(1+upsample_flag),
                      insert_xy*(1+upsample_flag):(seg_window_y[tif_ind]+insert_xy)*(1+upsample_flag),
                      insert_z:seg_window_z[tif_ind]+insert_z]
        pred2 = np.transpose(pred2,(0,3,1,2))
        dec_list[batch_ind_start:batch_ind_start+pred1.shape[0],...]=pred2

    # fuse
    output_dec = np.zeros((inp_z,inp_x*(1+upsample_flag),inp_y*(1+upsample_flag)), dtype=np.float32)
    output_den = np.zeros((inp_z,inp_x,inp_y), dtype=np.float32)
    print('fusing...')
    for z_ind,zz in enumerate(zz_list):
        for r_ind,rr in enumerate(rr_list):
            for c_ind,cc in enumerate(cc_list):
                if zz == 0:
                    zz_min = 0
                    zz_min_patch = 0
                else:
                    zz_min = zz + math.ceil(overlap_z[tif_ind]/2)
                    zz_min_patch = math.ceil(overlap_z[tif_ind]/2)
                if zz + seg_window_z[tif_ind] == inp_z:
                    zz_max = inp_z
                    zz_max_patch = seg_window_z[tif_ind]
                else:
                    zz_max = zz + seg_window_z[tif_ind] - math.floor(overlap_z[tif_ind]/2)
                    zz_max_patch = seg_window_z[tif_ind] - math.floor(overlap_z[tif_ind]/2)

                if rr == 0:
                    rr_min = 0
                    rr_min_patch = 0
                else:
                    rr_min = rr + math.ceil(overlap_x[tif_ind]/2)
                    rr_min_patch = math.ceil(overlap_x[tif_ind]/2)
                if rr + seg_window_x[tif_ind] == inp_x:
                    rr_max = inp_x
                    rr_max_patch = seg_window_x[tif_ind]
                else:
                    rr_max = rr + seg_window_x[tif_ind] - math.floor(overlap_x[tif_ind]/2)
                    rr_max_patch = seg_window_x[tif_ind] - math.floor(overlap_x[tif_ind]/2)

                if cc == 0:
                    cc_min = 0
                    cc_min_patch = 0
                else:
                    cc_min = cc + math.ceil(overlap_y[tif_ind]/2)
                    cc_min_patch = math.ceil(overlap_y[tif_ind]/2)
                if cc + seg_window_y[tif_ind] == inp_y:
                    cc_max = inp_y
                    cc_max_patch = seg_window_y[tif_ind]
                else:
                    cc_max = cc + seg_window_y[tif_ind] - math.floor(overlap_y[tif_ind]/2)
                    cc_max_patch = seg_window_y[tif_ind] - math.floor(overlap_y[tif_ind]/2)

                cur_patch = dec_list[z_ind*len(rr_list)*len(cc_list)+r_ind*len(cc_list)+c_ind,zz_min_patch:zz_max_patch,rr_min_patch*(1+upsample_flag):rr_max_patch*(1+upsample_flag),cc_min_patch*(1+upsample_flag):cc_max_patch*(1+upsample_flag)].astype(np.float32)
                output_dec[zz_min:zz_max,rr_min*(1+upsample_flag):rr_max*(1+upsample_flag),cc_min*(1+upsample_flag):cc_max*(1+upsample_flag)] = cur_patch
                cur_patch = den_list[z_ind*len(rr_list)*len(cc_list)+r_ind*len(cc_list)+c_ind,zz_min_patch:zz_max_patch,rr_min_patch:rr_max_patch,cc_min_patch:cc_max_patch].astype(np.float32)
                output_den[zz_min:zz_max,rr_min:rr_max,cc_min:cc_max] = cur_patch

    # post-process
    if args.Fourier_damping_flag:
        output_dec = np.fft.fftshift(np.fft.fft2(output_dec))
        half_y = output_dec.shape[2]//2
        output_dec[:,0:args.Fourier_damping_length,half_y-args.Fourier_damping_width:half_y+args.Fourier_damping_width+1] = 0
        output_dec[:,-args.Fourier_damping_length:,half_y-args.Fourier_damping_width:half_y+args.Fourier_damping_width+1] = 0  
        output_dec = np.real(np.fft.ifft2(np.fft.ifftshift(output_dec)))
        
    # save
    output_dec = np.uint16(1e4 * prctile_norm(output_dec,3,100))
    output_den = np.uint16(1e4 * prctile_norm(output_den,3,100))
    tiff.imwrite(save_path + image_name+'_dec.tif', np.flip(output_dec,axis=1), dtype='uint16')
    tiff.imwrite(save_path + image_name+'_den.tif', np.flip(output_den,axis=1), dtype='uint16')
  
    del segmented_inp,dec_list,den_list,output_dec,output_den