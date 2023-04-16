import argparse
from models import twostage_Unet
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
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0)
tf.compat.v1.Session(config=tf.compat.v1.compat.v1.ConfigProto(gpu_options=gpu_options))

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=[1], nargs='*')
parser.add_argument("--num_seg_window_x", type=int, default=[1], nargs='*')
parser.add_argument("--overlap_x", type=int, default=[20], nargs='*')
parser.add_argument("--num_seg_window_y", type=int, default=[1], nargs='*')
parser.add_argument("--overlap_y", type=int, default=[20], nargs='*')
parser.add_argument("--input_dir", type=str, default='../saved_models/WF2D_Lysosome/test_data/NoisyInput.tif')
parser.add_argument("--load_weights_path", type=str, default='../saved_models/WF2D_Lysosome/saved_model/weights_20000.h5')
parser.add_argument("--insert_xy", type=int, default=16)
parser.add_argument("--upsample_flag", type=int, default=1)

args = parser.parse_args()

bs = args.bs
num_seg_window_x = args.num_seg_window_x
overlap_x = args.overlap_x
num_seg_window_y = args.num_seg_window_y
overlap_y = args.overlap_y
input_dir = args.input_dir
load_weights_path = args.load_weights_path
insert_xy = args.insert_xy
upsample_flag = args.upsample_flag

sep_ind = load_weights_path.rfind('/')
save_path = load_weights_path[0:sep_ind] +'/Inference/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
    
# --------------------------------------------------------------------------------
#                           select models and optimizer
# --------------------------------------------------------------------------------
modelFN_generator = twostage_Unet.Unet
optimizer_g = optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, decay=1e-5)
    
# --------------------------------------------------------------------------------
#                      predict
# --------------------------------------------------------------------------------
path = glob.glob(input_dir)
num_tif = len(path)
seg_window_x = [0 for _ in range(num_tif)] #corresponds to y in tif, or height
seg_window_y = [0 for _ in range(num_tif)]

for tif_ind in range(num_tif):
    
    image_name = 'img'+str(tif_ind)
    if 'tif' in path[tif_ind]:
        image = tiff.imread(path[tif_ind]).astype('float')
    else:
        header,image = read_mrc(path[tif_ind])
        image = image.astype('float')
        image = image.transpose((1,0))
    image[image<0] = 0
    image = prctile_norm(image)
        
    inp_x,inp_y = image.shape
    seg_window_x[tif_ind]=math.ceil((inp_x+(num_seg_window_x[tif_ind]-1)*overlap_x[tif_ind])/num_seg_window_x[tif_ind])
    seg_window_y[tif_ind]=math.ceil((inp_y+(num_seg_window_y[tif_ind]-1)*overlap_y[tif_ind])/num_seg_window_y[tif_ind])
    conv_block_num = 4
    n = math.ceil(seg_window_x[tif_ind]/2**conv_block_num)
    while 16*n-seg_window_x[tif_ind]<2*insert_xy:
        n = n+1
    insert_x = int((16*n-seg_window_x[tif_ind])/2)
    m = math.ceil(seg_window_y[tif_ind]/2**conv_block_num)
    while 16*m-seg_window_y[tif_ind]<2*insert_xy:
        m = m+1
    insert_y = int((16*m-seg_window_y[tif_ind])/2)
   
    # segment
    segmented_inp = []

    rr_list = list(range(0,inp_x-seg_window_x[tif_ind]+1,seg_window_x[tif_ind]-overlap_x[tif_ind]))
    if rr_list[-1] != inp_x-seg_window_x[tif_ind]:
        rr_list.append(inp_x-seg_window_x[tif_ind])

    cc_list = list(range(0,inp_y-seg_window_y[tif_ind]+1,seg_window_y[tif_ind]-overlap_y[tif_ind]))
    if cc_list[-1] != inp_y-seg_window_y[tif_ind]:
        cc_list.append(inp_y-seg_window_y[tif_ind])

    print('segmenting...')
    for rr in rr_list:
        for cc in cc_list:
            segmented_inp.append(image[rr:rr+seg_window_x[tif_ind],cc:cc+seg_window_y[tif_ind]])
    segmented_inp = np.array(segmented_inp).astype(np.float32)
    segmented_inp = segmented_inp[...,np.newaxis]
    seg_num = segmented_inp.shape[0]
    insert_shape_x = np.zeros([seg_num,insert_x,seg_window_y[tif_ind],1]).astype(np.float32)
    insert_shape_y = np.zeros([seg_num,seg_window_x[tif_ind]+2*insert_x,insert_y,1]).astype(np.float32)
    segmented_inp = np.concatenate((insert_shape_x,segmented_inp,insert_shape_x),axis=1)
    segmented_inp = np.concatenate((insert_shape_y,segmented_inp,insert_shape_y),axis=2)

    cur_bs = bs[tif_ind]
    bs_list = list(range(0,seg_num-cur_bs+1,cur_bs))
    if bs_list[-1] != seg_num-cur_bs:
        last_bs = seg_num-bs_list[-1]
    else:
        last_bs = cur_bs
    
    # predict
    p = modelFN_generator([seg_window_x[tif_ind]+2*insert_x,seg_window_y[tif_ind]+2*insert_y,1], upsample_flag=upsample_flag, insert_x=insert_x, insert_y=insert_y)
    p.compile(loss=None, optimizer=optimizer_g)
    p.load_weights(load_weights_path)
        
    dec_list = np.zeros([seg_num,seg_window_x[tif_ind]*(1+upsample_flag),seg_window_y[tif_ind]*(1+upsample_flag)], dtype=np.float32)
    den_list = np.zeros([seg_num,seg_window_x[tif_ind],seg_window_y[tif_ind]], dtype=np.float32)
    for batch_ind_start in bs_list:
        if batch_ind_start == bs_list[-1]:
            print('predicting patches %03d'%(batch_ind_start+1)+'-%03d'%(batch_ind_start+last_bs)+' out of '+'%03d'%(seg_num))
            pred = p.predict(segmented_inp[batch_ind_start:batch_ind_start+last_bs,...])
        else:
            print('predicting patches %03d'%(batch_ind_start+1)+'-%03d'%(batch_ind_start+cur_bs)+' out of '+'%03d'%(seg_num))
            pred = p.predict(segmented_inp[batch_ind_start:batch_ind_start+cur_bs,...])
        pred1 = np.squeeze(pred[0],axis=3).astype(np.float32)
        den_list[batch_ind_start:batch_ind_start+pred1.shape[0],...]=pred1
        pred2 = np.squeeze(pred[1],axis=3).astype(np.float32)
        pred2 = pred2[:,insert_x*(1+upsample_flag):(seg_window_x[tif_ind]+insert_x)*(1+upsample_flag),
                      insert_y*(1+upsample_flag):(seg_window_y[tif_ind]+insert_y)*(1+upsample_flag)]
        dec_list[batch_ind_start:batch_ind_start+pred1.shape[0],...]=pred2

    # fuse
    output_dec = np.zeros((inp_x*(1+upsample_flag),inp_y*(1+upsample_flag)), dtype=np.float32)
    output_den = np.zeros((inp_x,inp_y), dtype=np.float32)
    print('fusing...')
    for r_ind,rr in enumerate(rr_list):
        for c_ind,cc in enumerate(cc_list):    
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

            cur_patch = dec_list[r_ind*len(cc_list)+c_ind,rr_min_patch*(1+upsample_flag):rr_max_patch*(1+upsample_flag),cc_min_patch*(1+upsample_flag):cc_max_patch*(1+upsample_flag)].astype(np.float32)
            output_dec[rr_min*(1+upsample_flag):rr_max*(1+upsample_flag),cc_min*(1+upsample_flag):cc_max*(1+upsample_flag)] = cur_patch
            cur_patch = den_list[r_ind*len(cc_list)+c_ind,rr_min_patch:rr_max_patch,cc_min_patch:cc_max_patch].astype(np.float32)
            output_den[rr_min:rr_max,cc_min:cc_max] = cur_patch

    output_dec = np.uint16(1e4 * prctile_norm(output_dec,3,100))
    output_den = np.uint16(1e4 * prctile_norm(output_den,3,100))
    tiff.imwrite(save_path + image_name+'_denoised.tif', output_den, dtype='uint16')
    tiff.imwrite(save_path + image_name+'_deconved.tif', output_dec, dtype='uint16')
