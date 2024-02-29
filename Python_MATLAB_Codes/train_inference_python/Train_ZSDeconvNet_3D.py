import argparse
from models import twostage_RCAN3D, twostage_Unet3D
from tensorflow.keras import optimizers
import glob
import numpy as np
import datetime
from utils.utils import prctile_norm
from utils.data_loader import DataLoader
import tifffile as tiff
from scipy.interpolate import interp1d
import os
import cv2
import tensorflow as tf
import math
from utils.loss import create_NBR2NBR_loss, psf_estimator_3d, create_psf_loss_3D_NBR2NBR
import imageio
from tensorflow.keras import backend as K

parser = argparse.ArgumentParser()
# models
parser.add_argument("--model", type=str, default="twostage_RCAN3D")
parser.add_argument("--upsample_flag", type=int, default=1)

# training settings
parser.add_argument("--load_all_data", type=int, default=1) # load all training set into memory before training for faster computation
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--gpu_memory_fraction", type=float, default=0.8)
parser.add_argument("--mixed_precision_training", type=int, default=1)
parser.add_argument("--load_init_model_iter", type=int, default=0)
parser.add_argument("--iterations", type=int, default=10000)
parser.add_argument("--valid_interval", type=int, default=1000)
parser.add_argument("--test_interval", type=int, default=1000)

# lr decay
parser.add_argument("--start_lr", type=float, default=1e-4)
parser.add_argument("--lr_decay_factor", type=float, default=0.5)

# about data
parser.add_argument("--save_weights_dir", type=str, default="../your_saved_models/")
parser.add_argument("--save_weights_suffix", type=str, default="_Hess0.1_MAE_up")
parser.add_argument("--psf_path", type=str)
parser.add_argument("--data_dir", type=str) 
parser.add_argument("--test_images_path", type=str)
parser.add_argument("--folder", type=str) 
parser.add_argument("--background", type=int, default=100) # set to the value you want to extract in test image

parser.add_argument("--input_y", type=int, default=64)
parser.add_argument("--input_x", type=int, default=64)
parser.add_argument("--input_z", type=int, default=13)
parser.add_argument("--insert_z", type=int, default=2)
parser.add_argument("--insert_xy", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--dx", type=float, default=0.0926)
parser.add_argument("--dz", type=float, default=0.3704)
parser.add_argument("--dxpsf", type=float, default=0.0926)
parser.add_argument("--dzpsf", type=float, default=0.05)
parser.add_argument("--norm_flag", type=int, default=0)

# about loss functions
parser.add_argument("--mse_flag", type=int, default=0)
parser.add_argument("--TV_weight", type=float, default=0)
parser.add_argument("--Hess_weight", type=float, default=0.1)

args = parser.parse_args()

data_dir = args.data_dir
folder = args.folder
psf_path = args.psf_path
save_weights_path = data_dir+folder 
save_weights_suffix = args.save_weights_suffix

model = args.model
upsample_flag = args.upsample_flag
save_weights_dir = args.save_weights_dir

load_all_data = args.load_all_data
gpu_id = str(args.gpu_id)
gpu_memory_fraction = args.gpu_memory_fraction
mixed_precision_training = str(args.mixed_precision_training)
load_init_model_iter = args.load_init_model_iter
iterations = args.iterations
valid_interval = args.valid_interval
test_interval = args.test_interval

background = args.background
input_y = args.input_y
input_x = args.input_x
input_z = args.input_z
insert_z = args.insert_z
insert_xy = args.insert_xy
dz = args.dz
dx = args.dx
dxpsf = args.dxpsf
dzpsf = args.dzpsf
batch_size = args.batch_size
norm_flag = args.norm_flag

mse_flag = args.mse_flag
TV_weight = args.TV_weight
Hess_weight = args.Hess_weight

lr_decay_factor = args.lr_decay_factor
start_lr = args.start_lr

os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = mixed_precision_training
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
       
# define and make paths
save_weights_path = save_weights_dir + "/" + folder + "_" + model + save_weights_suffix
train_images_path = data_dir + folder + "/input" 
train_gt_path = data_dir + folder + "/gt" 
test_images_path = args.test_images_path
sample_path = save_weights_path + "/TestSampled/"
valid_path = save_weights_path + "/TrainSampled/"

if not os.path.exists(save_weights_dir):
    os.mkdir(save_weights_dir)
if not os.path.exists(save_weights_path):
    os.mkdir(save_weights_path)
if not os.path.exists(sample_path):
    os.mkdir(sample_path)
if not os.path.exists(valid_path):
    os.mkdir(valid_path)

with open(save_weights_path+"/config.txt","w") as f:
    f.write(str(args))

# determine test input
path = glob.glob(test_images_path)
image_batch = []
for curp in path:
    image = tiff.imread(curp).astype('float')
    image = image - background
    image[image < 0] = 0
    image_batch.append(image)
image = prctile_norm(np.array(image_batch))
input_test = np.transpose(image, (0,2,3,1))
bs,h,w,d = input_test.shape
for vol in range(bs):
    input_testtosave = np.transpose(1e4 * prctile_norm(input_test[vol,...]), (2, 0, 1)).astype('uint16')
    tiff.imwrite(sample_path + 'input' + str(vol) + '.tif', input_testtosave, dtype='uint16')
insert_shape = np.zeros([len(path),input_test.shape[1],input_test.shape[2],insert_z])
input_test = np.concatenate((insert_shape,input_test,insert_shape),axis=3)
insert_shape = np.zeros([len(path),insert_xy,input_test.shape[2],input_test.shape[3]])
input_test = np.concatenate((insert_shape,input_test,insert_shape),axis=1)
insert_shape = np.zeros([len(path),input_test.shape[1],insert_xy,input_test.shape[3]])
input_test = np.concatenate((insert_shape,input_test,insert_shape),axis=2)
bs,h,w,d = input_test.shape
    
# --------------------------------------------------------------------------------
#                           select models and optimizer
# --------------------------------------------------------------------------------
modelFns = {'twostage_RCAN3D':twostage_RCAN3D.RCAN3D,
            'twostage_Unet3D':twostage_Unet3D.Unet}
modelFN = modelFns[model]
my_optimizer = optimizers.Adam(learning_rate=start_lr, beta_1=0.9, beta_2=0.999, decay=1e-5)
 
# --------------------------------------------------------------------------------
#                          calculate and process otf & psf
# --------------------------------------------------------------------------------
psf_g = np.float32(imageio.mimread(psf_path))
psf_g = np.transpose(psf_g,[1,2,0])
psf_width,psf_height,psf_depth = psf_g.shape
half_psf_depth = math.floor(psf_depth/2)

if psf_depth%2==0:
    raise ValueError('The depth of PSF should be an odd number.')

#get the desired dz by interpolation
z = np.arange((half_psf_depth+1) * dzpsf, (psf_depth+0.1) * dzpsf, dzpsf)
zi = np.arange((half_psf_depth+1) * dzpsf, (psf_depth+0.1) * dzpsf, dz)
if zi[-1]>z[-1]:
    zi = zi[0:-1]
PSF1 = np.zeros((psf_width,psf_height,len(zi)))
for i in range(psf_width):
    for j in range(psf_height):
        curCol = psf_g[i,j,half_psf_depth:psf_depth]
        interp = interp1d(z, curCol, 'slinear')
        PSF1[i,j,:] = interp(zi)
z2 = np.zeros((half_psf_depth))
zi2 = np.zeros((len(zi)-1))
for n in range(half_psf_depth):
    z2[half_psf_depth-n-1]=z[0]-dzpsf*(n+1)
for n in range(zi2.shape[0]):
    zi2[len(zi)-1-n-1]=zi[0]-dz*(n+1)
PSF2 = np.zeros((psf_width,psf_height,len(zi2)))
for i in range(psf_width):
    for j in range(psf_height):
        curCol = psf_g[i,j,0:half_psf_depth]
        interp = interp1d(z2, curCol, 'slinear')
        PSF2[i,j,:] = interp(zi2)
psf_g = np.concatenate((PSF2,PSF1),axis=2)
psf_g = psf_g/np.sum(psf_g)
psf_width,psf_height,psf_depth = psf_g.shape
half_psf_width = psf_width//2

#get the desired dxy
if psf_width%2==1:
    sr_ratio = dxpsf/dx
    sr_x = round(psf_width*sr_ratio)
    if sr_x%2==0:
        if sr_x>psf_width*sr_ratio:
            sr_x = sr_x - 1
        else:
            sr_x = sr_x + 1
    sr_y = round(psf_height*sr_ratio)
    if sr_y%2==0:
        if sr_y>psf_height*sr_ratio:
            sr_y = sr_y - 1
        else:
            sr_y = sr_y + 1
    psf_tmp = psf_g
    psf_g = np.zeros([sr_x,sr_y,psf_depth])
    for z in range(psf_g.shape[2]):
        psf_g[:,:,z] = cv2.resize(psf_tmp[:,:,z],(sr_x,sr_y))
else:
    x = np.arange((half_psf_width+1) * dxpsf, (psf_width+0.1) * dxpsf, dxpsf)
    xi = np.arange((half_psf_width+1) * dxpsf, (psf_width+0.1) * dxpsf, dx)
    if xi[-1]>x[-1]:
        xi = xi[0:-1]
    PSF1 = np.zeros((len(xi),psf_height,psf_depth))
    for i in range(psf_height):
        for j in range(psf_depth):
            curCol = psf_g[half_psf_width:psf_width,i,j]
            interp = interp1d(x, curCol, 'slinear')
            PSF1[:,i,j] = interp(xi)
    x2 = np.zeros(len(x))
    xi2 = np.zeros(len(xi))
    for n in range(len(x)):
        x2[len(x)-n-1]=x[0]-dxpsf*n
    for n in range(len(xi)):
        xi2[len(xi)-n-1]=xi[0]-dx*n
    PSF2 = np.zeros((len(xi2),psf_height,psf_depth))
    for i in range(psf_height):
        for j in range(psf_depth):
            curCol = psf_g[1:half_psf_width+1+psf_width%2,i,j]
            interp = interp1d(x2, curCol, 'slinear')
            PSF2[:,i,j] = interp(xi2)
    psf_g = np.concatenate((PSF2[:-1,:,:],PSF1),axis=0)
    psf_g = psf_g/np.sum(psf_g)
    psf_width,psf_height,psf_depth = psf_g.shape
    half_psf_height = psf_height//2
    
    x = np.arange((half_psf_height+1) * dxpsf, (psf_height+0.1) * dxpsf, dxpsf)
    xi = np.arange((half_psf_height+1) * dxpsf, (psf_height+0.1) * dxpsf, dx)
    if xi[-1]>x[-1]:
        xi = xi[0:-1]
    PSF1 = np.zeros((psf_width,len(xi),psf_depth))
    for i in range(psf_width):
        for j in range(psf_depth):
            curCol = psf_g[i,half_psf_height:psf_height,j]
            interp = interp1d(x, curCol, 'slinear')
            PSF1[i,:,j] = interp(xi)
    x2 = np.zeros(len(x))
    xi2 = np.zeros(len(xi))
    for n in range(len(x2)):
        x2[len(x2)-n-1]=x[0]-dxpsf*n
    for n in range(len(xi2)):
        xi2[len(xi2)-n-1]=xi[0]-dx*n
    PSF2 = np.zeros((psf_width,len(xi2),psf_depth))
    for i in range(psf_width):
        for j in range(psf_depth):
            curCol = psf_g[i,1:half_psf_height+1+psf_height%2,j]
            interp = interp1d(x2, curCol, 'slinear')
            PSF2[i,:,j] = interp(xi2)
    psf_g = np.concatenate((PSF2[:,:-1,:],PSF1),axis=1)
psf_g = psf_g/np.sum(psf_g)
psf_width,psf_height,psf_depth = psf_g.shape
        
# get OTF
psf = np.zeros([psf_g.shape[0],psf_g.shape[1],input_z])
if psf_depth<input_z:
    psf[:,:,input_z//2-psf_depth//2:input_z//2+psf_depth//2+1] = psf_g
else:
    psf = psf_g[:,:,psf_depth//2-input_z//2:psf_depth//2+input_z//2+1]
otf = np.fft.fftshift(np.fft.fftn(psf))
otf = np.abs(otf)
otf_g = np.zeros([input_x*(upsample_flag+1),input_y*(upsample_flag+1),input_z])
for z in range(otf.shape[2]):
    otf_g[:,:,z] = cv2.resize(otf[:,:,z],(otf_g.shape[0],otf_g.shape[1]))
otf_g = otf_g/np.sum(otf_g)

# crop PSF for faster computation
halfz = min(psf_depth//2,input_z-1)
psf_g = psf_g[:,:,psf_depth//2-halfz:psf_depth//2+halfz+1]
sigma_y, sigma_x, _ = psf_estimator_3d(psf_g)
ksize = int(sigma_y * 4)
halfx = psf_width // 2
halfy = psf_height // 2
if ksize<=halfx:
    psf_g = psf_g[halfx-ksize:halfx+ksize+1, halfy-ksize:halfy+ksize+1,:]
    psf_g = np.reshape(psf_g,(2*ksize+1,2*ksize+1,-1,1,1)).astype(np.float32)
else:
    psf_g = np.reshape(psf_g,(psf_width,psf_height,-1,1,1)).astype(np.float32)
psf_g = psf_g/np.sum(psf_g)

# save PSF and OTF for checking
psf_g_tosave = np.squeeze(psf_g)
otf_g_tosave = np.uint16(65535*prctile_norm(np.abs(otf_g)))
psf_g_tosave = np.transpose(psf_g_tosave,[2,0,1])
otf_g_tosave = np.transpose(otf_g_tosave,[2,0,1])
imageio.volwrite(save_weights_path+'/psf.tif',psf_g_tosave)
imageio.volwrite(save_weights_path+'/otf.tif',otf_g_tosave)

# --------------------------------------------------------------------------------
#                      compile model
# --------------------------------------------------------------------------------
g = modelFN((input_x+2*insert_xy, input_y+2*insert_xy, input_z+2*insert_z, 1), 
                      upsample_flag=upsample_flag,
                      insert_z=insert_z,insert_xy=insert_xy)
p = modelFN((h,w,d,1), upsample_flag=upsample_flag,
                      insert_z=insert_z,insert_xy=insert_xy)
g_copy = modelFN((input_x+2*insert_xy, input_y+2*insert_xy, input_z*2+2*insert_z, 1), 
                           upsample_flag=0,
                           insert_z=insert_z,insert_xy=insert_xy)
if os.path.exists(save_weights_path + '/weights_' + str(load_init_model_iter) + '.h5'):
    g.load_weights(save_weights_path + '/weights_' + str(load_init_model_iter) + '.h5')
    print('Load initial weights!')
else:
    load_init_model_iter = 0

NBR2NBR_loss = create_NBR2NBR_loss(0,mse_flag)
loss = create_psf_loss_3D_NBR2NBR(psf_g, mse_flag, batch_size, 
    upsample_flag,TV_weight=TV_weight,Hess_weight=Hess_weight,
    insert_z=insert_z,insert_xy=insert_xy)
g_loss = [NBR2NBR_loss,loss]

g.compile(loss=g_loss, optimizer=my_optimizer)
p.compile(loss=None, optimizer=my_optimizer)
g_copy.compile(loss=None, optimizer=my_optimizer)

# --------------------------------------------------------------------------------
#                              write in tensorboard
# --------------------------------------------------------------------------------
def write_log(writer, names, logs, batch_no):
    with writer.as_default():
        tf.summary.scalar(names, logs, step=batch_no)
        writer.flush()

log_path = save_weights_path + '/graph'
if os.path.exists(log_path):
    for file_name in os.listdir(log_path):
        path_file = os.path.join(log_path,file_name)
        if os.path.isfile(path_file):
           os.remove(path_file)
else:
    os.mkdir(log_path)

writer = tf.summary.create_file_writer(log_path)

# --------------------------------------------------------------------------------
#                                Sample function
# --------------------------------------------------------------------------------
# choose a few samples from training set to display during training
valid_num = 3
images_path = glob.glob(train_images_path + '/*')
print(train_images_path)
input_valid, gt_valid = DataLoader(images_path, train_images_path, train_gt_path, batch_size=valid_num, norm_flag=norm_flag)
input_valid = np.reshape(input_valid, (valid_num, 1, input_z, input_y, input_x), order='F')
gt_valid = np.reshape(gt_valid, (valid_num, 1, input_z, input_y, input_x), order='F')
input_valid = np.transpose(input_valid, (0, 3, 4, 2, 1))
gt_valid = np.transpose(gt_valid, (0, 3, 4, 2, 1))
for i in range(valid_num):
    cur_input = np.transpose(input_valid[i,:,:,:,0],[2,0,1])
    cur_input = np.uint16(1e4 * prctile_norm(cur_input))
    tiff.imwrite(valid_path + str(i) + 'input.tif', cur_input, dtype='uint16')
    
    cur_gt = np.transpose(gt_valid[i,:,:,:,0],[2,0,1])
    cur_gt = np.uint16(1e4 * prctile_norm(cur_gt))
    tiff.imwrite(valid_path + str(i) + 'gt.tif', cur_gt, dtype='uint16')
insert_shape = np.zeros([valid_num,input_y,input_x,insert_z,1])
input_valid = np.concatenate((insert_shape,input_valid,insert_shape),axis=3)
insert_shape = np.zeros([valid_num,insert_xy,input_x,input_z+2*insert_z,1])
input_valid = np.concatenate((insert_shape,input_valid,insert_shape),axis=1)
insert_shape = np.zeros([valid_num,input_y+2*insert_xy,insert_xy,input_z+2*insert_z,1])
input_valid = np.concatenate((insert_shape,input_valid,insert_shape),axis=2)

def Validate(it):
    
    output = g.predict(input_valid)
    output_fft = np.fft.fftshift(np.fft.fftn(np.squeeze(output[1])))
    if upsample_flag:
        output_fft = output_fft[:,insert_xy*2:output_fft.shape[1]-insert_xy*2,insert_xy*2:output_fft.shape[2]-insert_xy*2,insert_z:output_fft.shape[3]-insert_z]
    else:
        output_fft = output_fft[:,insert_xy:output_fft.shape[1]-insert_xy,insert_xy:output_fft.shape[2]-insert_xy,insert_z:output_fft.shape[3]-insert_z]
    output_mul_otf = np.real(np.fft.ifftn(np.fft.ifftshift(output_fft*np.expand_dims(otf_g,axis=0))))
    for i in range(valid_num):    
        imageio.volwrite(valid_path+str(i)+'out_mul_otf_iter'+'%05d'%it+'.tif', np.transpose(np.uint16(prctile_norm(output_mul_otf[i,:,:,:])*65535),[2,0,1]))   
        imageio.volwrite(valid_path+str(i)+'out_denoise_iter'+'%05d'%it+'.tif', np.transpose(np.uint16(prctile_norm(output[0][i,:,:,:,0])*65535),[2,0,1]))
        if upsample_flag:
            imageio.volwrite(valid_path+str(i)+'out_deconv_iter'+'%05d'%it+'.tif', np.transpose(np.uint16(prctile_norm(output[1][i,insert_xy*2:insert_xy*2+input_y*2,insert_xy*2:(insert_xy+input_x)*2,insert_z:(insert_z+input_z),0])*65535),[2,0,1]))
        else:
            imageio.volwrite(valid_path+str(i)+'out_deconv_iter'+'%05d'%it+'.tif', np.transpose(np.uint16(prctile_norm(output[1][i,insert_xy:insert_xy+input_y,insert_xy:insert_xy+input_x,insert_z:insert_z+input_z,0])*65535),[2,0,1]))

def test(it):
    _,h,w,d = input_test.shape
    p.load_weights(save_weights_path + '/weights_' + str(it) + '.h5')
    for vol in range(input_test.shape[0]):
        input_cur = input_test[vol,:,:,:]
        pred = p.predict(input_cur[np.newaxis,:,:,:,np.newaxis])
        pred1 = np.squeeze(pred[0])
        if upsample_flag:
            pred2 = np.squeeze(pred[1][:,insert_xy*2:(h-insert_xy)*2,insert_xy*2:(w-insert_xy)*2,insert_z:(d-insert_z),:])
        else:
            pred2 = np.squeeze(pred[1][:,insert_xy:(h-insert_xy),insert_xy:(w-insert_xy),insert_z:d-insert_z,:])
        pred1 = np.transpose(1e4 * prctile_norm(pred1), (2, 0, 1)).astype('uint16')
        tiff.imwrite(sample_path + str(vol)+'denoise_'+str(it) + '.tif', pred1, dtype='uint16')
        pred2 = np.transpose(1e4 * prctile_norm(pred2), (2, 0, 1)).astype('uint16')
        tiff.imwrite(sample_path + str(vol)+'deconv_'+str(it) + '.tif', pred2, dtype='uint16')
        
# --------------------------------------------------------------------------------
#                                  Training
# --------------------------------------------------------------------------------
start_time = datetime.datetime.now()
loss_denoise = []
loss_deconv = []
images_path = glob.glob(train_images_path + '/*')
if load_all_data:
    num_total = len(images_path)
    inputs, gts = DataLoader(images_path, train_images_path, train_gt_path, batch_size=num_total, norm_flag=norm_flag)
    inputs = np.reshape(inputs, (num_total, 1, input_z, input_y, input_x), order='F')
    gts = np.reshape(gts, (num_total, 1, input_z, input_y, input_x), order='F')

for it in range(iterations-load_init_model_iter):

    if load_all_data:
        index = np.random.choice(num_total,batch_size)
        input = inputs[index,:,:,:,:]
        gt = gts[index,:,:,:,:]
    else:
        input, gt = DataLoader(images_path, train_images_path, train_gt_path, batch_size=batch_size, norm_flag=norm_flag)
        images_path = glob.glob(train_images_path + '/*')
        input = np.reshape(input, (batch_size, 1, input_z, input_y, input_x), order='F')
        gt = np.reshape(gt, (batch_size, 1, input_z, input_y, input_x), order='F')
    input = np.transpose(input, (0, 3, 4, 2, 1))
    insert_shape = np.zeros([batch_size,input_y,input_x,insert_z,1])
    input = np.concatenate((insert_shape,input,insert_shape),axis=3)
    insert_shape = np.zeros([batch_size,insert_xy,input_x,input_z+2*insert_z,1])
    input = np.concatenate((insert_shape,input,insert_shape),axis=1)
    insert_shape = np.zeros([batch_size,input_y+2*insert_xy,insert_xy,input_z+2*insert_z,1])
    input = np.concatenate((insert_shape,input,insert_shape),axis=2)
    gt = np.transpose(gt, (0, 3, 4, 2, 1))
    input_G = np.zeros([batch_size,input_y+insert_xy*2,input_x+insert_xy*2,(input_z+insert_z)*2,1])
    for z in range(insert_z,input_z*2+insert_z,2):
        input_G[:,insert_xy:input_y+insert_xy,insert_xy:input_x+insert_xy,z,0] = gt[:,:,:,(z-insert_z)//2,0]
        input_G[:,:,:,z+1,0] = input[:,:,:,(z+insert_z)//2,0]
    
    g_copy.set_weights(g.get_weights())
    output_G = g_copy.predict(input_G)[0]
    output_G = output_G[:,:,:,1::2,:]-output_G[:,:,:,0::2,:]
    gt = [np.concatenate((gt,output_G),axis=4),np.concatenate((gt,output_G),axis=4)]
    
    loss_total = g.train_on_batch(input, gt)
    loss_denoise.append(loss_total[1])
    loss_deconv.append(loss_total[2])

    elapsed_time = datetime.datetime.now() - start_time
    print("%d it: time: %s, denoise_loss = %.3e, deconv_loss = %.3e" % (it + 1 + load_init_model_iter, elapsed_time, loss_total[1], loss_total[2]))
    
    if (it + 1 + load_init_model_iter) % valid_interval == 0 or it==0:
        Validate(it + 1 + load_init_model_iter)

    if (it + 1 + load_init_model_iter) % test_interval == 0 or it==0:
        write_log(writer, 'NBR2NBR_loss', np.mean(loss_denoise), it + 1 + load_init_model_iter)
        write_log(writer, 'deconv_loss', np.mean(loss_deconv), it + 1 + load_init_model_iter)
        loss_denoise = []
        loss_deconv = []
        curlr = K.get_value(g.optimizer.learning_rate)
        write_log(writer, 'lr', curlr, it + 1 + load_init_model_iter)
        g.save_weights(save_weights_path + '/weights_' + str(it + 1 + load_init_model_iter) + '.h5')
        test(it + 1 + load_init_model_iter)

    curlr = K.get_value(g.optimizer.learning_rate)
    if (it + 1 + load_init_model_iter)%5000==0 or (it + 1 + load_init_model_iter)%7500==0:
        K.set_value(g.optimizer.learning_rate, curlr*lr_decay_factor)
