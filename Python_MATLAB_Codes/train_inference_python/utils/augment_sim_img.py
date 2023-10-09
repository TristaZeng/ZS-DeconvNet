import os
import glob
import tifffile
import numpy as np
import math

def aug_sim_img_2D(save_dir,input_dir,gt_dir,patch_size,num_patch=50000):
    input_list = sorted(glob.glob(os.path.join(input_dir,'*.tif')))
    gt_list = sorted(glob.glob(os.path.join(gt_dir,'*.tif')))
    
    img_num = len(input_list)
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    input_save_dir = os.path.join(save_dir,'input')
    gt_save_dir = os.path.join(save_dir,'gt')
    
    if not os.path.exists(input_save_dir):
        os.mkdir(input_save_dir)
        
    if not os.path.exists(gt_save_dir):
        os.mkdir(gt_save_dir)
    
    for save_id in range(num_patch):
        select_id = np.random.randint(0,img_num)
        
        input_img = np.squeeze(tifffile.imread(input_list[select_id]))
        gt_img = np.squeeze(tifffile.imread(gt_list[select_id]))
        
        # crop
        h = input_img.shape[0]
        w = input_img.shape[1]
        
        x_ind = math.floor((h-patch_size)*np.random.rand(1))
        y_ind = math.floor((w-patch_size)*np.random.rand(1))
        input_img = input_img[x_ind:x_ind+patch_size,y_ind:y_ind+patch_size]
        gt_img = gt_img[x_ind:x_ind+patch_size,y_ind:y_ind+patch_size]
        
        #augment
        mode = np.random.randint(0,8)
        input_img = augment_img(input_img,mode)
        gt_img = augment_img(gt_img,mode)
        
        save_name = '%08d'%(save_id+1) + '.tif'
        tifffile.imwrite(os.path.join(input_save_dir,save_name),input_img)
        tifffile.imwrite(os.path.join(gt_save_dir,save_name),gt_img)

def aug_sim_img_3D(save_dir,input_dir,gt_dir,patch_size,z_size,num_patch=10000):
    input_list = sorted(glob.glob(os.path.join(input_dir,'*.tif')))
    gt_list = sorted(glob.glob(os.path.join(gt_dir,'*.tif')))
    
    img_num = len(input_list)
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    input_save_dir = os.path.join(save_dir,'input')
    gt_save_dir = os.path.join(save_dir,'gt')
    
    if not os.path.exists(input_save_dir):
        os.mkdir(input_save_dir)
        
    if not os.path.exists(gt_save_dir):
        os.mkdir(gt_save_dir)
    
    for save_id in range(num_patch):
        select_id = np.random.randint(0,img_num)
        
        input_img = np.squeeze(tifffile.imread(input_list[select_id]))
        gt_img = np.squeeze(tifffile.imread(gt_list[select_id]))
        
        # crop
        z = input_img.shape[0]
        h = input_img.shape[1]
        w = input_img.shape[2]
        
        z_ind = math.floor((z-z_size)*np.random.rand(1))
        x_ind = math.floor((h-patch_size)*np.random.rand(1))
        y_ind = math.floor((w-patch_size)*np.random.rand(1))
        
        input_img = input_img[z_ind:z_ind+z_size,x_ind:x_ind+patch_size,y_ind:y_ind+patch_size]
        gt_img = gt_img[z_ind:z_ind+z_size,x_ind:x_ind+patch_size,y_ind:y_ind+patch_size]
        
        #augment
        mode = np.random.randint(0,8)
        
        input_img = augment_img_3D(input_img,mode)
        gt_img = augment_img_3D(gt_img,mode)
        
        save_name = '%08d'%(save_id+1) + '.tif'
        tifffile.imwrite(os.path.join(input_save_dir,save_name),input_img)
        tifffile.imwrite(os.path.join(gt_save_dir,save_name),gt_img)
        
def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
        
def augment_img_3D(img, mode=0):
    img_out = img.copy()
    Z = img_out.shape[0]
    for z in range(Z):
        img_out[z,...] = augment_img(img[z,...], mode)
    
    return img_out