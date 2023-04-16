import numpy as np
import imageio
from utils.utils import prctile_norm
import os

def DataLoader(images_path, data_path, gt_path, 
                  batch_size, norm_flag):
    batch_images_path = np.random.choice(images_path, size=batch_size, replace=False)
    image_batch = []
    gt_batch = []
    for path in batch_images_path:
        path_gt = path.replace(data_path, gt_path) #str.replace(old, new)
        while not os.path.exists(path_gt):
            path= np.random.choice(images_path, size=1, replace=False)
            path_gt = path.replace(data_path, gt_path) #str.replace(old, new)
        img = np.array(imageio.mimread(path)).astype(np.float32)
        gt = np.array(imageio.mimread(path_gt)).astype(np.float32)
        
        if norm_flag:
            img = prctile_norm(img)
            gt = prctile_norm(gt)
        else:
            img = img / 65535
            gt = gt / 65535
        
        image_batch.append(img)
        gt_batch.append(gt)

    image_batch = np.array(image_batch).astype(np.float32)
    gt_batch = np.array(gt_batch).astype(np.float32)
    
    return image_batch, gt_batch

def augment_img(img,mode):
    if mode==1:
        img = np.flipud(np.rot90(img))
    elif mode==2:
        img = np.flipud(img)
    elif mode==3:
        img = np.rot90(img,k=3)
    elif mode==4:
        img = np.flipud(np.rot90(img,k=2))
    elif mode==5:
        img = np.rot90(img)
    elif mode==6:
        img = np.rot90(img,k=2)
    elif mode==7:
        img = np.flipud(np.rot90(img,k=3))
        
    return img