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
        
        if norm_flag==1:
            img = prctile_norm(img)
            gt = prctile_norm(gt)
        elif norm_flag==0:
            img = img / 65535
            gt = gt / 65535
        elif norm_flag==2:
            img = img/np.max(img)
            gt = gt/np.max(gt)
        
        image_batch.append(img)
        gt_batch.append(gt)

    image_batch = np.array(image_batch).astype(np.float32)
    gt_batch = np.array(gt_batch).astype(np.float32)
    
    return image_batch, gt_batch

