import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from utils.utils import read_mrc,prctile_norm
from scipy.interpolate import interp1d
import numpy.fft as F
from scipy import asarray as ar, exp
from scipy.optimize import leastsq, minimize, curve_fit
import math

def create_psf_loss(psf, TV_weight, Hess_weight, laplace_weight,
                    l1_rate, mse_flag,  
                    upsample_flag, insert_xy, deconv_flag):
    def psf_loss(y_true, y_pred):
        
        _,height,width,_ = y_pred.get_shape().as_list()
        
        y_true = tf.cast(y_true, tf.float32)
        if deconv_flag:
            psf_local = psf
            y_conv = K.conv2d(y_pred, psf_local, padding='same')
        else:
            y_conv = y_pred
        if upsample_flag:
            y_conv = tf.image.resize(y_conv,[height//2,width//2])
        y_conv = y_conv[:,insert_xy:y_conv.shape[1]-insert_xy,insert_xy:y_conv.shape[2]-insert_xy,:]
        if mse_flag:
            psf_loss = K.mean(K.square(y_true - y_conv))
        else:
            psf_loss = K.mean(K.abs(y_true - y_conv))
        
        if TV_weight>0 or laplace_weight>0:
            y = tf.slice(y_pred, [0, 0, 0, 0], tf.stack([-1, height-1, -1, -1])) - tf.slice(y_pred, [0, 1, 0, 0], [-1, -1, -1, -1])
            x = tf.slice(y_pred, [0, 0, 0, 0], tf.stack([-1, -1, width-1, -1])) - tf.slice(y_pred, [0, 0, 1, 0], [-1, -1, -1, -1])
            x_diff = tf.nn.l2_loss(x) / tf.cast(tf.size(x),tf.float32)
            y_diff = tf.nn.l2_loss(y) / tf.cast(tf.size(y),tf.float32)
            TV_loss = x_diff + y_diff
        else:
            TV_loss = 0
            
        if Hess_weight>0:
            if not TV_weight>0:
                y = tf.slice(y_pred, [0, 0, 0, 0], tf.stack([-1, height-1, -1, -1])) - tf.slice(y_pred, [0, 1, 0, 0], [-1, -1, -1, -1])
                x = tf.slice(y_pred, [0, 0, 0, 0], tf.stack([-1, -1, width-1, -1])) - tf.slice(y_pred, [0, 0, 1, 0], [-1, -1, -1, -1])
            xx = tf.slice(x, [0, 0, 0, 0], tf.stack([-1, -1, width-2, -1])) - tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
            yy = tf.slice(y, [0, 0, 0, 0], tf.stack([-1, height-2, -1, -1])) - tf.slice(y, [0, 1, 0, 0], [-1, -1, -1, -1])
            xy = tf.slice(y, [0, 0, 0, 0], tf.stack([-1, -1, width-1, -1])) - tf.slice(y, [0, 0, 1, 0], [-1, -1, -1, -1])
            yx = tf.slice(x, [0, 0, 0, 0], tf.stack([-1, height-1, -1, -1])) - tf.slice(x, [0, 1, 0, 0], [-1, -1, -1, -1])
            Hess_loss = tf.nn.l2_loss(xx) / tf.cast(tf.size(xx),tf.float32) + tf.nn.l2_loss(yy) / tf.cast(tf.size(yy),tf.float32) + tf.nn.l2_loss(yx) / tf.cast(tf.size(yx),tf.float32) + tf.nn.l2_loss(xy) / tf.cast(tf.size(xy),tf.float32)
        else:
            Hess_loss = 0
            
        if laplace_weight>0:
            kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]]).astype(np.float32)
            kernel = np.reshape(kernel,(3,3,1,1))
            laplace_loss = K.conv2d(y_pred, kernel)
            laplace_loss = K.mean(K.square(laplace_loss))
        else:
            laplace_loss = 0
            
        if l1_rate>0:
            l1_loss = K.mean(K.abs(y_pred))
        else:
            l1_loss = 0

        return psf_loss+TV_weight*TV_loss+laplace_weight*laplace_loss+Hess_weight*Hess_loss+l1_rate*l1_loss
    return psf_loss

def create_NBR2NBR_loss(TV_rate,mse_flag):
    def NBR2NBR_loss(gt,output):
        output_G = gt[:,:,:,:,1]
        gt = gt[:,:,:,:,0]
        output = tf.squeeze(output,axis=4)
        if output.shape[1]>gt.shape[1]:
            output = tf.image.resize(output,(output.shape[1]//2,output.shape[2]//2))
        _,height,width,depth = output.shape
        
        if mse_flag:
            loss = K.mean(K.square(gt-output))
            reg_loss = K.mean(K.square(output-gt-output_G))
        else:
            loss = K.mean(K.abs(gt-output))
            reg_loss = K.mean(K.abs(output-gt-output_G))
        
        TV_loss = 0
        if TV_rate>0:
            y = tf.slice(output, [0, 0, 0, 0], tf.stack([-1, height-1, -1, -1])) - tf.slice(output, [0, 1, 0, 0], [-1, -1, -1, -1])
            x = tf.slice(output, [0, 0, 0, 0], tf.stack([-1, -1, width-1, -1])) - tf.slice(output, [0, 0, 1, 0], [-1, -1, -1, -1])
            z = tf.slice(output, [0, 0, 0, 0], tf.stack([-1, -1, -1, depth-1])) - tf.slice(output, [0, 0, 0, 1], [-1, -1, -1, -1])
            TV_loss = K.mean(K.square(x))+K.mean(K.square(y))+K.mean(K.square(z))
        
        return loss+reg_loss+TV_rate*TV_loss
    return NBR2NBR_loss

def create_psf_loss_3D_NBR2NBR(psf, mse_flag, batch_size, upsample_flag, 
                               TV_weight, Hess_weight, 
                               insert_z, insert_xy):
    def psf_loss(y_true,y_pred):
        output_G = y_true[:,:,:,:,1]
        y_true = y_true[:,:,:,:,0]
        h = y_pred.shape[1]
        w = y_pred.shape[2]
        d = y_pred.shape[3]
        if upsample_flag:
            insert_xy_local = insert_xy*2
        else:
            insert_xy_local = insert_xy
        y_pred_conv = K.conv3d(y_pred, psf, padding='same')[:,insert_xy_local:h-insert_xy_local,insert_xy_local:w-insert_xy_local,insert_z:d-insert_z,:]
        y_pred = tf.squeeze(y_pred,axis=4)[:,insert_xy_local:h-insert_xy_local,insert_xy_local:w-insert_xy_local,insert_z:d-insert_z]
        y_pred_conv = tf.squeeze(y_pred_conv,axis=4)
        if upsample_flag:
            y_pred_conv = tf.image.resize(y_pred_conv,[y_pred_conv.shape[1]//2,y_pred_conv.shape[2]//2])
        if mse_flag:
            loss = K.mean(K.square(y_true-y_pred_conv))
            reg_loss = K.mean(K.square(y_pred_conv-y_true-output_G))
        else:
            loss = K.mean(K.abs(y_true-y_pred_conv))  
            reg_loss = K.mean(K.abs(y_pred_conv-y_true-output_G))

        h = y_pred.shape[1]
        w = y_pred.shape[2]
        d = y_pred.shape[3]
        if TV_weight>0:
            y = tf.slice(y_pred, [0, 0, 0, 0], tf.stack([-1, h-1, -1, -1])) - tf.slice(y_pred, [0, 1, 0, 0], [-1, -1, -1, -1])
            x = tf.slice(y_pred, [0, 0, 0, 0], tf.stack([-1, -1, w-1, -1])) - tf.slice(y_pred, [0, 0, 1, 0], [-1, -1, -1, -1])
            if d==1:
                z=0.0
            else:
                z = tf.slice(y_pred, [0, 0, 0, 0], tf.stack([-1, -1, -1, d-1])) - tf.slice(y_pred, [0, 0, 0, 1], [-1, -1, -1, -1])
            TV_loss = K.mean(K.square(x))+K.mean(K.square(y))+K.mean(K.square(z))
        else:
            TV_loss = 0
        hess_loss = 0
        if Hess_weight>0:
            x = y_pred[:,1:,:,:]-y_pred[:,:h-1,:,:]
            y = y_pred[:,:,1:,:]-y_pred[:,:,:w-1,:]
            if d>1:
                z = y_pred[:,:,:,1:]-y_pred[:,:,:,:d-1]
                for tv in [x,y,z]:
                    hess = tv[:,1:,:,:]-tv[:,:-1,:,:]
                    hess_loss = hess_loss + K.mean(K.square(hess))
                    hess = tv[:,:,1:,:]-tv[:,:,:-1,:]
                    hess_loss = hess_loss + K.mean(K.square(hess))
                    hess = tv[:,:,:,1:]-tv[:,:,:,:-1]
                    hess_loss = hess_loss + K.mean(K.square(hess))
            else:
                for tv in [x,y]:
                    hess = tv[:,1:,:,:]-tv[:,:-1,:,:]
                    hess_loss = hess_loss + K.mean(K.square(hess))
                    hess = tv[:,:,1:,:]-tv[:,:,:-1,:]
                    hess_loss = hess_loss + K.mean(K.square(hess))

        return loss+reg_loss+TV_weight*TV_loss+Hess_weight*hess_loss
    return psf_loss

def gaussian_1d(x, *param):
    return param[0] * np.exp(-np.power(x - param[1], 2.) / (2 * np.power(param[2], 2.)))

def cal_psf_2d(otf_path, Ny, Nx, dky, dkx):
    dkr = np.min([dkx, dky])
    headerotf, rawOTF = read_mrc(otf_path)
    nxotf = headerotf[0][0]
    dkrotf = headerotf[0][10]
    diagdist = int(np.sqrt(np.square(Nx / 2) + np.square(Ny / 2)) + 2)
    rawOTF = np.squeeze(rawOTF)
    OTF = rawOTF[0:-1:2]
    x = np.arange(0, nxotf * dkrotf, dkrotf)
    xi = np.arange(0, (nxotf - 1) * dkrotf, dkr)
    interp = interp1d(x, OTF, kind='slinear')
    OTF = interp(xi)
    sizeOTF = len(OTF)
    prol_OTF = np.zeros((diagdist * 2))
    prol_OTF[0:sizeOTF] = OTF
    OTF = prol_OTF
    kxx = dkx * np.arange(-Nx / 2, Nx / 2, 1)
    kyy = dky * np.arange(-Ny / 2, Ny / 2, 1)
    [dX, dY] = np.meshgrid(kyy, kxx)
    rdist = np.sqrt(np.square(dX) + np.square(dY))
    otflen = len(OTF)
    x = np.arange(0, otflen * dkr, dkr)
    interp = interp1d(x, OTF, kind='slinear')
    OTF = interp(rdist)
    PSF = abs(F.ifftshift(F.ifft2(OTF)))
    PSF = PSF / np.sum(PSF)
    return PSF, OTF

def psf_estimator_2d(psf):
    shape = psf.shape
    max_index = np.where(psf == psf.max())
    index_y = max_index[0][0]
    index_x = max_index[1][0]
    # estimate y sigma
    x = ar(range(shape[0]))
    y = prctile_norm(np.squeeze(psf[:, index_x]))
    fit_y, cov_y = curve_fit(gaussian_1d, x, y, p0=[1, index_y, 2])
    print('estimated psf sigma_y: ', fit_y[2])
    # estimate x sigma
    x = ar(range(shape[1]))
    y = prctile_norm(np.squeeze(psf[index_y, :]))
    fit_x, cov_x = curve_fit(gaussian_1d, x, y, p0=[1, index_x, 2])
    print('estimated psf sigma_x: ', fit_x[2])
    return fit_y[2], fit_x[2]

def cal_psf_3d(headerotf, rawOTF, Ny, Nx, Nz, dky, dkx, dkz):
    # read raw OTF, reconstruct it and get the 0y slice as new raw OTF
    dkr = np.min([dkx, dky])
    nxotf = headerotf[0][0]
    nyotf = headerotf[0][1]
    nzotf = headerotf[0][2]
    dkzotf = 0.0497
    dkrotf = 0.0302
    lenotf = len(rawOTF)
    rOTF = rawOTF[np.arange(0, lenotf, 2)]
    iOTF = rawOTF[np.arange(1, lenotf, 2)]
    rawOTF = rOTF + 1j * iOTF
    rawOTF = np.abs(np.reshape(rawOTF, (nzotf, nyotf, nxotf)))
    rawOTF = np.transpose(rawOTF, (2, 1, 0))
    rawOTF = rawOTF[:, :, 0]
    
    #get OTF on x0z plane, which has desired dkz and dkr
    z = np.arange(0, nxotf * dkzotf, dkzotf)
    zi = np.arange(0, nxotf * dkzotf, dkz)
    zi = zi[0:-1]
    x = np.arange(0, nyotf * dkrotf, dkrotf)
    xi = np.arange(0, nyotf * dkrotf, dkr)
    xi = xi[0:-1]

    OTF1 = []
    for j in range(nxotf):
        curRow = rawOTF[j, :]
        interp = interp1d(x, curRow, 'slinear')
        OTF1.append(interp(xi))
    OTF1 = np.transpose(OTF1, (1, 0))

    OTF2 = []
    for k in range(np.size(OTF1, 0)):
        curCol = OTF1[k]
        interp = interp1d(z, curCol, 'slinear')
        OTF2.append(interp(zi))
    OTF2 = np.transpose(OTF2, (1, 0))

    OTF = F.fftshift(OTF2, 0)
    
    #expand OTF's value on diagdist
    #rotate OTF around z-axis by interpolation
    otflen = np.size(OTF, 1)
    otfheight = np.size(OTF, 0)
    
    diagdist = math.ceil(np.sqrt(np.square(Nx / 2) + np.square(Ny / 2)) + 1)
    prol_OTF = np.zeros((Nz, diagdist))
    if Nz >= otfheight:
        prol_OTF[Nz//2-otfheight//2:Nz//2+otfheight//2, 0:otflen] = OTF
    else:
        prol_OTF[:,0:otflen] = OTF[otfheight//2-Nz//2:otfheight//2+Nz//2+1,:]
    OTF = prol_OTF

    x = np.arange(-Nx / 2, Nx / 2, 1) * dkx
    y = np.arange(-Ny / 2, Ny / 2, 1) * dky
    [X, Y] = np.meshgrid(x, y)
    rdist = np.sqrt(np.square(X) + np.square(Y))

    curOTF = np.zeros((Ny, Nx, Nz))
    x = np.arange(0, diagdist * dkr, dkr)
    for z in range(Nz):
        OTFz = OTF[z, :]
        interp = interp1d(x, OTFz, 'slinear')
        curOTF[:, :, z] = interp(rdist)

    curOTF = np.abs(curOTF)
    curOTF = curOTF / np.max(curOTF)

    temp = np.zeros_like(curOTF) + 1j * np.zeros_like(curOTF)
    for j in range(Nz):
        temp[:, :, j] = F.ifftshift(F.ifft2(np.squeeze(curOTF[:, :, j])))
    PSF = np.abs(F.ifftshift(F.ifft(temp, axis=2), axes=2))
    PSF = PSF / np.sum(PSF)
    return PSF, curOTF

def psf_estimator_3d(psf):
    shape = psf.shape
    max_index = np.where(psf == psf.max())
    index_y = max_index[0][0]
    index_x = max_index[1][0]
    index_z = max_index[2][0]
    # estimate y sigma
    x = ar(range(shape[0]))
    y = prctile_norm(np.squeeze(psf[:, index_x, index_z]))
    fit_y, cov_y = curve_fit(gaussian_1d, x, y, p0=[1, index_y, 2])
    print('estimated psf sigma_y: ', fit_y[2])
    # estimate x sigma
    x = ar(range(shape[1]))
    y = prctile_norm(np.squeeze(psf[index_y, :, index_z]))
    fit_x, cov_x = curve_fit(gaussian_1d, x, y, p0=[1, index_x, 2])
    print('estimated psf sigma_x: ', fit_x[2])
    # estimate z sigma
    x = ar(range(shape[2]))
    y = prctile_norm(np.squeeze(psf[index_y, index_x, :]))
    fit_z, cov_z = curve_fit(gaussian_1d, x, y, p0=[1, index_z, 2])
    print('estimated psf sigma_z: ', fit_z[2])
    return fit_y[2], fit_x[2], fit_z[2]