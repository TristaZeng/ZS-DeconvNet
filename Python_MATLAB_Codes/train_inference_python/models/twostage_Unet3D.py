from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Permute, Layer
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Dropout, add, concatenate
from tensorflow.keras.layers import MaxPooling3D, UpSampling3D, Conv3D, LeakyReLU
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
import tensorflow as tf


def Unet(input_shape,upsample_flag,insert_z=2,insert_xy=8,conv_block_num=4,conv_num=3):
    
    inputs = Input(input_shape)
    _,h,w,d,_ = inputs.shape

    pool = inputs
    
    #Encoder
    conv_list = []
    for n in range(conv_block_num):
        channels = 2 ** (n+5)
        pool, conv = conv_block(pool, conv_num, channels)
        conv_list.append(conv)

    mid = Conv3D(channels*2, kernel_size=3, activation='relu', padding='same')(pool)
    mid = Conv3D(channels, kernel_size=3, activation='relu', padding='same')(mid)
    
    #Decoder
    concat_block_num = conv_block_num
    init_channels = channels
    conv = mid
    for n in range(concat_block_num):
        channels = init_channels // (2 ** n)
        conv = concat_block(conv, conv_list[-(n+1)], conv_num, channels)

    #output denoised img
    output1 = Conv3D(1, kernel_size=3, activation='relu', padding='same')(conv)

    #Encoder
    pool = output1
    conv_list = []
    for n in range(conv_block_num):
        channels = 2 ** (n+5)
        pool, conv = conv_block(pool, conv_num, channels)
        conv_list.append(conv)

    mid = Conv3D(channels*2, kernel_size=3, activation='relu', padding='same')(pool)
    mid = Conv3D(channels, kernel_size=3, activation='relu', padding='same')(mid)
    
    #Decoder
    concat_block_num = conv_block_num
    init_channels = channels
    conv = mid
    for n in range(concat_block_num):
        channels = init_channels // (2 ** n)
        conv = concat_block(conv, conv_list[-(n+1)], conv_num, channels)

    #output deconved img
    if upsample_flag:
        conv = UpSampling3D(size=(2, 2, 1))(conv)
    conv = Conv3D(64, kernel_size=3, activation='relu', padding='same')(conv)
    conv = Conv3D(64, kernel_size=3, activation='relu', padding='same')(conv)
    output2 = Conv3D(1, kernel_size=3, activation='relu', padding='same')(conv)
    output1 = output1[:,insert_xy:h-insert_xy,insert_xy:w-insert_xy,insert_z:d-insert_z,:]
    
    model = Model(inputs=inputs, outputs=[output1,output2])
    return model

def conv_block(input_layer, conv_num, channels):
    conv = input_layer
    for _ in range(conv_num):
        conv = Conv3D(channels, kernel_size=3, activation='relu', padding='same')(conv)
    #conv = conv + input_layer
    pool = MaxPooling3D(pool_size=(2, 2, 1))(conv)
    print(pool.shape)
    return pool, conv

def concat_block(concat1, concat2, conv_num, channels):
    up = concatenate([UpSampling3D(size=(2, 2, 1))(concat1), concat2], axis=4)
    conv = Conv3D(channels, kernel_size=3, activation='relu', padding='same')(up)
    for _ in range(conv_num-1):
        conv = Conv3D(channels//2, kernel_size=3, activation='relu', padding='same')(conv)
    return conv