from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, add, multiply, Conv3D, LeakyReLU, Lambda, UpSampling3D, ReLU
import tensorflow as tf

def XxGlobalAveragePooling(input):
    return tf.reduce_mean(input, axis=(1, 2, 3), keepdims=True)


def CALayer(input, channel, reduction=16):
    W = Lambda(XxGlobalAveragePooling)(input)
    W = Conv3D(channel // reduction, kernel_size=1, activation='relu', padding='same')(W)
    W = Conv3D(channel, kernel_size=1, activation='sigmoid', padding='same')(W)
    mul = multiply([input, W])
    return mul


def RCAB(input, channel):
    conv = Conv3D(channel, kernel_size=3, padding='same')(input)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(channel, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    att = CALayer(conv, channel, reduction=16)
    output = add([att, input])
    return output


def ResidualGroup(input, channel, n_RCAB=5):
    conv = input
    for _ in range(n_RCAB):
        conv = RCAB(conv, channel)
    return conv


def RCAN3D(input_shape, upsample_flag=0, insert_z=2, insert_xy=8,
           n_ResGroup=2, n_RCAB=2):

    inputs = Input(input_shape)
    _,h,w,d,_ = inputs.shape
    conv = Conv3D(64, kernel_size=3, padding='same')(inputs)
    res = conv
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 64, n_RCAB=n_RCAB)
    conv = res+conv

    conv = Conv3D(64, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(1, kernel_size=3, padding='same')(conv)
    output1 = LeakyReLU(alpha=0.2)(conv)
    
    conv = Conv3D(64, kernel_size=3, padding='same')(output1)
    res = conv
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 64, n_RCAB=n_RCAB)
    conv = res+conv

    if upsample_flag:
        conv = UpSampling3D(size=(2, 2, 1))(conv)
    conv = Conv3D(64, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(1, kernel_size=3, padding='same')(conv)
    output2 = LeakyReLU(alpha=0.2)(conv)
    
    output1 = output1[:,insert_xy:h-insert_xy,insert_xy:w-insert_xy,insert_z:d-insert_z,:]
    model = Model(inputs=inputs, outputs=[output1,output2])

    return model

def RCAN3D_SIM(input_shape, upsample_flag, insert_z=2,insert_xy=8,
           n_ResGroup=2, n_RCAB=4):

    inputs = Input(input_shape)
    _,h,w,d,_ = inputs.shape
    conv = Conv3D(64, kernel_size=3, padding='same')(inputs)
    res = conv
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 64, n_RCAB=n_RCAB)
    conv = res+conv

    conv = Conv3D(256, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(1, kernel_size=3, padding='same')(conv)
    output1 = LeakyReLU(alpha=0.2)(conv)
    output1_reg = ReLU()(output1)
    conv = Conv3D(64, kernel_size=3, padding='same')(output1_reg)
    res = conv
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 64, n_RCAB=n_RCAB)
    conv = res+conv

    if upsample_flag:
        conv = UpSampling3D(size=(2, 2, 1))(conv)
    conv = Conv3D(256, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(1, kernel_size=3, padding='same')(conv)
    output2 = LeakyReLU(alpha=0.2)(conv)
    
    output1 = output1[:,insert_xy:h-insert_xy,insert_xy:w-insert_xy,insert_z:d-insert_z,:]
    model = Model(inputs=inputs, outputs=[output1,output2])

    return model


def RCAN3D_SIM_compact(input_shape, upsample_flag, insert_z=2,insert_xy=8,
           n_ResGroup=2, n_RCAB=2):

    inputs = Input(input_shape)
    _,h,w,d,_ = inputs.shape
    conv = Conv3D(64, kernel_size=3, padding='same')(inputs)
    res = conv
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 64, n_RCAB=n_RCAB)
    conv = res+conv

    conv = Conv3D(64, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(1, kernel_size=3, padding='same')(conv)
    output1 = LeakyReLU(alpha=0.2)(conv)
    output1_reg = ReLU()(output1)
    conv = Conv3D(64, kernel_size=3, padding='same')(output1_reg)
    res = conv
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 64, n_RCAB=n_RCAB)
    conv = res+conv

    if upsample_flag:
        conv = UpSampling3D(size=(2, 2, 1))(conv)
    conv = Conv3D(64, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(1, kernel_size=3, padding='same')(conv)
    output2 = LeakyReLU(alpha=0.2)(conv)
    
    output1 = output1[:,insert_xy:h-insert_xy,insert_xy:w-insert_xy,insert_z:d-insert_z,:]
    model = Model(inputs=inputs, outputs=[output1,output2])

    return model

def RCAN3D_SIM_compact2(input_shape, upsample_flag, insert_z=2,insert_xy=8,
           n_ResGroup=2, n_RCAB=2):

    inputs = Input(input_shape)
    _,h,w,d,_ = inputs.shape
    conv = Conv3D(64, kernel_size=3, padding='same')(inputs)
    res = conv
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 32, n_RCAB=n_RCAB)
    conv = res+conv

    conv = Conv3D(32, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(1, kernel_size=3, padding='same')(conv)
    output1 = LeakyReLU(alpha=0.2)(conv)
    output1_reg = ReLU()(output1)
    conv = Conv3D(32, kernel_size=3, padding='same')(output1_reg)
    res = conv
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 32, n_RCAB=n_RCAB)
    conv = res+conv

    if upsample_flag:
        conv = UpSampling3D(size=(2, 2, 1))(conv)
    conv = Conv3D(32, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(1, kernel_size=3, padding='same')(conv)
    output2 = LeakyReLU(alpha=0.2)(conv)
    
    output1 = output1[:,insert_xy:h-insert_xy,insert_xy:w-insert_xy,insert_z:d-insert_z,:]
    model = Model(inputs=inputs, outputs=[output1,output2])

    return model