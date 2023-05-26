from keras.models import Model
from keras.layers import Input, add, multiply, Conv3D, LeakyReLU, Lambda, UpSampling3D, UpSampling2D
import tensorflow as tf
from keras.layers import Layer
from keras import initializers
import keras.backend as K
import numpy as np


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


# NSM_flag=0
def RCAN3D(input_shape, NSM_flag=0, upsample_flag=0, insert_slices=2, insert_xy=8,
           n_ResGroup=2, n_RCAB=4):
    inputs = Input(input_shape)
    _, h, w, d, _ = inputs.shape

    if NSM_flag:
        NSM = NoiseSuppressionModule()
        ns = NSM(inputs)
        weighted = Conv3D(input_shape[3], kernel_size=1, padding='same')(inputs)
        weighted = LeakyReLU(alpha=0.2)(weighted)
        ns = ns + weighted
    else:
        ns = inputs
    conv = Conv3D(64, kernel_size=3, padding='same')(ns)
    res = conv
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 64, n_RCAB=n_RCAB)
    conv = add([res, conv])

    conv = Conv3D(256, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(1, kernel_size=3, padding='same')(conv)
    output1 = LeakyReLU(alpha=0.2)(conv)

    conv = Conv3D(64, kernel_size=3, padding='same')(output1)
    res = conv
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 64, n_RCAB=n_RCAB)
    conv = add([res, conv])
    # before_up = Lambda(slice)(conv)
    if upsample_flag:
        conv = Lambda(up_3d)(conv)
        # conv = UpSampling3D(size=(2, 2, 1))(conv)
    # up = Lambda(slice)(conv)
    conv = Conv3D(256, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(1, kernel_size=3, padding='same')(conv)
    output2 = LeakyReLU(alpha=0.2)(conv)

    # output1 = output1[:, insert_xy:h - insert_xy, insert_xy:w - insert_xy, insert_slices:d - insert_slices, :]
    model = Model(inputs=inputs, outputs=[output1, output2])

    return model

def RCAN3D_prun(input_shape, NSM_flag=0, upsample_flag=0, insert_slices=2, insert_xy=8,
                n_ResGroup=2, n_RCAB=2):
    inputs = Input(input_shape)
    _, h, w, d, _ = inputs.shape

    if NSM_flag:
        NSM = NoiseSuppressionModule()
        ns = NSM(inputs)
        weighted = Conv3D(input_shape[3], kernel_size=1, padding='same')(inputs)
        weighted = LeakyReLU(alpha=0.2)(weighted)
        ns = ns + weighted
    else:
        ns = inputs
    conv = Conv3D(64, kernel_size=3, padding='same')(ns)
    res = conv
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 64, n_RCAB=n_RCAB)
    conv = add([res, conv])

    conv = Conv3D(64, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(1, kernel_size=3, padding='same')(conv)
    output1 = LeakyReLU(alpha=0.2)(conv)

    conv = Conv3D(64, kernel_size=3, padding='same')(output1)
    res = conv
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 64, n_RCAB=n_RCAB)
    conv = add([res, conv])
    # before_up = Lambda(slice)(conv)
    if upsample_flag:
        conv = Lambda(up_3d)(conv)
        # conv = UpSampling3D(size=(2, 2, 1))(conv)
    # up = Lambda(slice)(conv)
    conv = Conv3D(64, kernel_size=3, padding='same')(conv)
    conv = LeakyReLU(alpha=0.2)(conv)
    conv = Conv3D(1, kernel_size=3, padding='same')(conv)
    output2 = LeakyReLU(alpha=0.2)(conv)

    # output1 = output1[:, insert_xy:h - insert_xy, insert_xy:w - insert_xy, insert_slices:d - insert_slices, :]
    model = Model(inputs=inputs, outputs=[output1, output2])

    return model


def slice(x):
    return x[:, :, :, :, 0:1]


def up_3d(x):
    _, _, _, _, c = x.get_shape()
    x = K.permute_dimensions(x, [4, 0, 1, 2, 3])
    sum = UpSampling2D(size=(2, 2))(x[0])[np.newaxis]
    for i in range(c - 1):
        tmp = UpSampling2D(size=(2, 2))(x[i + 1])[np.newaxis]
        sum = K.concatenate([sum, tmp], axis=0)
    sum = K.permute_dimensions(sum, [1, 2, 3, 4, 0])
    return sum


class NoiseSuppressionModule(Layer):

    def __init__(self, init_cutoff_freq=4.1, dxy=0.0926, init_slop=100):
        super(NoiseSuppressionModule, self).__init__()
        self.cutoff_freq = self.add_weight(shape=(1,), initializer=initializers.constant(init_cutoff_freq),
                                           trainable=True, name='cutoff_freq')
        self.slop = self.add_weight(shape=(1,), initializer=initializers.constant(init_slop),
                                    trainable=True, name='slop')
        self.dxy = tf.Variable(initial_value=(dxy), trainable=False, name='dxy')

    def call(self, inputs):
        bs, ny, nx, nz, ch = inputs.get_shape().as_list()
        ny = tf.cast(ny, tf.float32)
        nx = tf.cast(nx, tf.float32)
        dkx = tf.divide(1, tf.multiply(nx, self.dxy))
        dky = tf.divide(1, tf.multiply(ny, self.dxy))

        y = tf.multiply(tf.cast(tf.range(-ny // 2, ny // 2), tf.float32), dky)
        x = tf.multiply(tf.cast(tf.range(-nx // 2, nx // 2), tf.float32), dkx)
        [X, Y] = tf.meshgrid(x, y)
        rdist = tf.sqrt(tf.square(X) + tf.square(Y))

        otf_mask = tf.sigmoid(tf.multiply(self.cutoff_freq - rdist, self.slop))
        otf_mask = tf.expand_dims(tf.expand_dims(tf.expand_dims(otf_mask, 0), 0), 0)
        otf_mask = tf.tile(otf_mask, (1, nz, ch, 1, 1))
        otf_mask = tf.complex(otf_mask, tf.zeros_like(otf_mask))

        inputs = tf.complex(inputs, tf.zeros_like(inputs))
        inputs = tf.transpose(inputs, [0, 3, 4, 1, 2])
        fft_feature = tf.signal.fftshift(tf.signal.fft2d(inputs))
        output = tf.signal.ifft2d(tf.signal.fftshift(tf.multiply(otf_mask, fft_feature)))
        output = tf.transpose(output, [0, 3, 4, 1, 2])
        output = tf.math.real(output)

        return output
