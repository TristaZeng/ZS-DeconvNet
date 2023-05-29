# todo upgrade to keras 2.0

from keras import optimizers
from keras.models import Model
from keras.layers import Layer, Lambda
import keras.backend as K
from keras.layers import Input, concatenate, AveragePooling2D
from keras.layers import MaxPooling2D, UpSampling2D, Conv2D, subtract, add, multiply, GlobalAveragePooling2D, Average
from keras import initializers
import tensorflow as tf



def Unet(input_shape, NSM_flag, upsample_flag, conv_block_num=4, conv_num=3):
    input1 = Input(input_shape)

    # OTF Attenuation
    if NSM_flag:
        oa = OTFAttenuation(init_cutoff_freq=4.95, dxy=0.0626)(input1)
        pool = oa
    else:
        pool = input1

    # Encoder
    conv_list = []
    for n in range(conv_block_num):
        channels = 2 ** (n + 5)
        pool, conv = conv_block(pool, conv_num, channels)
        conv_list.append(conv)

    mid = Conv2D(channels * 2, kernel_size=3, activation='relu', padding='same')(pool)
    mid = Conv2D(channels, kernel_size=3, activation='relu', padding='same')(mid)

    # Decoder
    concat_block_num = conv_block_num
    init_channels = channels
    conv = mid
    for n in range(concat_block_num):
        channels = init_channels // (2 ** n)
        conv = concat_block(conv, conv_list[-(n + 1)], conv_num, channels)

    # output denoised img
    output1 = Conv2D(1, kernel_size=3, activation='relu', padding='same')(conv)

    # Encoder
    pool = output1
    conv_list = []
    for n in range(conv_block_num):
        channels = 2 ** (n + 5)
        pool, conv = conv_block(pool, conv_num, channels)
        conv_list.append(conv)

    mid = Conv2D(channels * 2, kernel_size=3, activation='relu', padding='same')(pool)
    mid = Conv2D(channels, kernel_size=3, activation='relu', padding='same')(mid)

    # Decoder
    concat_block_num = conv_block_num
    init_channels = channels
    conv = mid
    for n in range(concat_block_num):
        channels = init_channels // (2 ** n)
        conv = concat_block(conv, conv_list[-(n + 1)], conv_num, channels)

    # output deconved img
    if upsample_flag:
        conv = UpSampling2D(size=(2, 2))(conv)
    conv = Conv2D(128, kernel_size=3, activation='relu', padding='same')(conv)
    conv = Conv2D(128, kernel_size=3, activation='relu', padding='same')(conv)
    output2 = Conv2D(1, kernel_size=3, activation='relu', padding='same')(conv)

    model = Model(inputs=input1, outputs=[output1, output2])
    return model


def conv_block(input_layer, conv_num, channels):
    conv = input_layer
    for _ in range(conv_num):
        conv = Conv2D(channels, kernel_size=3, activation='relu', padding='same')(conv)
    # conv = conv + input_layer
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return pool, conv


def concat_block(concat1, concat2, conv_num, channels):
    up = concatenate([UpSampling2D(size=(2, 2))(concat1), concat2], axis=3)
    conv = Conv2D(channels, kernel_size=3, activation='relu', padding='same')(up)
    for _ in range(conv_num - 1):
        conv = Conv2D(channels // 2, kernel_size=3, activation='relu', padding='same')(conv)
    return conv

def identity(x):
    return K.identity(x)

def compute_y_pred(x):
    y_pred = x[0]
    psf = x[1]
    psf_kernel = K.permute_dimensions(psf, [1, 2, 3, 0])
    y_pred = K.conv2d(y_pred, psf_kernel, padding='same')

    _, height, width, _ = y_pred.get_shape()
    y_pred = tf.image.resize_images(y_pred, [height.value // 2, width.value // 2])
    # y_pred = AveragePooling2D(pool_size=(2, 2))(y_pred)
    return y_pred


class OTFAttenuation(Layer):

    def __init__(self, init_cutoff_freq, dxy, init_slop=100):
        # slop defines the smoothness of the edge
        super(OTFAttenuation, self).__init__()
        self.cutoff_freq = self.add_weight(shape=(1,), initializer=initializers.constant(init_cutoff_freq),
                                           trainable=True, name='cutoff_freq')
        self.slop = self.add_weight(shape=(1,), initializer=initializers.constant(init_slop),
                                    trainable=True, name='slop')
        self.dxy = tf.Variable(initial_value=(dxy), trainable=False, name='dxy')

    def call(self, inputs):
        bs, ny, nx, ch = inputs.get_shape().as_list()
        ny = tf.cast(ny, tf.float32)
        nx = tf.cast(nx, tf.float32)
        dkx = tf.divide(1, tf.multiply(nx, self.dxy))
        dky = tf.divide(1, tf.multiply(ny, self.dxy))

        y = tf.multiply(tf.cast(tf.range(-ny // 2, ny // 2), tf.float32), dky)
        x = tf.multiply(tf.cast(tf.range(-nx // 2, nx // 2), tf.float32), dkx)
        [X, Y] = tf.meshgrid(x, y)
        rdist = tf.sqrt(tf.square(X) + tf.square(Y))

        otf_mask = tf.sigmoid(tf.multiply(self.cutoff_freq - rdist, self.slop))
        otf_mask = tf.expand_dims(tf.expand_dims(otf_mask, 0), 0)
        otf_mask = tf.tile(otf_mask, (1, ch, 1, 1))
        otf_mask = tf.complex(otf_mask, tf.zeros_like(otf_mask))

        inputs = tf.complex(inputs, tf.zeros_like(inputs))
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
        fft_feature = tf.signal.fftshift(tf.signal.fft2d(inputs))
        output = tf.signal.ifft2d(tf.signal.fftshift(tf.multiply(otf_mask, fft_feature)))
        output = tf.transpose(output, [0, 2, 3, 1])
        output = tf.math.real(output)

        return output


def PS(I, r):
    _, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, [-1, a, b, c // (r ** 2), r, r])
    X = tf.transpose(X, (0, 1, 2, 5, 4, 3))  # [bsize,a,b,r,r,c/(r**2)]
    X = tf.split(X, a, axis=1)  # a x [bsize, b, r, r, c/(r**2)]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=2)  # [bsize, b, a*r, r, c/(r**2)]
    X = tf.split(X, b, axis=1)  # b x [bsize, a*r, r, c/(r**2)]
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=2)  # [bsize, a*r, b*r, c/(r**2)]
    return tf.reshape(X, [-1, a * r, b * r, c // (r ** 2)])
