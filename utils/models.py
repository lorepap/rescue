import tensorflow as tf
from tensorflow.keras.layers import Add, Conv2D, Input, Lambda
from tensorflow.keras.models import Model
import numpy as np


def srcnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=(None, None, 1)),
        tf.keras.layers.Conv2D(32, (1, 1), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(1, (5, 5), activation='linear', padding='same')
    ])
    return model


def vdsr():

    model = tf.keras.Sequential()
    
    model.add(
         tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                                   kernel_initializer=tf.keras.initializers.RandomNormal(stddev=np.sqrt(2.0/9/64)),
                                   bias_initializer='zeros',
                                   activation='relu')
    )

    # 18 additional convolutional layers
    for i in range(18):
        model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                                   kernel_initializer=tf.keras.initializers.RandomNormal(stddev=np.sqrt(2.0/9/64)),
                                   bias_initializer='zeros',
                                   activation='relu')
        )
    
    model.add(
        tf.keras.layers.Conv2D(1, (3, 3), strides=(1, 1), padding='same',
                               kernel_initializer=tf.keras.initializers.RandomNormal(stddev=np.sqrt(2.0/9/64)),
                               bias_initializer='zeros')
    )

    return model


def edsr(scale, input_depth=1, num_filters=64, num_res_blocks=8, res_block_scaling=None):

    def pixel_shuffle(scale):
        return lambda x: tf.nn.depth_to_space(x, scale)


    def res_block(x_in, filters, scaling):
        x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
        x = Conv2D(filters, 3, padding='same')(x)
        if scaling:
            x = Lambda(lambda t: t * scaling)(x)
        x = Add()([x_in, x])
        return x

    def upsample(x, scale, num_filters):
        upsample_counter = 0

        def upsample_1(x, factor, **kwargs):
            x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
            nonlocal upsample_counter
            upsample_counter += 1
            return Lambda(pixel_shuffle(scale=factor), name=f"pixel_shuffle_{upsample_counter}")(x)

        if scale == 2:
            x = upsample_1(x, 2, name='conv2d_1_scale_2')
        elif scale == 3:
            x = upsample_1(x, 3, name='conv2d_1_scale_3')
        elif scale == 4:
            x = upsample_1(x, 2, name='conv2d_1_scale_2')
            x = upsample_1(x, 2, name='conv2d_2_scale_2')
        elif scale == 6:
            x = upsample_1(x, 2, name='conv2d_1_scale_2')
            x = upsample_1(x, 3, name='conv2d_1_scale_3')
        elif scale == 8:
            x = upsample_1(x, 2, name='conv2d_1_scale_2')
            x = upsample_1(x, 2, name='conv2d_2_scale_2')
            x = upsample_1(x, 2, name='conv2d_3_scale_2')
        return x

    x_in = Input(shape=(None, None, input_depth))
    # x = Lambda(normalize)(x_in)

    x = b = Conv2D(num_filters, 3, padding='same')(x_in)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = upsample(x, scale, num_filters)
    # x = Conv2D(3, 3, padding='same')(x)
    x = Conv2D(input_depth, 3, padding='same')(x)

    # x = Lambda(denormalize)(x)
    #return keras.Model(x_in, x, name="edsr")
    return Model(x_in, x, name="edsr")
