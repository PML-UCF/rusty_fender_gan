# ______          _           _     _ _ _     _   _
# | ___ \        | |         | |   (_) (_)   | | (_)
# | |_/ / __ ___ | |__   __ _| |__  _| |_ ___| |_ _  ___
# |  __/ '__/ _ \| '_ \ / _` | '_ \| | | / __| __| |/ __|
# | |  | | | (_) | |_) | (_| | |_) | | | \__ \ |_| | (__
# \_|  |_|  \___/|_.__/ \__,_|_.__/|_|_|_|___/\__|_|\___|
# ___  ___          _                 _
# |  \/  |         | |               (_)
# | .  . | ___  ___| |__   __ _ _ __  _  ___ ___
# | |\/| |/ _ \/ __| '_ \ / _` | '_ \| |/ __/ __|
# | |  | |  __/ (__| | | | (_| | | | | | (__\__ \
# \_|  |_/\___|\___|_| |_|\__,_|_| |_|_|\___|___/
#  _           _                     _
# | |         | |                   | |
# | |     __ _| |__   ___  _ __ __ _| |_ ___  _ __ _   _
# | |    / _` | '_ \ / _ \| '__/ _` | __/ _ \| '__| | | |
# | |___| (_| | |_) | (_) | | | (_| | || (_) | |  | |_| |
# \_____/\__,_|_.__/ \___/|_|  \__,_|\__\___/|_|   \__, |
#                                                   __/ |
#                                                  |___/
#
# MIT License
#
# Copyright (c) 2022 Probabilistic Mechanics Laboratory
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def rusty_texture_network_generator(input_channels=1, output_channels=3):
    inputs = tf.keras.layers.Input(shape=[256, 32, input_channels])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 16, 64)
        downsample(128, 4),  # (batch_size, 64, 8, 128)
        downsample(256, 4),  # (batch_size, 32, 4, 256)
        downsample(512, 4),  # (batch_size, 16, 2, 512)
        downsample(512, 4),  # (batch_size, 8, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 16, 2, 1024)
        upsample(256, 4),  # (batch_size, 32, 4, 512)
        upsample(128, 4),  # (batch_size, 64, 8, 256)
        upsample(64, 4),  # (batch_size, 128, 16, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(output_channels, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 256, 32, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def rusty_texture_network_discriminator(n_channels=1):

    inputs = layers.Input(shape=[256, 32, n_channels])

    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)

    output = layers.Conv2D(1, (3, 3), strides=(2, 2), padding='same')(x)

    return tf.keras.Model(inputs=inputs, outputs=output)


def rusty_level_network_generator(input_channels=1, output_channels=1, batch_size=48):
    height = 256
    width = 32
    noise_n = 8 * 512

    noise_width = np.round(noise_n / (height * input_channels)).astype(int)
    total_width = width + noise_width

    inputs = tf.keras.layers.Input(shape=[height, total_width, input_channels])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 16, 64)
        downsample(128, 4),  # (batch_size, 64, 8, 128)
        downsample(256, 4),  # (batch_size, 32, 4, 256)
        downsample(512, 4),  # (batch_size, 16, 2, 512)
        downsample(512, 4),  # (batch_size, 8, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 16, 2, 1024)
        upsample(256, 4),  # (batch_size, 32, 4, 512)
        upsample(128, 4),  # (batch_size, 64, 8, 256)
        upsample(64, 4),  # (batch_size, 128, 16, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(output_channels, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 256, 32, 3)

    map_inputs = inputs[:, :, :width, :]
    noise = inputs[:, :, width:, :]

    x = map_inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    noise = tf.reshape(noise, (batch_size, 8, 1, 512))
    x = tf.keras.layers.Add()([x, noise])
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def rusty_level_network_discriminator(n_channels=1):

    inputs = layers.Input(shape=[256, 32, n_channels])

    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(inputs)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(512)(x)
    x = layers.Dense(256)(x)
    x = layers.Dense(128)(x)

    output = layers.Dense(1, activation='sigmoid')(x)

    return tf.keras.Model(inputs=inputs, outputs=output)
