# -*- coding=utf-8 -*-


import tensorflow as tf

from recognizer.tools.config import config

Conv2D = tf.keras.layers.Conv2D
Dense = tf.keras.layers.Dense
TimeDistributed = tf.keras.layers.TimeDistributed
Flatten = tf.keras.layers.Flatten
Reshape = tf.keras.layers.Reshape


def densenet_crnn_time(inputs, activation=None, include_top=False):
    densenet = tf.keras.applications.DenseNet121(input_tensor=inputs, include_top=include_top, weights=None)
    x = Conv2D(filters=512, kernel_size=(5, 5), strides=(2, 1), padding='same')(densenet.layers[50].output)
    x = Reshape((280, 1, 512), name='reshape')(x)
    x = TimeDistributed(Flatten(), name='flatten')(x)
    x = Dense(config.num_class, name='output', activation=activation)(x)
    return x
