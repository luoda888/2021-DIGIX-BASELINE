# -*- coding=utf-8 -*-

import tensorflow as tf

from recognizer.models.crnn import densenet_crnn_time

K = tf.keras.backend
Lambda = tf.keras.layers.Lambda
Input = tf.keras.layers.Input


def ctc_lambda_func(args):
    prediction, label, input_length, label_length = args
    return K.ctc_batch_cost(label, prediction, input_length, label_length)


def crnn_model_based_on_densenet_crnn_time_softmax_activate(initial_learning_rate=0.0005):
    shape = (32, 280, 3)
    inputs = tf.keras.layers.Input(shape=shape, name='input_data')
    output = densenet_crnn_time(inputs=inputs, activation='softmax')
    model_body = tf.keras.models.Model(inputs, output)
    model_body.summary()

    label = Input(name='label', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([output, label, input_length, label_length])

    model = tf.keras.models.Model(inputs=[inputs, label, input_length, label_length], outputs=loss_out)
    model.compile(loss={'ctc': lambda y_true, prediction: prediction},
                  optimizer=tf.keras.optimizers.Adam(initial_learning_rate), metrics=['accuracy'])
    return model_body, model
