import numpy as np
import tensorflow as tf

import tflib as lib
from tflib.nn.sn import spectral_normalization

NO_OPS = 'NO_OPS'


def Conv2D(
    name,
    input_dim,
    output_dim,
    filter_size,
    inputs,
    he_init=True,
    stride=1,
    weight_norm_scale=0.,
    spectral_norm=None,
    update_collection=None,
    biases=True,
    padding_size=0,
    padding='SAME',
    data_format='NCHW'
):

    with tf.variable_scope(name):

        def uniform(stdev, shape):
            return tf.random_uniform(
                shape=shape,
                dtype=tf.float32,
                minval=-stdev * tf.sqrt(3),
                maxval=stdev * tf.sqrt(3)
            )

        fan_in = input_dim * filter_size**2
        fan_out = output_dim * filter_size**2 / (stride**2)

        if he_init:
            filters_stdev = np.sqrt(4./(fan_in+fan_out))
        else:  # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2./(fan_in+fan_out))

        filter_values = uniform(
            filters_stdev,
            (filter_size, filter_size, input_dim, output_dim)
        )

        # weight normarization
        regularizer = tf.contrib.layers.l2_regularizer(
            scale=weight_norm_scale)

        filters = tf.get_variable(
            'filters', initializer=filter_values, regularizer=regularizer)

        # stride
        strides = []
        if data_format == 'NHWC':
            strides = [1, stride, stride, 1]
        else:
            strides = [1, 1, stride, stride]

        # padding
        if padding_size > 0:
            if data_format == 'NHWC':
                inputs = tf.pad(
                    inputs, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]])
            else:
                inputs = tf.pad(
                    inputs, [[0, 0], [0, 0], [padding_size, padding_size], [padding_size, padding_size]])
            padding = 'VALID'

        if spectral_norm:
            result = tf.nn.conv2d(
                input=inputs,
                filter=spectral_normalization(
                    filters, update_collection=update_collection),
                strides=strides,
                padding=padding,
                data_format=data_format
            )
        else:
            result = tf.nn.conv2d(
                input=inputs,
                filter=filters,
                strides=strides,
                padding=padding,
                data_format=data_format
            )

        if biases:
            _biases = tf.get_variable(
                'biases', initializer=np.zeros(output_dim, dtype='float32'))

            result = tf.nn.bias_add(result, _biases, data_format=data_format)

        return result
