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
    initializer=None,
    he_init=False,
    stride=1,
    weight_decay_scale=0.,
    spectral_norm=None,
    update_collection=None,
    pytorch=False,
    biases=True,
    pytorch_biases=False,
    padding_size=0,
    padding='SAME',
    data_format='NCHW'
):

    with tf.variable_scope(name):
        shape = None

        def uniform(stdev, shape):
            return tf.random_uniform(
                shape=shape,
                dtype=tf.float32,
                minval=-stdev,
                maxval=stdev
            )

        fan_in = input_dim * filter_size**2
        fan_out = output_dim * filter_size**2 / (stride**2)

        if initializer == None:
            if he_init:
                filters_stdev = np.sqrt(12./(fan_in+fan_out))
            elif pytorch:
                filters_stdev = np.sqrt(1.0 / fan_in)
            else:  # Normalized init (Glorot & Bengio)
                filters_stdev = np.sqrt(6./(fan_in+fan_out))

            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, input_dim, output_dim)
            )

        else:
            filter_values = initializer
            shape = (filter_size, filter_size, input_dim, output_dim)

        # weight normarization
        if weight_decay_scale != 0.:
            regularizer = tf.contrib.layers.l2_regularizer(
                scale=weight_decay_scale)
        else:
            regularizer = None

        filters = tf.get_variable(
            'filters', shape=shape, initializer=filter_values, regularizer=regularizer)

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
            if pytorch_biases:
                k = 1.0 / input_dim * filter_size * filter_size
                _biases = tf.get_variable(
                    'biases', initializer=np.random.uniform(-np.sqrt(k), np.sqrt(k), output_dim).astype('float32'))
                result = tf.nn.bias_add(
                    result, _biases, data_format=data_format)
            else:
                _biases = tf.get_variable(
                    'biases', initializer=np.zeros(output_dim, dtype='float32'))
                result = tf.nn.bias_add(
                    result, _biases, data_format=data_format)

        return result
