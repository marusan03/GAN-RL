import tflib as lib

import numpy as np
import tensorflow as tf


def Deconv2D(
    name,
    input_dim,
    output_dim,
    filter_size,
    inputs,
    initializer=None,
    he_init=True,
    stride=2,
    weight_decay_scale=None,
    biases=True,
    padding_size=0,
    padding='SAME',
    data_format='NCHW'

):
    """
    inputs: tensor of shape (batch size, height, width, input_dim)
    returns: tensor of shape (batch size, 2*height, 2*width, output_dim)
    """
    with tf.variable_scope(name):
        shape = None

        def uniform(stdev, shape):
            return tf.random_uniform(
                shape=shape,
                dtype=tf.float32,
                minval=-stdev * tf.sqrt(3.),
                maxval=stdev * tf.sqrt(3.)
            )

        fan_in = input_dim * filter_size**2 / (stride**2)
        fan_out = output_dim * filter_size ** 2

        if initializer == None:

            if he_init:
                filters_stdev = np.sqrt(4./(fan_in+fan_out))
            else:  # Normalized init (Glorot & Bengio)
                filters_stdev = np.sqrt(2./(fan_in+fan_out))

            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, output_dim, input_dim)
            )
        else:
            filter_values = initializer
            shape = (filter_size, filter_size, output_dim, input_dim)

        # weight normarization
        if weight_decay_scale != None:
            regularizer = tf.contrib.layers.l2_regularizer(
                scale=weight_decay_scale)
        else:
            regularizer = None

        filters = tf.get_variable(
            'filters', shape=shape, initializer=filter_values, regularizer=regularizer)

        # calculated output dimension
        if data_format == 'NHWC':
            input_shape = inputs.shape.as_list()
            output_hgiht = (input_shape[1] - 1)*stride - \
                2 * padding_size + filter_size
            output_width = (input_shape[2] - 1)*stride - \
                2 * padding_size + filter_size
            output_shape = tf.stack(
                [tf.shape(inputs)[0], output_hgiht, output_width, output_dim], name='output_shape')
        else:
            input_shape = inputs.shape.as_list()
            output_hgiht = (input_shape[2] - 1)*stride - \
                2 * padding_size + filter_size
            output_width = (input_shape[3] - 1)*stride - \
                2 * padding_size + filter_size
            output_shape = tf.stack(
                [tf.shape(inputs)[0], output_dim, output_hgiht, output_width], name='output_shape')

        # stride
        strides = []
        if data_format == 'NHWC':
            strides = [1, stride, stride, 1]
        else:
            strides = [1, 1, stride, stride]

        result = tf.nn.conv2d_transpose(
            value=inputs,
            filter=filters,
            output_shape=output_shape,
            strides=strides,
            padding=padding,
            data_format=data_format
        )

        if biases:
            _biases = tf.get_variable(
                'biases', initializer=np.zeros(output_dim, dtype='float32'))
            result = tf.nn.bias_add(result, _biases, data_format=data_format)

        return result
