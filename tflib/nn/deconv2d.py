import tflib as lib

import numpy as np
import tensorflow as tf

_default_weightnorm = False


def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True


_weights_stdev = None


def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev


def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None


def Deconv2D(
    name,
    input_dim,
    output_dim,
    filter_size,
    inputs,
    he_init=True,
    mask_type=None,
    stride=2,
    weight_norm_scale=0.,
    biases=True,
    gain=1.,
    padding_size=0,
    padding='SAME'

):
    """
    inputs: tensor of shape (batch size, height, width, input_dim)
    returns: tensor of shape (batch size, 2*height, 2*width, output_dim)
    """
    with tf.variable_scope(name):

        if mask_type != None:
            raise Exception('Unsupported configuration')

        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev,
                high=stdev,
                size=size
            ).astype('float32')

        fan_in = input_dim * filter_size**2 / (stride**2)
        fan_out = output_dim * filter_size**2

        if he_init:
            filters_stdev = np.sqrt(2./fan_out)
        else:  # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(6./(fan_in+fan_out))

        if _weights_stdev is not None:
            filter_values = uniform(
                _weights_stdev,
                (filter_size, filter_size, output_dim, input_dim)
            )
        else:
            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, output_dim, input_dim)
            )

        filter_values *= gain

        # weight normarization
        regularizer = tf.contrib.layers.l2_regularizer(
            scale=weight_norm_scale)

        filters = tf.get_variable(
            'filters', initializer=filter_values, regularizer=regularizer)

        # calculated output dimension
        input_shape = inputs.shape.as_list()
        output_hgiht = (input_shape[1] - 1)*stride - \
            2 * padding_size + filter_size
        output_width = (input_shape[2] - 1)*stride - \
            2 * padding_size + filter_size
        output_shape = tf.stack(
            [tf.shape(inputs)[0], output_hgiht, output_width, output_dim], name='output_shape')

        # if padding_size > 0:
        #     padding_size = filter_size - 1 - padding_size
        #     inputs = tf.pad(
        #         inputs, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]])
        #     padding = 'VALID'

        result = tf.nn.conv2d_transpose(
            value=inputs,
            filter=filters,
            output_shape=output_shape,
            strides=[1, stride, stride, 1],
            padding=padding
        )

        if biases:
            _biases = tf.get_variable(
                'biases', initializer=np.zeros(output_dim, dtype='float32'))
            result = tf.nn.bias_add(result, _biases)

        return result
