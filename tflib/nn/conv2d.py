import tflib as lib
from tflib.nn.sn import spectral_normalization

import numpy as np
import tensorflow as tf


_default_weightnorm = False
NO_OPS = 'NO_OPS'


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


def Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=True, mask_type=None, stride=1, weight_norm_scale=0., spectral_norm=None, update_collection=None, biases=True, gain=1., padding_size=0, padding='SAME'):
    """
    inputs: tensor of shape (batch size, height, width, num channels)
    mask_type: one of None, 'a', 'b'

    returns: tensor of shape (batch size, height, width, num channels)
    """
    with tf.variable_scope(name):

        if mask_type is not None:
            mask_type, mask_n_channels = mask_type

            mask = np.ones(
                (filter_size, filter_size, input_dim, output_dim),
                dtype='float32'
            )
            center = filter_size // 2

            # Mask out future locations
            # filter shape is (height, width, input channels, output channels)
            mask[center+1:, :, :, :] = 0.
            mask[center, center+1:, :, :] = 0.

            # Mask out future channels
            for i in range(mask_n_channels):
                for j in range(mask_n_channels):
                    if (mask_type == 'a' and i >= j) or (mask_type == 'b' and i > j):
                        mask[
                            center,
                            center,
                            i::mask_n_channels,
                            j::mask_n_channels
                        ] = 0.

        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev,
                high=stdev,
                size=size
            ).astype('float32')

        fan_in = input_dim * filter_size**2
        fan_out = output_dim * filter_size**2 / (stride**2)

        if mask_type is not None:  # only approximately correct
            fan_in /= 2.
            fan_out /= 2.

        if he_init:
            filters_stdev = np.sqrt(2./fan_in)
        else:  # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(6./(fan_in+fan_out))

        if _weights_stdev is not None:
            filter_values = uniform(
                _weights_stdev,
                (filter_size, filter_size, input_dim, output_dim)
            )
        else:
            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, input_dim, output_dim)
            )

        # print "WARNING IGNORING GAIN"
        filter_values *= gain

        # weight normarization
        regularizer = tf.contrib.layers.l2_regularizer(
            scale=weight_norm_scale)

        filters = tf.get_variable(
            'filters', initializer=filter_values, regularizer=regularizer)

        if mask_type is not None:
            with tf.name_scope('filter_mask'):
                filters = filters * mask

        if padding_size > 0:
            inputs = tf.pad(
                inputs, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]])
            padding = 'VALID'

        if spectral_norm:
            result = tf.nn.conv2d(
                input=inputs,
                filter=spectral_normalization(
                    filters, update_collection=update_collection),
                strides=[1, stride, stride, 1],
                padding=padding
            )
        else:
            result = tf.nn.conv2d(
                input=inputs,
                filter=filters,
                strides=[1, stride, stride, 1],
                padding=padding
            )

        if biases:
            _biases = tf.get_variable(
                'biases', initializer=np.zeros(output_dim, dtype='float32'))

            result = tf.nn.bias_add(result, _biases)

        return result
