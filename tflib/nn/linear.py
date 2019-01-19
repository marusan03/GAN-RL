import tflib as lib
from tflib.nn.sn import spectral_normalization

import numpy as np
import tensorflow as tf

_default_weightnorm = False


def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True


def disable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = False


_weights_stdev = None


def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev


def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None


def Linear(
        name,
        input_dim,
        output_dim,
        inputs,
        initializer=None,
        biases=True,
        pytorch_biases=False,
        initialization=None,
        weight_decay_scale=0.,
        spectral_norm=False,
        update_collection=None,
):
    """
    initialization: None, `lecun`, 'glorot', `he`, 'glorot_he', `orthogonal`, `("uniform", range)`
    """
    with tf.variable_scope(name):
        shape = None

        def uniform(stdev, shape):
            if _weights_stdev is not None:
                stdev = _weights_stdev
            return tf.random_uniform(
                shape=shape,
                dtype=tf.float32,
                minval=-1*stdev * tf.sqrt(3.),
                maxval=stdev * tf.sqrt(3.)
            )
        if initializer == None:

            if initialization == 'lecun':  # and input_dim != output_dim):
                # disabling orth. init for now because it's too slow
                weight_values = uniform(
                    np.sqrt(1./input_dim),
                    (input_dim, output_dim)
                )

            elif initialization == 'glorot' or (initialization == None):

                weight_values = uniform(
                    np.sqrt(2./(input_dim+output_dim)),
                    (input_dim, output_dim)
                )

            elif initialization == 'he':

                weight_values = uniform(
                    np.sqrt(2./input_dim),
                    (input_dim, output_dim)
                )

            elif initialization == 'glorot_he':

                weight_values = uniform(
                    np.sqrt(4./(input_dim+output_dim)),
                    (input_dim, output_dim)
                )

            elif initialization == 'orthogonal' or \
                    (initialization == None and input_dim == output_dim):

                # From lasagne
                def sample(shape):
                    if len(shape) < 2:
                        raise RuntimeError("Only shapes of length 2 or more are "
                                           "supported.")
                    flat_shape = (shape[0], np.prod(shape[1:]))
                    # TODO: why normal and not uniform?
                    a = np.random.normal(0.0, 1.0, flat_shape)
                    u, _, v = np.linalg.svd(a, full_matrices=False)
                    # pick the one with the correct shape
                    q = u if u.shape == flat_shape else v
                    q = q.reshape(shape)
                    return q.astype('float32')
                weight_values = sample((input_dim, output_dim))

            elif initialization[0] == 'uniform':

                weight_values = np.random.uniform(
                    low=-initialization[1],
                    high=initialization[1],
                    size=(input_dim, output_dim)
                ).astype('float32')

            elif initialization == 'pytorch':
                stdev = np.sqrt(1./input_dim)
                weight_values = tf.random_uniform(
                    shape=(input_dim, output_dim),
                    dtype=tf.float32,
                    minval=-1*stdev,
                    maxval=stdev
                )

            else:

                raise Exception('Invalid initialization!')

        else:
            weight_values = initializer
            shape = (input_dim, output_dim)

        # weight normarization
        regularizer = tf.contrib.layers.l2_regularizer(
            scale=weight_decay_scale)

        weight = tf.get_variable(
            'weights', shape=shape, initializer=weight_values, regularizer=regularizer)

        if spectral_norm:
            result = tf.matmul(inputs, spectral_normalization(
                weight, update_collection=update_collection))
        else:
            if inputs.get_shape().ndims == 2:
                result = tf.matmul(inputs, weight)
            else:
                reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
                result = tf.matmul(reshaped_inputs, weight)
                result = tf.reshape(result, tf.stack(
                    tf.unstack(tf.shape(inputs))[:-1] + [output_dim]))

        if biases:
            if pytorch_biases:
                k = 1.0 / input_dim
                _biases = tf.get_variable(
                    'biases', initializer=np.random.uniform(-np.sqrt(k), np.sqrt(k), output_dim).astype('float32'))
                result = tf.nn.bias_add(result, _biases)
            else:
                _biases = tf.get_variable(
                    'biases', initializer=np.zeros(output_dim, dtype='float32'))
                result = tf.nn.bias_add(result, _biases)

        return result
