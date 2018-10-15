import tensorflow as tf
import tflib as lib
import tflib.nn.linear
import tflib.nn.conv2d

import numpy as np


class Reward_Predictor():

    def __init__(self):
        pass

    def model(self, inputs, action, is_training=False, num_actions=18, lookahead=1, ngf=32):

        self.output = lib.nn.conv2d.Conv2D(
            'RP_Conv.1', 4 + lookahead - 1, ngf, 8, inputs, stride=4, padding='VALID')
        self.output = tf.layers.batch_normalization(
            self.output, momentum=0.9, epsilon=1e-05, training=is_training, name='RP_BN1')
        self.output = tf.nn.leaky_relu(self.output, -0.1)
        # (None, 20, 20, 32)

        self.output = lib.nn.conv2d.Conv2D(
            'RP_Conv.2', ngf, ngf * 2, 4, self.output, stride=2, padding='VALID')
        self.output = tf.layers.batch_normalization(
            self.output, momentum=0.9, epsilon=1e-05, training=is_training, name='RP_BN2')
        self.output = tf.nn.leaky_relu(self.output, -0.1)
        # (None, 9, 9, 64)

        self.output = lib.nn.conv2d.Conv2D(
            'RP_Conv.3', ngf * 2, ngf * 2, 3, self.output, stride=1, padding='VALID')
        self.output = tf.layers.batch_normalization(
            self.output, momentum=0.9, epsilon=1e-05, training=is_training, name='RP_BN3')
        # (None, 7, 7, 64)

        self.output = tf.reshape(
            self.output, [self.output.get_shape()[0], -1])
        # (None, 3136)

        self.output = lib.nn.linear.Linear(
            'RP_Dence.1', 3136, 20, self.output)
        self.output = tf.nn.relu(self.output)
        # (None, 20)

        self.output = tf.concat([self.output, action], 1)
        # (None, 20+num_actions*lookahead)

        self.output = lib.nn.linear.Linear(
            'RP_Dence.2', 20 + num_actions*lookahead, 3*lookahead, self.output)
        # (None, 3*lookahead)

        return self.output


if __name__ == '__main__':
    inputs = tf.placeholder(tf.float32, shape=[100, 84, 84, 4])
    is_training = tf.placeholder_with_default(False, [])
    action = tf.placeholder(tf.float32, shape=[100, 5])
    act = np.random.randint(0, 2, (100, 5))
    with tf.variable_scope("a"):
        a = Reward_Predictor().model(inputs, action, is_training, num_actions=5)
    with tf.variable_scope("a", reuse=True):
        b = Reward_Predictor().model(inputs, action, is_training, num_actions=5)
    test = 2 * (np.random.rand(100, 84, 84, 4) - 0.5)

    with tf.Session() as session:

        with tf.summary.FileWriter('./log/', session.graph):

            session.run(tf.global_variables_initializer())
            result = session.run(
                a, feed_dict={inputs: test, is_training: True, action: act})
            print(result.shape)
            print('\n'.join([v.name for v in tf.global_variables()]))
