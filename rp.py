import tensorflow as tf
import tflib as lib
import tflib.nn.linear
import tflib.nn.conv2d

import numpy as np


def norm_state_Q_GAN(state):
    return np.clip(state, -1*127.5/130., 127.5/130.)


class RP():

    def __init__(self, session, config, num_actions=18):
        self.sess = session
        self.config = config
        self.num_actions = num_actions
        self.lookahead = self.config.lookahead
        self.num_rewards = self.config.num_rewards
        self.history_length = self.config.history_length
        self.state_width = self.config.screen_width
        self.state_height = self.config.screen_height
        self.data_format = self.config.cnn_format

        self.concat_dim = 1

        self.action = tf.placeholder(
            tf.int32, shape=[None, self.lookahead+1], name='actions')
        self.reward = tf.placeholder(
            tf.int32, shape=[None, self.lookahead+1, self.num_rewards], name='rewards')
        self.state = tf.placeholder(
            tf.float32, shape=[None, self.history_length + self.lookahead, self.state_width, self.state_height], name='state')

        if self.data_format == 'NHWC':
            self.concat_dim = 3
            self.state = tf.transpose(
                self.state, (0, 2, 3, 1), name='NCHW_to_NHWC')

        with tf.variable_scope('RP'):
            self.predicted_reward = self.build_rp(self.state, self.action)

        with tf.name_scope('opt'):
            self.rp_train_op, self.rp_summary, self.cross_entropy_loss, self.loss = self.build_training_op(
                self.state, self.action, self.reward)

    def get_reward(self, state, action):
        predicted_reward = self.sess.run(self.predicted_reward, feed_dict={
            self.state: norm_state_Q_GAN(state), self.action: action})
        return predicted_reward

    def train(self, state, action, reward):
        _, rp_summary, cross_entropy_loss, loss = self.sess.run([self.rp_train_op, self.rp_summary, self.cross_entropy_loss, self.loss], feed_dict={
            self.state: norm_state_Q_GAN(state), self.action: action, self.reward: reward})
        print(cross_entropy_loss, loss)
        return rp_summary

    def build_rp(self, state, action):

        output = lib.nn.conv2d.Conv2D(
            'RP_Conv.1', self.history_length+self.lookahead, 32, 8, state, weight_norm_scale=0.0001, stride=4, padding='VALID', data_format=self.data_format)
        output = tf.nn.leaky_relu(output, -0.1)
        # (None, 20, 20, 32)

        output = lib.nn.conv2d.Conv2D(
            'RP_Conv.2', 32, 64, 4, output, weight_norm_scale=0.0001, stride=2, padding='VALID', data_format=self.data_format)
        output = tf.nn.leaky_relu(output, -0.1)
        # (None, 9, 9, 64)

        output = lib.nn.conv2d.Conv2D(
            'RP_Conv.3', 64, 128, 3, output, weight_norm_scale=0.0001, stride=1, padding='VALID', data_format=self.data_format)
        # (None, 7, 7, 128)

        output = tf.layers.flatten(output, name='RP_Flatten')
        # (None, 6272)

        output = lib.nn.linear.Linear(
            'RP_Dence.1', 6272, 512, output, weight_norm_scale=0.0001)
        output = tf.nn.relu(output)
        # (None, 512)

        action_one_hot = tf.one_hot(
            action, self.num_actions, name='action_one_hot')

        action_one_hot = tf.layers.flatten(
            action_one_hot, name='action_one_hot_flatten')

        output = tf.concat([output, action_one_hot], self.concat_dim)
        # (None, 512+num_actions*lookahead)

        output = lib.nn.linear.Linear(
            'RP_Dence.2', 512+self.num_actions*(self.lookahead+1), self.num_rewards*(self.lookahead+1), output, weight_norm_scale=0.0001)
        # (None, 3*lookahead)

        return output

    def build_training_op(self, state, action, reward):
        cross_entropy_loss = 0.
        for ind in range(self.lookahead + 1):
            outputs = self.predicted_reward[:,
                                            self.num_rewards*ind: self.num_rewards*(ind + 1)]
            cross_entropy_loss = cross_entropy_loss + tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=reward[:, ind, 0], logits=outputs))

        with tf.name_scope('weight_decay'):
            rp_weight_decay = tf.losses.get_regularization_loss(
                scope='RP', name='rp_weight_decay')

        loss = rp_weight_decay - cross_entropy_loss

        rp_summary = tf.summary.scalar('rp_loss', loss)

        rp_train_op = tf.train.AdamOptimizer(
            learning_rate=2e-4, beta1=0.5, beta2=0.999, name='rp_adam').minimize(loss)

        return rp_train_op, rp_summary, cross_entropy_loss, loss
