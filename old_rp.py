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
        self.rp_weight_decay = self.config.rp_weight_decay

        self.initializer = tf.random_normal_initializer(0.0, 0.02)
        # self.initializer = None
        self.beta_initializer = tf.zeros_initializer()
        self.gamma_initializer = tf.truncated_normal_initializer(1.0, 0.02)

        self.concat_dim = 1

        self.action = tf.placeholder(
            tf.int32, shape=[None, self.lookahead], name='actions')
        self.reward = tf.placeholder(
            tf.int32, shape=[None, self.lookahead], name='rewards')
        self.state = tf.placeholder(
            tf.float32, shape=[None, self.history_length, self.state_width, self.state_height], name='state')
        self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

        if self.data_format == 'NHWC':
            self.concat_dim = 3
            self.state = tf.transpose(
                self.state, (0, 2, 3, 1), name='NCHW_to_NHWC')

        with tf.variable_scope('RP'):
            self.predicted_reward = self.build_rp(self.state, self.action)

        with tf.name_scope('opt'):
            self.rp_train_op, self.rp_summary = self.build_training_op(
                self.state, self.reward)

    def get_reward(self, state, action):
        predicted_reward = self.sess.run(self.predicted_reward, feed_dict={
            self.state: norm_state_Q_GAN(state), self.action: action})
        return predicted_reward

    def train(self, state, action, reward):
        _, rp_summary = self.sess.run([self.rp_train_op, self.rp_summary], feed_dict={
            self.state: norm_state_Q_GAN(state), self.action: action, self.reward: reward})
        return rp_summary

    def build_rp(self, state, action):

        output = lib.nn.conv2d.Conv2D(
            'RP_Conv.1', self.history_length+self.lookahead-1, 32, 8, state, initializer=self.initializer, weight_decay_scale=self.rp_weight_decay, stride=4, pytorch_biases=True, padding='VALID', data_format=self.data_format)
        output = tf.layers.batch_normalization(
            output, momentum=0.9, epsilon=1e-05, beta_initializer=self.beta_initializer, gamma_initializer=self.gamma_initializer, gamma_regularizer=tf.contrib.layers.l2_regularizer(scale=self.rp_weight_decay), training=self.is_training, name='BN1')
        output = tf.nn.leaky_relu(output, 0.1, name='ralu1')
        # (None, 20, 20, 32)

        output = lib.nn.conv2d.Conv2D(
            'RP_Conv.2', 32, 64, 4, output, initializer=self.initializer, weight_decay_scale=self.rp_weight_decay, stride=2, padding='VALID', pytorch_biases=True, data_format=self.data_format)
        output = tf.layers.batch_normalization(
            output, momentum=0.9, epsilon=1e-05, beta_initializer=self.beta_initializer, gamma_initializer=self.gamma_initializer, gamma_regularizer=tf.contrib.layers.l2_regularizer(scale=self.rp_weight_decay), training=self.is_training, name='BN2')
        output = tf.nn.leaky_relu(output, 0.1, name='ralu2')
        # (None, 9, 9, 64)

        output = lib.nn.conv2d.Conv2D(
            'RP_Conv.3', 64, 64, 3, output, initializer=self.initializer, weight_decay_scale=self.rp_weight_decay, stride=1, padding='VALID', pytorch_biases=True, data_format=self.data_format)
        output = tf.layers.batch_normalization(
            output, momentum=0.9, epsilon=1e-05, beta_initializer=self.beta_initializer, gamma_initializer=self.gamma_initializer, gamma_regularizer=tf.contrib.layers.l2_regularizer(scale=self.rp_weight_decay), training=self.is_training, name='BN3')
        # (None, 7, 7, 64)

        output = tf.layers.flatten(output, name='RP_Flatten')
        # (None, 6272)

        output = lib.nn.linear.Linear(
            'RP_Dence.1', 3136, 20, output, initialization='pytorch', weight_decay_scale=self.rp_weight_decay, pytorch_biases=True)
        # (None, 512)

        action_one_hot = tf.one_hot(
            action, self.num_actions, name='action_one_hot')

        action_one_hot = tf.layers.flatten(
            action_one_hot, name='action_one_hot_flatten')
        output = tf.nn.relu(output, name='ralu4')

        output = tf.concat([output, action_one_hot], self.concat_dim)
        # (None, 512+num_actions*lookahead)

        output = lib.nn.linear.Linear(
            'RP_Dence.2', 20+self.num_actions*self.lookahead, self.num_rewards*self.lookahead, output, initialization='pytorch', weight_decay_scale=self.rp_weight_decay, pytorch_biases=True)
        # (None, 3*lookahead)

        return output

    def build_training_op(self, state, reward):
        loss = tf.constant(0., dtype=tf.float32)
        for ind in range(self.lookahead):
            outputs = self.predicted_reward[
                :, self.num_rewards*ind: self.num_rewards*(ind + 1)]
            loss = loss + tf.losses.sparse_softmax_cross_entropy(
                labels=reward[:, ind], logits=outputs)

        with tf.name_scope('weight_decay'):
            rp_weight_decay = tf.losses.get_regularization_loss(
                scope='RP', name='rp_weight_decay')

        loss += rp_weight_decay

        rp_summary = tf.summary.scalar('rp_loss', loss)

        rp_train_op = tf.train.AdamOptimizer(
            learning_rate=2e-4, beta1=0.5, beta2=0.999, name='rp_adam').minimize(loss)

        return rp_train_op, rp_summary
