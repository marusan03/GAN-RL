import random

import tensorflow as tf
import tensorflow.layers
import numpy as np

import tflib as lib
import tflib.nn.linear
import tflib.nn.conv2d


class Agent():

    def __init__(self, sess, config, num_actions=18, action_interval=4):
        self.sess = sess
        self.config = config
        self.num_actions = num_actions
        self.discount = self.config.discount
        self.history_length = self.config.history_length
        self.screen_width = self.config.screen_width
        self.screen_height = self.config.screen_height
        self.learning_rate_minimum = self.config.learning_rate_minimum
        self.learning_rate = self.config.learning_rate
        self.learning_rate_decay_step = self.config.learning_rate_decay_step
        self.learning_rate_decay = self.config.learning_rate_decay
        self.data_format = self.config.cnn_format

        self.s_t = tf.placeholder(tf.float32, shape=(
            None, self.history_length, self.screen_width, self.screen_height))
        self.s_t_plas_1 = tf.placeholder(tf.float32, shape=(
            None, self.history_length, self.screen_width, self.screen_height))

        if self.data_format == 'NHWC':
            self.s_t = tf.transpose(
                self.s_t, (0, 2, 3, 1), name='NCHW_to_NHWC')
            self.s_t_plas_1 = tf.transpose(
                self.s_t_plas_1, (0, 2, 3, 1), name='NCHW_to_NHWC')

        with tf.variable_scope('dqn'):
            self.q_value, self.q_action = self.build_model(self.s_t)
        with tf.variable_scope('target_network'):
            self.target_q_value, _ = self.build_model(self.s_t_plas_1)
        with tf.name_scope('update_target_q_network'):
            self.update_target_q_network_op = self.copy_weight()
        with tf.name_scope('dqn_op'):
            self.dqn_op, self.loss, self.dqn_summary = self.build_training_op()

    def get_action(self, state):
        action = self.sess.run(self.q_action, feed_dict={self.s_t: state})[0]
        return action

    def get_q_value(self, state):
        q_value = self.sess.run(self.q_value,
                                feed_dict={self.s_t: state})
        return q_value

    def train(self, state, action, reward, next_state, terminal, step):
        terminal = np.array(terminal) + 0.
        max_q_t_plus_1 = np.max(self.sess.run(
            self.target_q_value, feed_dict={self.s_t_plas_1: next_state}), axis=1)

        target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward

        _, q_value, loss, dqn_summary = self.sess.run([self.dqn_op, self.q_value, self.loss, self.dqn_summary], feed_dict={
            self.s_t: state,
            self.action: action,
            self.target_q_t: target_q_t,
            self.learning_rate_step: step})

        return q_value, loss, dqn_summary

    def copy_weight(self):
        dqn_weights = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='dqn')
        target_q_network_weights = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_network')
        update_target_q_network_op = [target_q_network_weights[i].assign(
            dqn_weights[i]) for i in range(len(dqn_weights))]
        return update_target_q_network_op

    def updated_target_q_network(self):
        self.sess.run(self.update_target_q_network_op)

    def build_training_op(self):
        self.target_q_t = tf.placeholder(
            dtype=tf.float32, shape=[None], name='target_q_t')
        self.action = tf.placeholder(
            dtype=tf.uint8, shape=[None], name='action')

        action_one_hot = tf.one_hot(
            self.action, self.num_actions, name='action_one_hot')
        q_acted = tf.reduce_sum(self.q_value * action_one_hot,
                                axis=1, name='q_acted')

        delta = self.target_q_t - q_acted

        def clipped_error(x):
            # Huber loss
            return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

        # If you use RMSpropGraves, this code is tf.reduce_sum(). But it is not Implemented.
        loss = tf.reduce_mean(clipped_error(delta), name='loss')

        dqn_summary = tf.summary.scalar('dqn_loss', loss)

        self.learning_rate_step = tf.placeholder(
            'int64', None, name='learning_rate_step')
        learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                      tf.train.exponential_decay(
                                          self.learning_rate,
                                          self.learning_rate_step,
                                          self.learning_rate_decay_step,
                                          self.learning_rate_decay,
                                          staircase=True))

        dqn_op = tf.train.RMSPropOptimizer(
            learning_rate_op, momentum=0.95, epsilon=0.01).minimize(loss)
        return dqn_op, loss, dqn_summary

    def build_model(self, state):

        if self.data_format == 'NCHW':
            data_format = 'channels_first'
        else:
            data_format = 'channels_last'

        initializer = tf.truncated_normal_initializer(0, 0.02)

        output = tf.layers.conv2d(
            state, 32, 8, strides=4, padding='VALID', data_format=data_format, activation=tf.nn.relu, kernel_initializer=initializer, name='DQN_Conv.1')
        output = tf.nn.relu(output)
        # (None, 20, 20, 32)

        output = tf.layers.conv2d(
            output, 32 * 2, 4,  strides=2, padding='VALID', data_format=data_format, activation=tf.nn.relu, kernel_initializer=initializer, name='DQN_Conv.2')
        output = tf.nn.relu(output)
        # (None, 9, 9, 64)

        output = tf.layers.conv2d(
            output, 32 * 2, 3, strides=1, padding='VALID', data_format=data_format, activation=tf.nn.relu, kernel_initializer=initializer, name='DQN_Conv.3')
        output = tf.nn.relu(output)
        # (None, 7, 7, 64)

        output = tf.layers.flatten(output)
        # (None, 3136)

        dence_initializer = tf.random_normal_initializer(stddev=0.02)

        output = tf.layers.dense(
            output, 512, activation=tf.nn.relu, kernel_initializer=dence_initializer, name='DQN_Dence.1')
        output = tf.nn.relu(output)
        # (None, 512)

        q_value = tf.layers.dense(
            output, self.num_actions, kernel_initializer=dence_initializer, name='DQN_Dence.2')
        # (None, num_actions)

        q_action = tf.argmax(q_value, axis=1)

        return q_value, q_action
