import random
import pprint

import tensorflow as tf
import numpy as np

from train import train
from config import get_config

pp = pprint.PrettyPrinter().pprint
flags = tf.app.flags

FLAGS = flags.FLAGS

# Model
flags.DEFINE_string('model', 'm1', 'Type of model')
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

# GDM
flags.DEFINE_integer('lookahead', 1, 'The number of lookahead. int[1-4]')

# Environment
flags.DEFINE_string('env_name', 'PongNoFrameskip-v3',
                    'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 4, 'The number of action to be repeated')

# Etc
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/1',
                    'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean(
    'display', False, 'Whether to do display the game screen or not')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
    raise ValueError("--gpu_fraction should be defined")


def calc_gpu_fraction(fraction_string):
    idx, num = fraction_string.split('/')
    idx, num = float(idx), float(num)

    fraction = 1 / (num - idx + 1)
    print(" [*] GPU : %.4f" % fraction)
    return fraction


def main(_):

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        config = get_config(FLAGS) or FLAGS

        if not tf.test.is_gpu_available() and FLAGS.use_gpu:
            raise Exception("use_gpu flag is true when no GPUs are available")

        if FLAGS.use_gpu:
            config.cnn_format = 'NHWC'

        pp(config.__dict__)

        if FLAGS.is_train:
            train(sess, config)
        else:
            play(sess, config)


if __name__ == '__main__':
    tf.app.run()
