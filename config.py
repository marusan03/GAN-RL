class AgentConfig(object):
    scale = 10000
    display = False

    max_step = 5000 * scale
    memory_size = 100 * scale

    batch_size = 32
    random_start = 30
    cnn_format = 'NCHW'
    discount = 0.99
    target_q_update_step = 1 * scale
    learning_rate = 0.00025
    learning_rate_minimum = 0.00025
    learning_rate_decay = 0.96
    learning_rate_decay_step = 5 * scale

    epsilon_end = 0.1
    epsilon_start = 1.
    epsilon_end_t = memory_size

    history_length = 4
    train_frequency = 4
    learn_start = 5. * scale

    min_delta = -1
    max_delta = 1

    double_q = False
    dueling = False

    _test_step = 5 * scale
    _save_step = _test_step * 10


class EnvironmentConfig(object):
    env_name = 'PongNoFrameskip-v3'

    screen_width = 84
    screen_height = 84
    max_reward = 1.
    min_reward = -1.

    global_step = 0


class GDMConfig(object):
    lookahead = 1
    lamda = 10.
    gdm_train_frequency = 16
    gats = True


class DQNConfig(AgentConfig, EnvironmentConfig):
    model = ''
    pass


class M1(DQNConfig, GDMConfig):
    backend = 'tf'
    env_type = 'detail'
    action_repeat = 4


def get_config(FLAGS):
    if FLAGS.model == 'm1':
        config = M1
    elif FLAGS.model == 'm2':
        # config = M2
        pass

    for key in FLAGS.flag_values_dict():
        if key == 'gpu':
            if FLAGS[key].value == False:
                config.cnn_format = 'NHWC'
            else:
                config.cnn_format = 'NCHW'

        if hasattr(config, key):
            setattr(config, key, FLAGS[key].value)

    return config
