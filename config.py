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

    num_actions = 18


class EnvironmentConfig(object):
    env_name = 'PongNoFrameskip-v4'

    screen_width = 84
    screen_height = 84
    max_reward = 1.
    min_reward = -1.

    global_step = 0


class GDMConfig(object):
    gan_memory_size = 10000
    gan_learn_start = 10000
    gan_dqn_learn_start = 200000
    gan_batch_size = 128
    rp_batch_size = 128
    lookahead = 1
    num_rewards = 3
    lamda = 10.
    lambda_l1 = 20.
    lambda_l2 = 80.
    gdm_ngf = 24
    disc_ngf = 24
    gdm_weight_norm_scale = 1e-3
    disc_weight_norm_scale = 0.1
    gats = True
    rollout_frequency = 50000
    gan_warmup = 5000
    gif = True


class RPConfig(object):
    rp_learn_start = 10000


class DQNConfig(AgentConfig, EnvironmentConfig):
    model = ''
    pass


class M1(DQNConfig, GDMConfig, RPConfig):
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
        if key == 'use_gpu':
            if FLAGS[key].value == False:
                config.cnn_format = 'NHWC'
            else:
                config.cnn_format = 'NCHW'

        if hasattr(config, key):
            setattr(config, key, FLAGS[key].value)

    return config
