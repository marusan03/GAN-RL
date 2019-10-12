import os
import random
from tqdm import tqdm
from gym import wrappers

import tensorflow as tf
import numpy as np

from dqn.environment import GymEnvironment
from dqn.history import History
from dqn.agent import Agent
from gdm import GDM


def norm_frame(obs):
    x = (obs - 127.5)/130.
    return x


def norm_frame_Q(obs):
    x = obs/255.
    return x


def unnorm_frame(obs):
    return np.clip(obs * 130. + 127.5, 0., 255.).astype(np.int32)


def play(sess, config):

    env = GymEnvironment(config)
    if not os.path.exists('./video/'):
        os.makedirs('./video/')
    env = wrappers.Monitor(
        env, './video/', video_callable=(lambda ep: ep % 1 == 0))

    epsilon = 0.1
    gan_epsilon = 0.01

    log_dir = './log/{}_lookahead_{}_gats_{}/'.format(
        config.env_name, config.lookahead, config.gats)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints/')

    config.num_actions = env.action_size
    # config.num_actions = 3

    if config.gats:
        lookahead = config.lookahead
        gdm = GDM(sess, config, num_actions=config.num_actions)
        leaves_size = config.num_actions**config.lookahead

        def base_generator():
            tree_base = np.zeros((leaves_size, lookahead)).astype('uint8')
            for i in range(leaves_size):
                n = i
                j = 0
                while n:
                    n, r = divmod(n, config.num_actions)
                    tree_base[i, lookahead-1-j] = r
                    j = j + 1
            return tree_base

        tree_base = base_generator()

    agent = Agent(sess, config, num_actions=config.num_actions)
    history = History(config)

    tf.global_variables_initializer().run()
    saver = tf.train.Saver(max_to_keep=30)

    # model load, if exist ckpt.
    load_model(sess, saver, checkpoint_dir)

    screen, reward, action, terminal = env.new_random_game()

    # init state
    for _ in range(config.history_length):
        history.add(screen)

    start_episode = 0
    end_episode = 3

    # main
    for step in tqdm(range(start_episode, end_episode), ncols=70, initial=start_episode):

        # ε-greedy
        if random.random() < epsilon:
            action = random.randrange(config.num_actions)
        else:
            current_state = np.expand_dims(history.get(), axis=0)
            if config.gats and (step >= config.gan_dqn_learn_start):
                action = MCTS_planning(
                    gdm, agent, norm_frame(current_state), leaves_size, tree_base, config, gan_epsilon)
            else:
                action = agent.get_action(
                    norm_frame_Q(current_state))

        # GATS用?
        apply_action = action
        if int(action != 0):
            apply_action = action + 1

        # Observe
        screen, reward, terminal = env.act(apply_action, is_training=True)
        reward = max(config.min_reward, min(config.max_reward, reward))
        history.add(screen)

        # reinit
        if terminal:
            screen, reward, action, terminal = env.new_game()


def MCTS_planning(gdm, agent, state, leaves_size, tree_base, config, gan_epsilon):

    sample = random.random()
    epsiron = gan_epsilon

    state = np.repeat(state, leaves_size, axis=0)
    action = tree_base
    trajectories = gdm.get_state(state, action)
    leaves_q_value = agent.get_q_value(
        norm_frame_Q(unnorm_frame(trajectories[:, -1*config.history_length:, :, :])))
    leaves_Q_max = config.discount ** (config.lookahead) * \
        np.max(leaves_q_value, axis=1)
    leaves_act_max = np.argmax(leaves_q_value, axis=1)
    if sample < epsiron:
        leaves_act_max = np.random.randint(
            0, config.num_actions, leaves_act_max.shape)
    GATS_action = leaves_Q_max
    max_idx = np.argmax(GATS_action, axis=0)
    return_action = int(tree_base[max_idx, 0])
    return return_action


def load_model(sess, saver, checkpoint_dir):
    print(" [*] Loading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        fname = os.path.join(checkpoint_dir, ckpt_name)
        saver.restore(sess, fname)
        print(" [*] Load SUCCESS: %s" % fname)
        return True
    else:
        print(" [!] Load FAILED: %s" % checkpoint_dir)
        return False
