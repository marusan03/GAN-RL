import os
import shutil
import random
from tqdm import tqdm
from PIL import Image

import tensorflow as tf
import numpy as np

from dqn.environment import GymEnvironment
from dqn.replay_memory import ReplayMemory, GANReplayMemory
from dqn.history import History
from dqn.agent import Agent
from gdm import GDM
from rp import RP
from dqn.utils import LinearSchedule


def norm_frame(obs):
    x = (obs - 127.5)/130.
    return x


def norm_frame_Q(obs):
    x = obs/255.
    return x


def unnorm_frame(obs):
    return np.clip(obs * 130. + 127.5, 0., 255.).astype(np.int32)


def train(sess, config):

    env = GymEnvironment(config)

    log_dir = './log/{}_lookahead_{}_gats_{}/'.format(
        config.env_name, config.lookahead, config.gats)
    checkpoint_dir = os.path.join(log_dir, 'checkpoints/')
    image_dir = os.path.join(log_dir, 'rollout/')
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
        print(' [*] Removed log dir: ' + log_dir)

    with tf.variable_scope('step'):
        step_op = tf.Variable(0, trainable=False, name='step')
        step_input = tf.placeholder('int32', None, name='step_input')
        step_assign_op = step_op.assign(step_input)

    with tf.variable_scope('summary'):
        scalar_summary_tags = [
            'average.reward',
            'average.loss',
            'average.q value',
            'episode.max reward',
            'episode.min reward',
            'episode.avg reward',
            'episode.num of game',
            'training.learning_rate',
            'rp.rp_accuracy',
            'rp.rp_plus_accuracy',
            'rp.rp_minus_accuracy',
            'rp.nonzero_rp_accuracy'
        ]

        summary_placeholders = {}
        summary_ops = {}

        for tag in scalar_summary_tags:
            summary_placeholders[tag] = tf.placeholder(
                'float32', None, name=tag.replace(' ', '_'))
            summary_ops[tag] = tf.summary.scalar(
                "%s-%s/%s" % (config.env_name, config.env_type, tag), summary_placeholders[tag])

        histogram_summary_tags = ['episode.rewards', 'episode.actions']

        for tag in histogram_summary_tags:
            summary_placeholders[tag] = tf.placeholder(
                'float32', None, name=tag.replace(' ', '_'))
            summary_ops[tag] = tf.summary.histogram(
                tag, summary_placeholders[tag])

    config.num_actions = env.action_size
    # config.num_actions = 3

    exploration = LinearSchedule(config.epsilon_end_t, config.epsilon_end)

    agent = Agent(sess, config, num_actions=config.num_actions)

    if config.gats:
        lookahead = config.lookahead
        rp_train_frequency = 4
        gdm_train_frequency = 4
        gdm = GDM(sess, config, num_actions=config.num_actions)
        rp = RP(sess, config, num_actions=config.num_actions)
        leaves_size = config.num_actions**config.lookahead
        if config.dyna:
            gan_memory = GANReplayMemory(config)
        else:
            gan_memory = None

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

    # memory = ReplayMemory(config)
    memory = ReplayMemory(config, log_dir)
    history = History(config)

    tf.global_variables_initializer().run()
    saver = tf.train.Saver(max_to_keep=30)

    # model load, if exist ckpt.
    load_model(sess, saver, checkpoint_dir)

    agent.updated_target_q_network()

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    num_game, update_count, ep_reward = 0, 0, 0.
    total_reward, total_loss, total_q_value = 0., 0., 0.
    max_avg_ep_reward = -100
    ep_rewards, actions = [], []

    rp_accuracy = []
    rp_plus_accuracy = []
    rp_minus_accuracy = []
    nonzero_rp_accuracy = []

    screen, reward, action, terminal = env.new_random_game()

    # init state
    for _ in range(config.history_length):
        history.add(screen)

    start_step = step_op.eval()

    # main
    for step in tqdm(range(start_step, config.max_step), ncols=70, initial=start_step):

        if step == config.learn_start:
            num_game, update_count, ep_reward = 0, 0, 0.
            total_reward, total_loss, total_q_value = 0., 0., 0.
            ep_rewards, actions = [], []

        if step == config.gan_dqn_learn_start:
            rp_accuracy = []
            rp_plus_accuracy = []
            rp_minus_accuracy = []
            nonzero_rp_accuracy = []

        # ε-greedy
        MCTS_FLAG = False
        epsilon = exploration.value(step)
        if random.random() < epsilon:
            action = random.randrange(config.num_actions)
        else:
            current_state = norm_frame(np.expand_dims(history.get(), axis=0))
            if config.gats and (step >= config.gan_dqn_learn_start):
                action, predicted_reward = MCTS_planning(
                    gdm, rp, agent, current_state, leaves_size, tree_base, config, exploration, step, gan_memory)
                MCTS_FLAG = True
            else:
                action = agent.get_action(
                    norm_frame_Q(unnorm_frame(current_state)))

        # GATS用?
        apply_action = action
        # if int(apply_action != 0):
        #     apply_action += 1

        # Observe
        screen, reward, terminal = env.act(apply_action, is_training=True)
        reward = max(config.min_reward, min(config.max_reward, reward))
        history.add(screen)
        memory.add(screen, reward, action, terminal)

        if MCTS_FLAG:
            rp_accuracy.append(int(predicted_reward == reward))
            if reward != 0:
                nonzero_rp_accuracy.append(int(predicted_reward == reward))
                if reward == 1:
                    rp_plus_accuracy.append(int(predicted_reward == reward))
                elif reward == -1:
                    rp_minus_accuracy.append(int(predicted_reward == reward))

        # Train
        if step > config.gan_learn_start and config.gats:
            if step % rp_train_frequency == 0 and memory.can_sample(config.rp_batch_size):
                obs, act, rew = memory.reward_sample(
                    config.rp_batch_size)
                # obs, act, rew = memory.reward_sample2(
                #     config.rp_batch_size, config.lookahead)
                reward_obs, reward_act, reward_rew = memory.reward_sample(
                    config.nonzero_batch_size, nonzero=True)
                # reward_obs, reward_act, reward_rew = memory.nonzero_reward_sample(
                #     config.rp_batch_size, config.lookahead)
                obs_batch = norm_frame(
                    np.concatenate((obs, reward_obs), axis=0))
                act_batch = np.concatenate((act, reward_act), axis=0)
                rew_batch = np.concatenate((rew, reward_rew), axis=0)
                reward_label = rew_batch + 1

                trajectories = gdm.get_state(
                    obs_batch, act_batch[:, :-1])

                rp_summary = rp.train(
                    trajectories, act_batch, reward_label)
                writer.add_summary(rp_summary, step)

            if step % gdm_train_frequency == 0 and memory.can_sample(config.gan_batch_size):
                state_batch, action_batch, next_state_batch = memory.GAN_sample()
                # state_batch, act_batch, next_state_batch = memory.GAN_sample2(
                #     config.gan_batch_size, config.lookahead)

                # gdm.summary, disc_summary, merged_summary = gdm.train(
                #     norm_frame(state_batch), act_batch, norm_frame(next_state_batch), warmup_bool)
                gdm.summary, disc_summary = gdm.train(
                    norm_frame(state_batch), action_batch, norm_frame(next_state_batch))

        if step > config.learn_start:
            # if step % config.train_frequency == 0 and memory.can_sample(config.batch_size):
            if step % config.train_frequency == 0:
                # s_t, act_batch, rew_batch, s_t_plus_1, terminal_batch = memory.sample(
                #     config.batch_size, config.lookahead)
                s_t, act_batch, rew_batch, s_t_plus_1, terminal_batch = memory.sample()
                s_t, s_t_plus_1 = norm_frame(s_t), norm_frame(s_t_plus_1)
                if config.gats and config.dyna:
                    if step > config.gan_dqn_learn_start and gan_memory.can_sample(config.batch_size):
                        gan_obs_batch, gan_act_batch, gan_rew_batch, gan_terminal_batch = gan_memory.sample()
                        # gan_obs_batch, gan_act_batch, gan_rew_batch = gan_memory.sample(
                        #     config.batch_size)
                        gan_obs_batch = norm_frame(gan_obs_batch)
                        trajectories = gdm.get_state(
                            gan_obs_batch, np.expand_dims(gan_act_batch, axis=1))
                        gan_next_obs_batch = trajectories[:,
                                                          -config.history_length:, ...]

                        # gan_obs_batch, gan_next_obs_batch = \
                        #     norm_frame(gan_obs_batch), norm_frame(gan_next_obs_batch)

                        s_t = np.concatenate([s_t, gan_obs_batch], axis=0)
                        act_batch = np.concatenate(
                            [act_batch, gan_act_batch], axis=0)
                        rew_batch = np.concatenate(
                            [rew_batch, gan_rew_batch], axis=0)
                        s_t_plus_1 = np.concatenate(
                            [s_t_plus_1, gan_next_obs_batch], axis=0)
                        terminal_batch = np.concatenate(
                            [terminal_batch, gan_terminal_batch], axis=0)

                s_t, s_t_plus_1 = norm_frame_Q(unnorm_frame(
                    s_t)), norm_frame_Q(unnorm_frame(s_t_plus_1))

                q_t, loss, dqn_summary = agent.train(
                    s_t, act_batch, rew_batch, s_t_plus_1, terminal_batch, step)

                writer.add_summary(dqn_summary, step)
                total_loss += loss
                total_q_value += q_t.mean()
                update_count += 1

            if step % config.target_q_update_step == config.target_q_update_step - 1:
                agent.updated_target_q_network()

        # reinit
        if terminal:
            screen, reward, action, terminal = env.new_random_game()

            num_game += 1
            ep_rewards.append(ep_reward)
            ep_reward = 0.
        else:
            ep_reward += reward

        total_reward += reward

        # change train freqancy
        if config.gats:
            if step == 10000 - 1:
                rp_train_frequency = 8
                gdm_train_frequency = 8
            if step == 50000 - 1:
                rp_train_frequency = 16
                gdm_train_frequency = 16
            if step == 100000 - 1:
                rp_train_frequency = 24
                gdm_train_frequency = 24

        # rolloutを行い画像を保存
        if config.gats and step % config._test_step == config._test_step - 1:
            rollout_image(config, image_dir, gdm, memory, step+1, 16)

        # calcurate infometion
        if step >= config.learn_start:
            if step % config._test_step == config._test_step - 1:

                # plot
                if config.gats:
                    writer.add_summary(gdm.summary, step)
                    writer.add_summary(disc_summary, step)

                avg_reward = total_reward / config._test_step
                avg_loss = total_loss / update_count
                avg_q = total_q_value / update_count

                try:
                    max_ep_reward = np.max(ep_rewards)
                    min_ep_reward = np.min(ep_rewards)
                    avg_ep_reward = np.mean(ep_rewards)
                except:
                    max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

                print('\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d'
                      % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game))

                # require terget q network
                if max_avg_ep_reward * 0.9 <= avg_ep_reward:
                    step_assign_op.eval({step_input: step + 1})
                    save_model(sess, saver, checkpoint_dir,
                               step + 1)

                    max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

                if step >= config.gan_dqn_learn_start:
                    if len(rp_accuracy) > 0:
                        rp_accuracy = np.mean(rp_accuracy)
                        rp_plus_accuracy = np.mean(rp_plus_accuracy)
                        rp_minus_accuracy = np.mean(rp_minus_accuracy)
                        nonzero_rp_accuracy = np.mean(nonzero_rp_accuracy)
                    else:
                        rp_accuracy = 0
                        rp_plus_accuracy = 0
                        rp_minus_accuracy = 0
                        nonzero_rp_accuracy = 0
                else:
                    rp_accuracy = 0
                    rp_plus_accuracy = 0
                    rp_minus_accuracy = 0
                    nonzero_rp_accuracy = 0

                # summary
                if step > 180:
                    inject_summary(
                        sess, writer, summary_ops, summary_placeholders,
                        {
                            'average.reward': avg_reward,
                            'average.loss': avg_loss,
                            'average.q value': avg_q,
                            'episode.max reward': max_ep_reward,
                            'episode.min reward': min_ep_reward,
                            'episode.avg reward': avg_ep_reward,
                            'episode.num of game': num_game,
                            'episode.rewards': ep_rewards,
                            'episode.actions': actions,
                            'rp.rp_accuracy': rp_accuracy,
                            'rp.rp_plus_accuracy': rp_plus_accuracy,
                            'rp.rp_minus_accuracy': rp_minus_accuracy,
                            'rp.nonzero_rp_accuracy': nonzero_rp_accuracy
                        },
                        step)

                num_game = 0
                total_reward = 0.
                total_loss = 0.
                total_q_value = 0.
                update_count = 0
                ep_reward = 0.
                ep_rewards = []
                actions = []

                rp_accuracy = []
                rp_plus_accuracy = []
                rp_minus_accuracy = []
                nonzero_rp_accuracy = []


def inject_summary(sess, writer, summary_ops, summary_placeholders, tag_dict, step):
    summary_str_lists = sess.run([summary_ops[tag] for tag in tag_dict.keys()], {
        summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
        writer.add_summary(summary_str, step)


def MCTS_planning(gdm, rp, agent, state, leaves_size, tree_base, config, exploration, step, gan_memory=None):

    sample1 = random.random()
    sample2 = random.random()
    epsiron = exploration.value(step)

    state = np.repeat(state, leaves_size, axis=0)
    action = tree_base
    trajectories = gdm.get_state(state, action)
    leaves_q_value = agent.get_q_value(
        norm_frame_Q(unnorm_frame(trajectories[:, -config.history_length:, ...])))
    leaves_Q_max = (config.discount ** config.lookahead) * \
        np.max(leaves_q_value, axis=1)
    leaves_act_max = np.argmax(leaves_q_value, axis=1)
    if sample2 < epsiron:
        leaves_act_max = np.random.randint(
            0, config.num_actions, leaves_act_max.shape)
    reward_actions = np.concatenate(
        (tree_base, np.expand_dims(leaves_act_max, axis=1)), axis=1)
    predicted_cum_rew = rp.get_reward(trajectories, reward_actions)
    predicted_cum_return = np.zeros(leaves_size)
    for i in range(config.lookahead):
        predicted_cum_return = config.discount * predicted_cum_return + \
            (np.argmax(predicted_cum_rew[:, ((config.lookahead-i-1)*config.num_rewards):(
                (config.lookahead-i)*config.num_rewards)], axis=1)-1.)
    GATS_action = leaves_Q_max + predicted_cum_return
    max_idx = np.argmax(GATS_action, axis=0)
    predicted_reward = np.argmax(
        predicted_cum_rew[max_idx, 0:config.num_rewards], axis=0) - 1
    return_action = int(tree_base[max_idx, 0])
    # Dyna-Q
    if gan_memory:
        if sample1 < epsiron:
            max_idx = random.randrange(leaves_size)
        obs = unnorm_frame(trajectories[max_idx, -config.history_length:, ...])
        act_batch = np.squeeze(leaves_act_max[max_idx])
        rew_batch = np.argmax(
            predicted_cum_rew[max_idx, -config.num_rewards:], axis=0) - 1
        gan_memory.add_batch(obs, act_batch, rew_batch)
    return return_action, predicted_reward


def rollout_image(config, image_dir, gdm, memory, step, num_rollout=4):
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)
    states, actions = memory.rollout_state_action(num_rollout)
    states = norm_frame(states)
    actions = np.concatenate([actions, [0] * (config.lookahead - 1)])
    images = gdm.rollout(np.expand_dims(
        states[:4], axis=0), actions, num_rollout)
    action_label = [str(action) for action in actions]
    action_label = '.'.join(action_label)
    if config.gif:
        gif_images = np.concatenate([states, images], axis=1)
        pil_image = [Image.fromarray(np.uint8(unnorm_frame(image))).convert(mode='L')
                     for image in gif_images]
        pil_image[0].save(
            (image_dir + f'rollout_{step}_{action_label}.gif'), save_all=True, append_images=pil_image[1:], optimize=True, duration=100, loop=0)
    states = np.hstack(states)
    images = np.hstack(images)
    states = np.vstack([states, images])
    pil_image = Image.fromarray(unnorm_frame(states))
    pil_image.convert(mode='L').save(
        image_dir + 'rollout_{}_{}.jpg'.format(step, action_label))
    print('\n [*] created Image!')


def save_model(sess, saver, checkpoint_dir, step=None):
    print(" [*] Saving checkpoints...")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver.save(sess, checkpoint_dir+'model', global_step=step)


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
