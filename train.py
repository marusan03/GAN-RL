import os
import random
from tqdm import tqdm

import tensorflow as tf
import numpy as np

from dqn.environment import GymEnvironment
from dqn.replay_memory import ReplayMemory
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


def norm_state_Q_GAN(state):
    return np.clip(state, -1*127.5/130., 127.5/130.)


def unnorm_frame(obs):
    return np.clip(obs * 130. + 127.5, 0., 255.).astype(np.int32)


def train(sess, config):

    env = GymEnvironment(config)

    model_dir = './log/{}_lookahead_{}_gats_{}/'.format(
        config.env_name, config.lookahead, config.gats)
    checkpoint_dir = os.path.join(model_dir, 'checkpoints/')
    print(' [*] checkpont_dir = {}'.format(checkpoint_dir))

    with tf.variable_scope('step'):
        step_op = tf.Variable(0, trainable=False, name='step')
        step_input = tf.placeholder('int32', None, name='step_input')
        step_assign_op = step_op.assign(step_input)

    with tf.variable_scope('summary'):
        scalar_summary_tags = ['average.reward', 'average.loss', 'average.q value',
                               'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']

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
    exploration = LinearSchedule(config.epsilon_end_t, config.epsilon_end)
    exploration_gan = LinearSchedule(50000, 0.01)

    if config.gats == True:
        lookahead = config.lookahead
        rp_train_frequency = 4
        gdm_train_frequency = 4
        gdm = GDM(sess, config, num_actions=config.num_actions)
        rp = RP(sess, config, num_actions=config.num_actions)
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
    memory = ReplayMemory(config)
    history = History(config)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=30)

    # model load, if exist ckpt.
    load_model(sess, saver, checkpoint_dir)

    agent.updated_target_q_network()

    writer = tf.summary.FileWriter(model_dir, sess.graph)

    num_game, update_count, ep_reward = 0, 0, 0.
    total_reward, total_loss, total_q_value = 0., 0., 0.
    max_avg_ep_reward = -100
    ep_rewards, actions = [], []

    screen, reward, action, terminal = env.new_random_game()

    # init state
    for _ in range(config.history_length):
        history.add(norm_frame(screen))

    start_step = step_op.eval()

    # main
    for step in tqdm(range(start_step, config.max_step), ncols=70, initial=start_step):

        if step == config.learn_start:
            num_game, update_count, ep_reward = 0, 0, 0.
            total_reward, total_loss, total_q_value = 0., 0., 0.
            ep_rewards, actions = [], []

        # ε-greedy
        epsilon = exploration.value(step)
        if random.random() < epsilon:
            action = random.randrange(env.action_size)
        else:
            if config.gats and (step >= config.gan_dqn_learn_start):
                action = MCTS_planning(
                    gdm, rp, agent, np.expand_dims(history.get(), axis=0), leaves_size, tree_base, config, exploration, step)
            else:
                action = agent.get_action(
                    norm_frame_Q(unnorm_frame(np.expand_dims(history.get(), axis=0))))

        # Observe
        screen, reward, terminal = env.act(action, is_training=True)
        reward = max(config.min_reward, min(config.max_reward, reward))
        history.add(norm_frame(screen))
        memory.add(screen, action, reward, terminal)

        # Train
        if step > config.learn_start:
            if step % config.train_frequency == 0 and memory.can_sample(config.batch_size):
                s_t, act_batch, rew_batch, s_t_plus_1, terminal_batch = memory.sample(
                    config.batch_size, config.lookahead)
                s_t, s_t_plus_1 = norm_frame_Q(s_t), norm_frame_Q(s_t_plus_1)

                q_t, loss, dqn_summary = agent.train(
                    s_t, act_batch, rew_batch, s_t_plus_1, terminal_batch, step)
                print(s_t.dtype, act_batch.dtype,
                      rew_batch.dtype, s_t_plus_1.dtype, terminal_batch.dtype)

                writer.add_summary(dqn_summary, step)
                total_loss += loss
                total_q_value += q_t.mean()
                update_count += 1

            if step % config.target_q_update_step == config.target_q_update_step - 1:
                agent.updated_target_q_network()

        if step > config.gan_learn_start and memory.can_sample(config.gan_batch_size):
            if config.gats and step % gdm_train_frequency == 0:
                state_batch, act_batch, next_state_batch = memory.GAN_sample(
                    config.gan_batch_size, config.lookahead)
                gdm.summary, disc_summary = gdm.train(
                    norm_frame(state_batch), act_batch, norm_frame(next_state_batch))
                writer.add_summary(gdm.summary, step)
                writer.add_summary(disc_summary, step)

            if config.gats and step % rp_train_frequency == 0:
                obs, act, rew = memory.reward_sample(
                    config.rp_batch_size, config.lookahead)
                reward_obs, reward_act, reward_rew = memory.nonzero_reward_sample(
                    config.rp_batch_size, lookahead)
                obs_batch = norm_frame(
                    np.concatenate((obs, reward_obs), axis=0))
                act_batch = np.concatenate((act, reward_act), axis=0)
                rew_batch = np.concatenate((rew, reward_rew), axis=0)
                reward_label = rew_batch + 1

                trajectories = gdm.get_state(
                    obs_batch[:, -1*config.history_length:, :, :], act_batch[:, :-1])

                rp_summary = rp.train(
                    trajectories, act_batch, reward_label)
                writer.add_summary(rp_summary, step)

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

        # calcurate infometion
        if step >= config.learn_start:
            if step % config._test_step == config._test_step - 1:
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
                               step + 1)  # 修正必要 モデルの保存の仕方

                    max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

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
                            'episode.actions': actions},
                        step)

                num_game = 0
                total_reward = 0.
                total_loss = 0.
                total_q_value = 0.
                update_count = 0
                ep_reward = 0.
                ep_rewards = []
                actions = []


def inject_summary(sess, writer, summary_ops, summary_placeholders, tag_dict, step):
    summary_str_lists = sess.run([summary_ops[tag] for tag in tag_dict.keys()], {
        summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
        writer.add_summary(summary_str, step)


def MCTS_planning(gdm, rp, agent, state, leaves_size, tree_base, config, exploration, step):

    sample1 = random.random()
    sample2 = random.random()
    epsiron = exploration.value(step)

    state = np.repeat(state, leaves_size, axis=0)
    action = tree_base
    trajectories = gdm.get_state(state, action)
    leaves_q_value = agent.get_q_value(
        norm_frame_Q(unnorm_frame(trajectories[:, -1*config.history_length:, :, :])))
    leaves_Q_max = config.discount ** (config.lookahead) * \
        np.max(leaves_q_value, axis=1)
    leaves_act_max = np.argmax(leaves_q_value, axis=1)
    if sample2 < epsiron:
        leaves_act_max = np.random.randint(
            0, config.num_actions, leaves_act_max.shape)
    reward_actions = np.concatenate(
        (tree_base, np.expand_dims(leaves_act_max, axis=1)), axis=1)
    predicted_cum_rew = rp.get_reward(trajectories, reward_actions)
    predicted_cum_return = np.zeros(leaves_size)
    # ここが微妙
    for i in range(config.lookahead):
        predicted_cum_return = config.discount * predicted_cum_return + \
            (np.max(predicted_cum_rew[:, ((config.lookahead-i-1)*config.num_rewards):(
                (config.lookahead-i)*config.num_rewards)], axis=1)[1]-1.)
    GATS_action = leaves_Q_max + predicted_cum_return
    max_idx = np.argmax(GATS_action, axis=0)
    return_action = int(tree_base[max_idx, 0])
    # GANのメモリに代入するコードはまた今度
    return return_action


def save_model(sess, saver, checkpoint_dir, step=None):
    print(" [*] Saving checkpoints...")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver.save(sess, checkpoint_dir, global_step=step)


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
