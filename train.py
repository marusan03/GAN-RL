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


def train(sess, config):

    env = GymEnvironment(config)

    model_dir = './log/{}_lookahead_{}_gats_{}'.format(
        config.env_name, config.lookahead, config.gats)
    checkpoint_dir = os.path.join(model_dir, 'checkpoints')
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

    if config.gats == True:
        gdm = GDM(sess, config, num_actions=env.action_size)
    agent = Agent(sess, config, num_actions=env.action_size)
    memory = ReplayMemory(config, model_dir)
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

    # 初期状態
    for _ in range(config.history_length):
        history.add(screen)

    start_step = step_op.eval()

    for step in tqdm(range(start_step, config.max_step), ncols=70, initial=start_step):

        if step == config.learn_start:
            num_game, update_count, ep_reward = 0, 0, 0.
            total_reward, total_loss, total_q_value = 0., 0., 0.
            ep_rewards, actions = [], []

        # ε-greedy
        epsilon = (config.epsilon_end +
                   max(0., (config.epsilon_start - config.epsilon_end)
                       * (config.epsilon_end_t - max(0., step - config.learn_start)) / config.epsilon_end_t))
        if random.random() < epsilon:
            action = random.randrange(env.action_size)
        else:
            if config.gats:
                # GATS
                # とりあえずlookahead=1 の時のみ
                history_state = history.get()
                for j in range(0, env.action_size):
                    predict_state = gdm.get_state([history_state], [[j]])
                    q_value = agent.get_q_value(predict_state[:, 1:5, ...])
                    if j == 0 or max_q_value < np.max(q_value):
                        action = j
                        max_q_value = np.max(q_value)
            else:
                action = agent.get_action([history.get()])

        # Observe
        screen, reward, terminal = env.act(action, is_training=True)
        reward = max(config.min_reward, min(config.max_reward, reward))
        history.add(screen)
        memory.add(screen, reward, action, terminal)

        # Train
        if step > config.learn_start:
            if step % config.train_frequency == 0:
                s_t, action_batch, reward_batch, s_t_plus_1, terminal_batch = memory.sample()

                q_t, loss, dqn_summary = agent.train(
                    s_t, action_batch, reward_batch, s_t_plus_1, terminal_batch, step)

                writer.add_summary(dqn_summary, step)
                total_loss += loss
                total_q_value += q_t.mean()
                update_count += 1

            if step % config.target_q_update_step == config.target_q_update_step - 1:
                agent.updated_target_q_network()

            if config.gats and step % config.gdm_train_frequency == 0:
                gdm.summary, disc_summary = gdm.train(
                    s_t, np.reshape(
                        action_batch, [-1, 1]), np.reshape(s_t_plus_1[:, 3, ...], [-1, 1, 84, 84]))
                writer.add_summary(gdm.summary, step)
                writer.add_summary(disc_summary, step)

        # Reinit
        if terminal:
            screen, reward, action, terminal = env.new_random_game()

            num_game += 1
            ep_rewards.append(ep_reward)
            ep_reward = 0.
        else:
            ep_reward += reward

        total_reward += reward

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
