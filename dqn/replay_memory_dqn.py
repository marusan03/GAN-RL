"""Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import os
import random
import logging
import numpy as np

from .utils import save_npy, load_npy


class GANReplayMemory(object):
    def __init__(self, config):
        self.cnn_format = config.cnn_format
        self.memory_size = config.gan_memory_size
        self.history_length = config.history_length
        self.dims = (config.screen_height, config.screen_width)
        self.batch_size = config.batch_size
        self.count = 0
        self.current = 0

        self.states = np.empty(
            (self.memory_size, self.history_length) + self.dims, dtype=np.float32)
        self.actions = np.empty([self.memory_size], dtype=np.uint8)
        self.rewards = np.empty([self.memory_size], dtype=np.integer)
        self.terminals = np.full([self.batch_size], False)
        # pre-allocate prestates for minibatch
        self.prestates = np.empty(
            (self.batch_size, self.history_length) + self.dims, dtype=np.float32)

    def add_batch(self, frames, act, rew):
        self.states[self.current] = frames
        self.actions[self.current] = act
        self.rewards[self.current] = rew

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.count

    def sample(self):
            # memory must include poststate, prestate and history
        assert self.count > self.history_length
        # sample random indexes
        indexes = np.random.randint(
            self.history_length, self.count - 1, (self.batch_size))

        self.prestates = self.states[indexes]
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]

        if self.cnn_format == 'NHWC':
            return np.transpose(self.prestates, (0, 2, 3, 1)), actions, rewards, self.terminals
        else:
            return self.prestates, actions, rewards, self.terminals


class ReplayMemory:
    def __init__(self, config, model_dir):
        self.model_dir = model_dir

        self.cnn_format = config.cnn_format
        self.memory_size = config.memory_size
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.integer)
        self.screens = np.empty(
            (self.memory_size, config.screen_height, config.screen_width), dtype=np.uint8)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)
        self.history_length = config.history_length
        self.dims = (config.screen_height, config.screen_width)
        self.batch_size = config.batch_size
        self.gan_batch_size = config.gan_batch_size
        self.rp_batch_size = config.rp_batch_size
        self.lookahead = config.lookahead
        self.count = 0
        self.current = 0
        # reward predictor
        self.nonzero_rewards = []
        self.overwrite_index = None

        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty(
            (self.batch_size, self.history_length) + self.dims, dtype=np.uint8)
        self.poststates = np.empty(
            (self.batch_size, self.history_length) + self.dims, dtype=np.uint8)
        self.gan_states = np.empty(
            (self.gan_batch_size, self.history_length + self.lookahead) + self.dims, dtype=np.uint8)
        self.reward_states = np.empty(
            (self.rp_batch_size, self.history_length) + self.dims, dtype=np.uint8)

    def add(self, screen, reward, action, terminal):
        assert screen.shape == self.dims
        # NB! screen is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current, ...] = screen
        self.terminals[self.current] = terminal

        if(self.overwrite_index != None and self.current == self.nonzero_rewards[self.overwrite_index]):
            self.nonzero_rewards.pop(self.overwrite_index)
            if self.overwrite_index >= len(self.nonzero_rewards):
                self.overwrite_index = None

        if (self.current + 1) >= self.memory_size and len(self.nonzero_rewards):
            self.overwrite_index = 0

        if (reward != 0):
            if self.overwrite_index == None:
                self.nonzero_rewards.append(self.current)
            else:
                self.nonzero_rewards.insert(self.overwrite_index, self.current)
                self.overwrite_index += 1

        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def getState(self, index, lookahead=0):
        assert self.count > 0, "replay memory is empty, use at least --random_steps 1"
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        if index >= self.history_length - 1:
            # use faster slicing
            return self.screens[(index - (self.history_length - 1)):(index + 1 + lookahead), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) %
                       self.count for i in reversed(range(self.history_length + lookahead))]
            return self.screens[indexes, ...]

    def sample(self):
        # memory must include poststate, prestate and history
        assert self.count > self.history_length
        # sample random indexes
        indexes = []
        while len(indexes) < self.batch_size:
            # find random index
            while True:
                # sample one index (ignore states wraping over
                index = random.randint(self.history_length, self.count - 1)
                # if wraps over current pointer, then get new one
                if index >= self.current and index - self.history_length < self.current:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.terminals[(index - self.history_length):index].any():
                    continue
                # otherwise use this index
                break

            # NB! having index first is fastest in C-order matrices
            self.prestates[len(indexes), ...] = self.getState(index - 1)
            self.poststates[len(indexes), ...] = self.getState(index)
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        if self.cnn_format == 'NHWC':
            return np.transpose(self.prestates, (0, 2, 3, 1)), actions, \
                rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals
        else:
            return self.prestates, actions, rewards, self.poststates, terminals

    def GAN_sample(self):
        assert self.count > self.gan_batch_size

        indexes = []
        while len(indexes) < self.gan_batch_size:
            # find random index
            while True:
                # sample one index (ignore states wraping over
                index = random.randint(
                    self.history_length + self.lookahead, self.count - (1 + self.lookahead))
                # if wraps over current pointer, then get new one
                if index >= self.current and index - (self.history_length + self.lookahead) < self.current:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.terminals[(index - (self.history_length + self.lookahead)):index].any():
                    continue
                # otherwise use this index
                break

            # NB! having index first is fastest in C-order matrices
            self.gan_states[len(indexes), ...] = self.getState(
                index - 1, self.lookahead)
            indexes.append(index)

        if self.lookahead == 1:
            actions = np.expand_dims(self.actions[indexes], axis=1)
        else:
            actions = [self.actions[i:i+self.lookahead] for i in indexes]

        if self.cnn_format == 'NHWC':
            return np.transpose(self.gan_states[:, :self.history_length, ...], (0, 2, 3, 1)), actions, np.transpose(self.gan_states[:, self.history_length:, ...], (0, 2, 3, 1))
        else:
            return self.gan_states[:, :self.history_length, ...], actions, self.gan_states[:, self.history_length:, ...]

    def reward_sample(self, nonzero=False):
        assert self.count > self.rp_batch_size

        indexes = []
        while len(indexes) < self.rp_batch_size:
            # find random index
            while True:
                # sample one index (ignore states wraping over
                if nonzero == True and (len(self.nonzero_rewards) > 0):
                    index = np.random.choice(
                        self.nonzero_rewards, size=1)[0] - random.randint(0, self.lookahead)
                else:
                    index = random.randint(
                        self.history_length + self.lookahead + 1, self.count - (2 + self.lookahead))
                # if wraps over current pointer, then get new one
                if index >= self.current and index - (self.history_length + self.lookahead) < self.current:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                if self.terminals[(index - (self.history_length + self.lookahead)):index].any():
                    continue
                # otherwise use this index
                break

            # NB! having index first is fastest in C-order matrices
            self.reward_states[len(indexes), ...] = self.getState(index - 1)
            indexes.append(index)

        actions = [self.actions[i:i+self.lookahead+1] for i in indexes]
        rewards = [self.rewards[i:i+self.lookahead+1] for i in indexes]

        if self.cnn_format == 'NHWC':
            return np.transpose(self.reward_states, (0, 2, 3, 1)), actions, rewards
        else:
            return self.reward_states, actions, rewards

    # test code

    def reward_sample2(self, batch_size, lookahead):
        def sample_n_unique(sampling_f, n):
            res = []
            while len(res) < n:
                candidate = sampling_f()
                # print(candidate)
                if candidate not in res:
                    res.append(candidate)
            return res
        assert self.can_sample(lookahead)
        # idxes = sample_n_unique(lambda: random.randint(lookahead, self.current - 2 - lookahead), batch_size)
        idxes = sample_n_unique(lambda: (self.count-random.randint(lookahead+self.history_length, 60000)) %
                                (self.current-lookahead-self.history_length), batch_size)
        return self.reward_encode_sample(idxes, lookahead)

    def nonzero_reward_sample(self, batch_size, lookahead):
        # assert self.can_sample_nonzero_rewards(lookahead)
        # nonzero_idxes = np.random.choice(self.nonzero_rewards, size=batch_size)
        idxes = [self.get_rand_nonzero_idx(lookahead)
                 for i in range(batch_size)]
        return self.reward_encode_sample(idxes, lookahead)

    def reward_encode_sample(self, idxes, lookahead=1):
        self.reward_states = [self.getState(idx - 1) for idx in idxes]
        seq = [self._encode_reward_action(idx + 1, lookahead) for idx in idxes]
        act_batch = np.concatenate(
            [seq[i][0][np.newaxis, :, 0] for i in range(len(idxes))], 0)
        rew_batch = np.concatenate([seq[i][1][np.newaxis, :, 0]
                                    for i in range(len(idxes))], 0)
        return self.reward_states, act_batch, rew_batch

    def get_rand_nonzero_idx(self, lookahead=1):
        nonzero_idx = np.random.choice(self.nonzero_rewards, size=1)[
            0] - random.randint(0, lookahead)
        while nonzero_idx % (self.current-lookahead-2) != nonzero_idx:
            nonzero_idx = np.random.choice(self.nonzero_rewards, size=1)[
                0] - random.randint(0, lookahead)
        start_idx = nonzero_idx
        # for idx in range(start_idx, nonzero_idx):
        #     if self.terminal[idx % self.memory_size]:
        #         start_idx = idx + 1
        return start_idx

    def _encode_reward_action(self, idx, lookahead):
        end_idx = idx + lookahead + 1  # make noninclusive
        start_idx = idx
        if start_idx < 0 and self.current != self.memory_size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.terminals[idx % self.current]:
                start_idx = idx + 1
        missing_context = (lookahead + 1) - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            action = [0 for _ in range(missing_context)]
            reward = [0 for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                action.append(self.actions[(idx-1) % self.current])
                reward.append(self.rewards[(idx-1) % self.current])
            return np.asarray(action).reshape(-1, 1), np.asarray(reward).reshape(-1, 1)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            return self.actions[start_idx-1:end_idx-1].reshape(-1, 1), self.rewards[start_idx-1:end_idx-1].reshape(-1, 1)

    def can_sample(self, batch_size):
        return batch_size + 1 <= self.count

    def save(self):
        for idx, (name, array) in enumerate(
            zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
                [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
            save_npy(array, os.path.join(self.model_dir, name))

    def load(self):
        for idx, (name, array) in enumerate(
            zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
                [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
            array = load_npy(os.path.join(self.model_dir, name))
