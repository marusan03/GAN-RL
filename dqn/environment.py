import os
import gym
from gym import wrappers
import random
import numpy as np
from PIL import Image
from .utils import rgb2gray, imresize
from collections import deque


class Environment(object):
    def __init__(self, config):
        self.env = gym.make(config.env_name)
        if config.is_train == False:
            if not os.path.exists('./video/'):
                os.makedirs('./video/')
            self.env = wrappers.Monitor(self.env, './video/', video_callable=(lambda ep: ep % 1 == 0))
        # self.env = self.env.unwrapped

        screen_width, screen_height, self.action_repeat, self.random_start = \
            config.screen_width, config.screen_height, config.action_repeat, config.random_start

        self.display = config.display
        self.dims = (screen_width, screen_height)

        # cripping
        # self._screen_buffer = deque(maxlen=2)

        self._screen = None
        self.reward = 0
        self.terminal = True

    def new_game(self, from_random_game=False):
        if self.lives == 0:
            self._screen = self.env.reset()
        self._step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    def new_random_game(self):
        self.new_game(True)
        for _ in range(random.randint(0, self.random_start - 1)):
            self._step(0)
        # cripping
        # self._screen_buffer.clear()
        self.render()
        # self._screen_buffer.append(self.screen)

        return self.screen, 0, 0, self.terminal

    def _step(self, action):
        self._screen, self.reward, self.terminal, _ = self.env.step(action)

    def _random_step(self):
        action = self.env.action_space.sample()
        self._step(action)

    @ property
    def screen(self):
        # return imresize(rgb2gray(self._screen), self.dims)
        img = np.reshape(self._screen, [210, 160, 3]).astype(np.float32)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * \
            0.587 + img[:, :, 2] * 0.114
        img = Image.fromarray(img)
        resized_screen = img.resize((84, 110), Image.BILINEAR)
        resized_screen = np.array(resized_screen)
        x_t = resized_screen[18:102, :]

        # x_t = np.reshape(x_t, [84, 84])

        # cripping
        # self._screen_buffer.append(x_t)
        # x_t = np.max(np.stack(self._screen_buffer), axis=0)
        # 2値化？
        x_t = np.reshape(x_t, [84, 84, 1])
        x_t = imresize(x_t.mean(2), (84, 84))
        return x_t

    @property
    def action_size(self):
        return self.env.action_space.n

    @property
    def lives(self):
        if self.step_info is None:
            return 0
        else:
            return self.step_info['ale.lives']

    @property
    def state(self):
        return self.screen, self.reward, self.terminal

    def render(self):
        if self.display:
            self.env.render()

    def after_act(self, action):
        self.render()


class GymEnvironment(Environment):
    def __init__(self, config):
        super(GymEnvironment, self).__init__(config)

    def act(self, action, is_training=True):
        cumulated_reward = 0
        start_lives = self.lives

        for _ in range(self.action_repeat):
            self._step(action)
            cumulated_reward = cumulated_reward + self.reward

            if is_training and start_lives > self.lives:
                cumulated_reward -= 1
                self.terminal = True

            if self.terminal:
                break

        self.reward = cumulated_reward

        self.after_act(action)
        return self.state


class SimpleGymEnvironment(Environment):
    def __init__(self, config):
        super(SimpleGymEnvironment, self).__init__(config)

    def act(self, action, is_training=True):
        self._step(action)

        self.after_act(action)
        return self.state
