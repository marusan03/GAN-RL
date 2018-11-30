import time
import numpy as np
import tensorflow as tf
import sys
import pickle

try:
    from scipy.misc import imresize
except:
    import cv2
    imresize = cv2.resize


def rgb2gray(image):
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])


def normalize(state):
    return 2.*(state / 255. - 0.5)


def timeit(f):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()

        print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
        return result
    return timed


def get_time():
    return time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())


@timeit
def save_pkl(obj, path):
    with open(path, 'w') as f:
        pickle.dump(obj, f)
        print("  [*] save %s" % path)


@timeit
def load_pkl(path):
    with open(path) as f:
        obj = pickle.load(f)
        print("  [*] load %s" % path)
        return obj


@timeit
def save_npy(obj, path):
    np.save(path, obj)
    print("  [*] save %s" % path)


@timeit
def load_npy(path):
    obj = np.load(path)
    print("  [*] load %s" % path)
    return obj


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
