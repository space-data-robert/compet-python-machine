import numpy as np


def make_cycle_cosine_hour(num_hour):
    return np.cos(2 * np.pi * (num_hour / 24))
