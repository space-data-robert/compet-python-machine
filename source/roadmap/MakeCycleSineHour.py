import numpy as np


def make_cycle_sine_hour(num_hour):
    return np.sin(2 * np.pi * (num_hour / 24))
