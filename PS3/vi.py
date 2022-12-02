#!/usr/bin/python

import numpy as np
from typing import Tuple
from utils import action_space, transition_function


def vi(env: np.array, goal: Tuple) -> (np.array, np.array):
    """
    env is the grid enviroment
    goal is the goal state
    """
    policy, cost_to_go = np.zeros(env.shape, 'b'), np.ones(env.shape) * 1e2
    return policy, cost_to_go
