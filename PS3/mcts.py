#!/usr/bin/python

import numpy as np
from typing import Tuple
from utils import action_space, transition_function, pursuer_transition


def mcts(env: np.array, x_e: Tuple, x_p: Tuple, goal: Tuple, k_budget, default_policy) -> Tuple:
    """
    Monte-Carlo tree search
    env is the grid enviroment
    x_e evader
    x_p pursuer
    goal is the goal state
    """
    u = ...
    return u
