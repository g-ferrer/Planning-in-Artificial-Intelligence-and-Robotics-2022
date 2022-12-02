#!/usr/bin/python

import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# this is the set of possible actions admitted in this problem
action_space = [(-1, 0), (0, -1), (1, 0), (0, 1)]


def plot_joint_enviroment(env: np.array, x_e: Tuple, x_p: Tuple, goal: Tuple) -> np.array:
    """
    env is the grid enviroment
    x_e is the state of the evader
    x_p is the state of the pursuer
    goal is the goal state
    returns the copy of enviroment with the 'colored' states of the agents and goal to be plotted
    """
    current_env = np.copy(env)
    # plot evader
    current_env[x_e] = 0.9  # yellow
    # plot pursuer
    current_env[x_p] = 0.6  # cyan-ish
    # plot goal
    current_env[goal] = 0.3
    return current_env


def transition_function(env: np.array, x: Tuple, u: Tuple) -> (Tuple, bool):
    """Transition function for states in this problem
    x: current state, this is a tuple (i,j)
    u: current action, this is a tuple (i,j)
    env: enviroment
    
    Output:
    new state
    bool: True if correctly propagated
          False if this action can't be executed
    """
    xnew = np.array(x) + np.array(u)
    xnew = tuple(xnew)
    if state_consistency_check(env, xnew):
        return xnew, True
    return x, False


def state_consistency_check(env: np.array, x: Tuple) -> bool:
    """Checks wether or not the proposed state is a valid state, i.e. is in colision or our of bounds"""
    # check for collision
    if x[0] < 0 or x[1] < 0 or x[0] >= env.shape[0] or x[1] >= env.shape[1]:
        return False
    if env[x] >= 1.0 - 1e-4:
        return False
    return True


def pursuer_policy(x_e: Tuple, x_p: Tuple) -> int:
    """Returns the pursuer action"""
    ds = np.array(x_e) - np.array(x_p)
    theta = np.arctan2(ds[1], ds[0])
    theta = (theta + np.pi) / np.pi * 2
    u_index = np.floor(theta)
    delta = theta - u_index
    if np.random.rand() < delta:
        u_index += 1
    if u_index == 4:
        u_index = 0  # this is due to action 0  equals action 4 in this particular order of th action space

    return int(u_index)


def pursuer_transition(env: np.array, x_e: Tuple, x_p: Tuple) -> Tuple:
    """
    compact function for the transition function and policy for the pursuer
    5% is the probability of the pursuer to make a double move

    env - enviroment
    x_e - evader state
    x_p - pursuer state
    returns the new state of the pursuer
    """
    iters = 1
    if np.random.rand() < 0.05:
        iters = 2
    for i in range(iters):
        u_p = pursuer_policy(x_e, x_p)
        x_p, _ = transition_function(env, x_p, action_space[u_p])
    return x_p
