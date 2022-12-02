#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import plot_joint_enviroment, action_space, transition_function, pursuer_transition
from vi import vi
from tqdm import tqdm
from mcts import mcts

data = np.load('data_ps3.npz')
environment = data['environment']
# plt.matshow(environment)


# (row index, colum index). In the image row corresponds to y, and colum to x.
x_e = tuple(data["x_e"])  # (11, 6) you free to use it as np array if you prefer, but utils.py is expecting a tuple
x_p = tuple(data["x_p"])  # (7, 25)
goal = tuple(data["goal"])  # (15, 29)

# task 1 VI for evader
# ======================================
default_policy, Gopt = vi(environment, goal)

# visualize the optimal policy
plt.matshow(Gopt)

# task2 MCTS
# ======================================
# u = mcts(environment, x_e, x_p, goal, 100, default_policy)  # here is just to check

# Visualization
# ======================================
im = plot_joint_enviroment(environment, x_e, x_p, goal)
plt.matshow(im)
plt.show()

# The Game
fig = plt.figure()
imgs = []
pbar = tqdm(range(100))
for s in pbar:
    im = plot_joint_enviroment(environment, x_e, x_p, goal)
    plot = plt.imshow(im)
    imgs.append([plot])

    # according to the optimal policy of the evader, move the evader
    u_e = action_space[default_policy[x_e]]  # default_policy without taking into account the pursuer
    # u_e = mcts(environment, x_e, x_p, goal, 100, default_policy)  # taking into account pursuer, simulating the game
    x_e, _ = transition_function(environment, x_e, u_e)
    if x_e == goal:
        print('WIN!')
        break

    # propagate the pursuer: TODO uncomment the next line to release the beast
    # x_p = pursuer_transition(environment,x_e, x_p)
    if x_p == x_e:
        print('game over((')
        break
    pbar.set_description(f'x_e: {x_e}, x_p: {x_p},'
                         f' distance to goal: {np.linalg.norm(np.array(x_e) - np.array(goal)):0.2f}'
                         f' distance to pursuer: {np.linalg.norm(np.array(x_e) - np.array(x_p)):0.2f}')

im = plot_joint_enviroment(environment, x_e, x_p, goal)
plot = plt.imshow(im)
imgs.append([plot])
ani = animation.ArtistAnimation(fig, imgs, interval=100, blit=True)

ani.save('scape_solve.mp4')

plt.show()
