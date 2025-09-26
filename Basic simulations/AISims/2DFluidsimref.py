from phi.torch.flow import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import time

import math
from numba import jit
import matplotlib.pyplot as plt
import dash
from phi.flow import *
import numpy.typing as npt

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# from phi.flow import *  # If JAX is not installed. You can use phi.torch or phi.tf as well.
from tqdm.notebook import trange


def potential(pos):
    return math.cos(math.vec_length(pos))


landscape = CenteredGrid(potential, x=100, y=100, bounds=Box(x=(-5, 5), y=(-5, 5)))
plt.plot(landscape)

math.seed(0)
net = dense_net(2, 1, [32, 64, 32])
optimizer = adam(net)


def loss_function(x, label):
    prediction = math.native_call(net, x)
    return math.l2_loss(prediction - label), prediction


input_data = rename_dims(landscape.points, spatial, batch)
labels = rename_dims(landscape.values, spatial, batch)
loss_function(input_data, labels)[0]

loss_trj = []
pred_trj = []
for i in range(201):
    if i % 10 == 0:
        print(i)
    loss, pred = update_weights(net, optimizer, loss_function, input_data, labels)
    loss_trj.append(loss)
    pred_trj.append(pred)
loss_trj = stack(loss_trj, spatial("iteration"))
pred_trj = stack(pred_trj, batch("iteration"))
plt.plot(math.mean(loss_trj, "x,y"), err=math.std(loss_trj, "x,y"), size=(4, 3))

# pred_grid = rename_dims(pred_trj.iteration[::4], "x,y", spatial)
# plt. plot(pred_grid, animate="iteration", size=(6, 5))
