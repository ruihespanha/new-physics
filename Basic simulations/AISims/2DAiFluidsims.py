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
from phi.torch.flow import *
from phi.flow import *
import numpy.typing as npt

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
torch.set_default_device(device)
print(device)


print("Among us")

# create simulation

# -define equation

# --differential fluid equation

# -define voxels

# create model

# -use pyTorch to set up AI

# -base off of equation

# -run epochs

# --calculate and print loss
loss_function = nn.MSELoss()  # mean-square error
optimizer = optim.Adam(model.parameters(), lr=0.001)

# simulate with final epoch model
