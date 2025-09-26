import numba
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from matplotlib.animation import FuncAnimation
import time
from numba import jit
import matplotlib.cm as cm


# initial data
x = [1]
y = [random.randint(1, 10)]

# creating the first plot and frame
fig, ax = plt.subplots()
graph = ax.plot(x, y, color="g")[0]
plt.ylim(0, 10)


# updates the data and graph
def update(frame):
    global graph

    # updating the data
    x.append(x[-1] + 1)
    y.append(random.randint(1, 10))

    # creating a new graph or updating the graph
    graph.set_xdata(x)
    graph.set_ydata(y)
    plt.xlim(x[0], x[-1])


anim = FuncAnimation(fig, update, frames=None)
plt.show()
