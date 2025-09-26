import numba
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from matplotlib.animation import FuncAnimation
import time
from numba import jit
import matplotlib.cm as cm

ncols = 10
nrows = 10
t = np.linspace(0, 10, 100)
y = np.sin(t)

fig, axis = plt.subplots()
(animated_plot,) = axis.plot([], [])

axis.set_xlim([min(t), max(t)])
axis.set_ylim([-2, 2])


def update(frame):

    animated_plot.set_data(t[:frame], y[:frame])
    return animated_plot


anim = FuncAnimation(
    fig=fig,
    func=update,
    frames=None,
)

plt.show()
