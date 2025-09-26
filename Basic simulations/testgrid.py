import numba
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import time
from numba import jit
import matplotlib.cm as cm
import functools

ncols, nrows = 100, 100

grid = np.random.rand(ncols, nrows) > 0.5


"""@jit(nopython=True)
def updateGrid(grid1):
    tempGrid = grid1
    for x in range(ncols):
        for y in range(nrows):
            if grid1[x, y] == 1 and y >= 1 and grid1[x, y - 1] == 0:
                tempGrid[x, y] = 0
                tempGrid[x, y - 1] = 1
    return tempGrid
"""


@jit(nopython=True)
def updateGrid(arr2: np.ndarray):
    tempArr = arr2.copy()
    for x in range(ncols):
        for y in range(nrows):
            tempSum = 0
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        tempSum += 0
                    else:
                        if (
                            x + i - 1 < 0
                            or x + i - 1 >= ncols
                            or y + j - 1 < 0
                            or y + j - 1 >= nrows
                        ):
                            tempSum += 0
                        else:
                            if arr2[x + i - 1, y + j - 1] != 0:
                                tempSum += 1
                            else:
                                tempSum += 0
            if tempSum == 3:
                tempArr[x, y] = tempArr[x, y] + 1
            elif tempSum == 2 and arr2[x, y] != 0:
                tempArr[x, y] = tempArr[x, y] + 1
            else:
                tempArr[x, y] = 0
            if y % 2 == 0:
                tempArr[0, y] = 10
                tempArr[ncols - 1, y] = 10
    return tempArr


def update(data, grid2, mat3):
    global grid
    newGrid = updateGrid(grid)
    print(newGrid)
    # update data
    mat3.set_data(newGrid)
    grid = newGrid


plt.close("all")
fig, ax = plt.subplots()
mat = ax.matshow(grid)


ani = animation.FuncAnimation(
    fig,
    functools.partial(update, grid2=grid, mat3=mat),
    interval=0,
    cache_frame_data=False,
)

plt.show()
