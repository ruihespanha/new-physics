import numba
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from matplotlib.animation import FuncAnimation
import time
import pygame
import copy
from numba import jit


pygame.init()
size = 1000
size2 = 1000
resolution = 1
screen = pygame.display.set_mode((size2 * resolution, size * resolution))
pygame.display.set_caption("display")
clock = pygame.time.Clock()
running = True
randArr = np.random.rand(size2, size)
arr = np.zeros((size2, size))

for x in range(size2):
    for y in range(size):
        if y % 2 == 0 and (x != (size - 1) / 2 + 1 or y != ((size - 1) / 2 + 1)):
            arr[x, y] = 1


# arr[int(size2 / 2 - 0.5), int((size - 1) / 2 - 0.5) - 1] = 0


@jit(nopython=True)
def UpdateSimArr(size, arr2: np.ndarray):
    tempArr = arr2.copy()
    changed = []
    for x in range(size2):
        for y in range(size):
            old = tempArr[x, y]
            tempSum = 0
            for i in range(3):
                for j in range(3):
                    if i == 1 and j == 1:
                        tempSum += 0
                    else:
                        if (
                            x + i - 1 < 0
                            or x + i - 1 >= size2
                            or y + j - 1 < 0
                            or y + j - 1 >= size
                        ):
                            tempSum += 0
                        else:
                            if arr2[x + i - 1, y + j - 1] != 0:
                                tempSum += 1
                            else:
                                tempSum += 0
            if tempSum == 3:
                tempArr[x, y] = 1  # tempArr[x, y] + 1
            elif tempSum == 2 and arr2[x, y] != 0:
                tempArr[x, y] = 1  # tempArr[x, y] + 1
            else:
                tempArr[x, y] = 0
            if y % 2 == 0:
                tempArr[0, y] = 1
                tempArr[size2 - 1, y] = 1
            if old != tempArr[x, y]:
                changed.append((x, y))
    return tempArr, changed


def fillPix(size, arr2):
    xypos = pygame.mouse.get_pos()
    xypos = (int(xypos[0] / resolution), int(xypos[1] / resolution))
    arr2[(xypos)] = 1
    # arr2[size2 - xypos[0] - 1, xypos[1]] = 1


for x in range(size2):
    for y in range(size):
        if arr[x, y] != 0:
            if arr[x, y] > 0:
                arr[x, y] = 10
            pygame.draw.rect(
                screen,
                (25.5 * arr[x, y], 25.5 * arr[x, y], 25.5 * arr[x, y]),
                [
                    x * resolution,
                    y * resolution,
                    resolution,
                    resolution,
                ],
            )
t0 = time.time()
# for i in range(100):
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # screen.fill("black")

    pressed = pygame.mouse.get_pressed()[0]
    if pressed == True:
        fillPix(size, arr)

    if pygame.mouse.get_pressed()[2] != True:
        arr, changed = UpdateSimArr(size, arr)

    for x, y in changed:
        c = 255 * arr[x, y]
        pygame.draw.rect(
            screen,
            (c, c, c),
            [
                x * resolution,
                y * resolution,
                resolution,
                resolution,
            ],
        )

    """
    for x in range(size2):
        for y in range(size):
            if arr[x, y] != 0:
                if arr[x, y] > 0:
                    arr[x, y] = 10
                pygame.draw.rect(
                    screen,
                    (25.5 * arr[x, y], 25.5 * arr[x, y], 25.5 * arr[x, y]),
                    [
                        x * resolution,
                        y * resolution,
                        resolution,
                        resolution,
                    ],
                )
    """

    pygame.display.flip()
print(time.time() - t0)
